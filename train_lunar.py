# /// script
# requires-python = ">= 3.10"
# dependencies = [
#     "supermimic-pytorch",
#     "mean-conc-beta>=0.2.11",
#     "gymnasium[box2d]",
#     "ema-pytorch>=0.8.3",
#     "einops>=0.8.1",
#     "x-mlps-pytorch>=0.6.14",
#     "fire",
#     "imageio",
#     "matplotlib",
#     "numpy",
# ]
# ///

from __future__ import annotations

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

from collections import deque
from pathlib import Path

import fire
import imageio
import numpy as np
import gymnasium as gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import Tensor, as_tensor, arange, cat, full, randint, tensor, zeros
from torch.nn import Module, SiLU, init
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from einops import rearrange, repeat
from torch_einops_utils import temp_eval, tree_map_tensor_to_device
from ema_pytorch import EMA
from mean_conc_beta import Beta
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP

from supermimic_pytorch import WorldModel, Critic, SuperMimic

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

# actor

class Actor(Module):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        bounds = (-1., 1.),
        dim_hidden = 256,
        depth = 3
    ):
        super().__init__()
        self.net = NormedMLP(dim_state, *([dim_hidden] * depth), dim_action * 2, activation = SiLU())

        last_linear = self.net.layers[-1][0]
        init.orthogonal_(last_linear.weight, gain = 0.01)
        init.zeros_(last_linear.bias)

        self.beta = Beta(
            bounds = bounds,
            pos_fn = 'exp',
            init_conc = 40.,
            clamp_exp = (-2.2, 3.),
            detach_entropy_mean = True
        )

    def params(self, state):
        return rearrange(self.net(state), '... (d p) -> ... d p', p = 2)

    def entropy(self, state):
        return self.beta.entropy(self.params(state))

    def forward(self, state):
        params = self.params(state)
        return self.beta.rsample(params) if self.training else self.beta.mean(params)

# replay buffer

class ReplayBuffer:
    def __init__(
        self,
        num_envs: int,
        capacity: int,
        dim_state: int,
        dim_action: int,
        action_low,
        action_high,
        absorbing_steps = 7
    ):
        shape = (num_envs, capacity)
        self.capacity = capacity
        self.absorbing_steps = absorbing_steps
        self.action_low = action_low
        self.action_high = action_high

        self.states = zeros((*shape, dim_state))
        self.actions = zeros((*shape, dim_action))
        self.next_states = zeros((*shape, dim_state))
        self.rewards = zeros(shape)
        self.terminals = zeros(shape)
        self.episode_ids = full(shape, -1, dtype = torch.long)
        self.is_real = zeros(shape, dtype = torch.bool)

        self.write_ptrs = [0] * num_envs
        self.is_full = False
        self.current_episode_ids = list(range(num_envs))
        self.next_episode_id = num_envs

    def write(self, env_index: int, state, action, next_state, reward: float, terminal: float, is_real = True):
        ptr = self.write_ptrs[env_index]

        self.states[env_index, ptr] = as_tensor(state, dtype = torch.float32)
        self.actions[env_index, ptr] = as_tensor(action, dtype = torch.float32)
        self.next_states[env_index, ptr] = as_tensor(next_state, dtype = torch.float32)
        self.rewards[env_index, ptr] = reward
        self.terminals[env_index, ptr] = terminal
        self.episode_ids[env_index, ptr] = self.current_episode_ids[env_index]
        self.is_real[env_index, ptr] = is_real

        self.is_full = self.is_full or (ptr + 1 >= self.capacity)
        self.write_ptrs[env_index] = (ptr + 1) % self.capacity

    def add(self, env_index: int, state, action, next_state, reward: float, terminated: bool, truncated: bool):
        self.write(env_index, state, action, next_state, float(reward), float(terminated), is_real = True)

        if terminated:
            for _ in range(self.absorbing_steps):
                random_action = np.random.uniform(self.action_low, self.action_high).astype(np.float32)
                self.write(env_index, next_state, random_action, next_state, 0., 1., is_real = False)

        if terminated or truncated:
            self.current_episode_ids[env_index] = self.next_episode_id
            self.next_episode_id += 1

    def active_length(self, pad = 20):
        max_ptr = max(self.write_ptrs)
        return self.capacity if (self.is_full or max_ptr + pad >= self.capacity) else max(max_ptr, pad)

    def reward_rms(self, device):
        real_rewards = self.rewards[self.is_real]
        if real_rewards.numel() < 64:
            return tensor(1., device = device)

        return real_rewards.square().mean().sqrt().clamp(min = 1e-2).to(device)

    def sample_indices(self, valid_mask: Tensor, num_samples: int, recent_window = 4_000, recent_frac = 0.35):
        length = valid_mask.shape[1]
        age = (tensor(self.write_ptrs)[:, None] - 1 - arange(length)[None, :]) % self.capacity

        all_envs, all_times = valid_mask.nonzero(as_tuple = True)
        recent_envs, recent_times = (valid_mask & (age < recent_window)).nonzero(as_tuple = True)

        if recent_envs.numel() == 0:
            recent_envs, recent_times = all_envs, all_times

        num_recent = int(num_samples * recent_frac)
        recent_pick = randint(0, recent_envs.numel(), (num_recent,))
        all_pick = randint(0, all_envs.numel(), (num_samples - num_recent,))

        return cat((recent_envs[recent_pick], all_envs[all_pick])), cat((recent_times[recent_pick], all_times[all_pick]))

    def valid_window_mask(self, horizon: int):
        length = self.active_length(horizon + 2)
        times = arange(length)
        age = (tensor(self.write_ptrs)[:, None] - 1 - times[None, :]) % self.capacity
        end_times = (times + horizon - 1) % self.capacity

        same_episode = (self.episode_ids[:, :length] >= 0) & (self.episode_ids[:, :length] == self.episode_ids[:, end_times])
        return same_episode & (age >= horizon - 1), age

    def sample_sequences(self, batch_size: int, horizon: int, device, max_age: int | None = None):
        valid_mask, age = self.valid_window_mask(horizon)

        if exists(max_age):
            recent_mask = valid_mask & (age < max_age)
            candidate_envs, candidate_times = (recent_mask if recent_mask.any() else valid_mask).nonzero(as_tuple = True)
            pick = randint(0, candidate_envs.numel(), (batch_size,))
            env_indices, start_times = candidate_envs[pick], candidate_times[pick]
        else:
            env_indices, start_times = self.sample_indices(valid_mask, batch_size)

        time_offsets = (start_times[:, None] + arange(horizon)[None, :]) % self.capacity
        env_grid = repeat(env_indices, 'b -> b t', t = horizon)

        final_next_state = self.next_states[env_indices, (start_times + horizon - 1) % self.capacity]
        states = cat((self.states[env_grid, time_offsets], rearrange(final_next_state, 'b d -> b 1 d')), dim = 1)

        return tree_map_tensor_to_device((
            states,
            self.actions[env_grid, time_offsets],
            self.next_states[env_grid, time_offsets],
            self.terminals[env_grid, time_offsets],
            self.rewards[env_grid, time_offsets]
        ), device)

    def sample_states(self, batch_size: int, device):
        env_indices, time_indices = self.sample_indices(self.is_real[:, :self.active_length()], batch_size)
        return self.states[env_indices, time_indices].to(device)

    def sample_terminals(self, batch_size: int, device, min_age = 600):
        length = self.active_length()
        age = (tensor(self.write_ptrs)[:, None] - 1 - arange(length)[None, :]) % self.capacity
        term_envs, term_times = (self.is_real[:, :length] & (self.terminals[:, :length] > 0.5) & (age >= min_age)).nonzero(as_tuple = True)

        if term_envs.numel() == 0:
            return None

        pick = randint(0, term_envs.numel(), (batch_size,))
        env_indices, time_indices = term_envs[pick], term_times[pick]
        return tree_map_tensor_to_device((self.states[env_indices, time_indices], self.rewards[env_indices, time_indices]), device)

# evaluation and trajectory gif recording

def evaluate(
    env_name: str,
    actor: Module,
    device,
    num_episodes = 20,
    seed_base = 10_000,
    record_dir: Path | None = None
):
    should_record = exists(record_dir)

    if should_record:
        record_dir.mkdir(parents = True, exist_ok = True)

    envs = [gym.make(env_name, render_mode = 'rgb_array' if should_record else None) for _ in range(num_episodes)]
    states = np.asarray([env.reset(seed = seed_base + i)[0] for i, env in enumerate(envs)], dtype = np.float32)

    frames = [[env.render()] for env in envs] if should_record else None
    dones = np.zeros(num_episodes, dtype = bool)
    scores = np.zeros(num_episodes, dtype = np.float64)

    with temp_eval(actor), torch.no_grad():
        while not dones.all():
            active = ~dones
            actions = actor(as_tensor(states[active], device = device)).cpu().numpy()

            for k, env_index in enumerate(np.flatnonzero(active)):
                next_state, reward, terminated, truncated, _ = envs[env_index].step(actions[k])
                scores[env_index] += float(reward)

                if should_record:
                    frames[env_index].append(envs[env_index].render())

                if terminated or truncated:
                    dones[env_index] = True
                else:
                    states[env_index] = next_state

    for env in envs:
        env.close()

    if should_record:
        for ep_index, (ep_frames, ep_return) in enumerate(zip(frames, scores)):
            gif_path = record_dir / f'episode_{ep_index + 1:02d}_return_{ep_return:+.0f}.gif'
            imageio.mimsave(gif_path, ep_frames[::2], fps = 25, loop = 0)

    return float(scores.mean()), float(scores.std())

def save_returns_plot(train_returns: list[float], eval_steps: list[int], eval_means: list[float], out_path: Path):
    fig, ax = plt.subplots(figsize = (8.5, 4.5), dpi = 150)

    if len(train_returns) >= 50:
        smoothed = np.convolve(train_returns, np.ones(50) / 50., mode = 'valid')
        x_train = np.linspace(0, eval_steps[-1] / 1e3 if eval_steps else len(smoothed), len(smoothed))
        ax.plot(x_train, smoothed, color = '#64748b', alpha = 0.7, label = 'Train (50-ep mean)')

    if eval_steps:
        ax.plot([step / 1e3 for step in eval_steps], eval_means, color = '#1d4ed8', marker = 'o', markersize = 4, linewidth = 2.2, label = 'Eval mean (EMA actor)')

    ax.axhline(200., color = '#16a34a', linestyle = '--', linewidth = 1.3, label = 'Solved (200+)')
    ax.set_xlabel('Environment Steps (×1,000)')
    ax.set_ylabel('Return')
    ax.set_title('LunarLanderContinuous-v3 — SuperMimic')
    ax.grid(True, alpha = 0.25)
    ax.legend(loc = 'lower right')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# main training

def main(
    env_name = 'LunarLanderContinuous-v3',
    out_dir = './results_lunar',
    seed = 42,
    num_envs = 16,
    buffer_capacity = 48_000,
    num_iterations = 1_500,
    env_steps_per_iter = 16,
    warmup_steps_per_env = 320,
    wm_pretrain_steps = 200,
    wm_updates_per_iter = 6,
    wm_horizon = 8,
    imagination_horizon = 24,
    critic_updates_per_iter = 4,
    critic_horizon = 16,
    critic_max_age = 600,
    gae_lam = 0.85,
    terminal_batch_size = 64,
    terminal_anchor_weight = 0.02,
    batch_size = 256,
    wm_lr = 3e-4,
    actor_lr = 1.8e-4,
    critic_lr = 3e-4,
    weight_decay = 1e-5,
    ema_decay = 0.98,
    use_ema_critic = True,
    critic_ema_decay = 0.99,
    observation_noise_scale = 0.005,
    max_grad_norm = 1.,
    state_loss_weight = 150.,
    discount_factor = 0.993,
    entropy_weight = 2e-3,
    rolling_window = 50,
    target_return: float | None = 250.,
    eval_every = 25,
    num_record_episodes = 8
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    out_path = Path(out_dir)
    out_path.mkdir(parents = True, exist_ok = True)

    # environments

    envs = [gym.make(env_name) for _ in range(num_envs)]
    env_states = [env.reset(seed = seed + i)[0] for i, env in enumerate(envs)]

    dim_state = int(envs[0].observation_space.shape[0])
    dim_action = int(envs[0].action_space.shape[0])
    action_low, action_high = envs[0].action_space.low, envs[0].action_space.high

    # models

    actor = Actor(dim_state, dim_action, bounds = np.stack((action_low, action_high), axis = -1)).to(device)
    ema_actor = EMA(actor, beta = ema_decay, include_online_model = False, update_after_step = 0, update_every = 1).to(device)

    world_model = WorldModel(
        dim_state = dim_state,
        dim_action = dim_action,
        dim_sim_context = 0,
        dim_hiddens = (256, 256, 256),
        activation = SiLU(),
        normed_mlp = True,
        pred_residual_state = True,
        pred_terminal = True,
        pred_reward = True,
        state_loss_weight = state_loss_weight
    ).to(device)

    critic = Critic(
        dim_state = dim_state,
        dim_goal = 0,
        dim_sim_context = 0,
        dim_hiddens = (256, 256),
        activation = SiLU(),
        normed_mlp = True
    ).to(device)

    supermimic = SuperMimic(
        actor = actor,
        world_model = world_model,
        critic = critic,
        use_ema_critic = use_ema_critic,
        critic_ema_decay = critic_ema_decay,
        observation_noise_scale = observation_noise_scale,
        discount_factor = discount_factor
    ).to(device)

    # optimizers

    world_model_opt = AdamW(world_model.parameters(), lr = wm_lr, weight_decay = weight_decay)
    actor_opt = AdamW(actor.parameters(), lr = actor_lr, weight_decay = weight_decay)
    actor_scheduler = CosineAnnealingLR(actor_opt, T_max = num_iterations, eta_min = actor_lr * 0.2)
    critic_opt = AdamW(critic.parameters(), lr = critic_lr, weight_decay = weight_decay)

    # replay state

    buffer = ReplayBuffer(num_envs, buffer_capacity, dim_state, dim_action, action_low, action_high, absorbing_steps = wm_horizon - 1)
    running_returns = np.zeros(num_envs)
    train_returns, recent_returns = [], deque(maxlen = rolling_window)
    eval_steps, eval_means = [], []
    total_env_steps = 0

    # environment step helper

    def collect_env_steps(num_steps: int, random_actions = False):
        nonlocal total_env_steps
        actor.train()

        for _ in range(num_steps):
            states_tensor = as_tensor(np.asarray(env_states, dtype = np.float32), device = device)

            with torch.no_grad():
                actions = np.array([env.action_space.sample() for env in envs]) if random_actions else actor(states_tensor).cpu().numpy()

            for env_index, env in enumerate(envs):
                next_state, reward, terminated, truncated, _ = env.step(actions[env_index])
                buffer.add(env_index, env_states[env_index], actions[env_index], next_state, reward, terminated, truncated)

                total_env_steps += 1
                running_returns[env_index] += reward

                if terminated or truncated:
                    episode_return = float(running_returns[env_index])
                    train_returns.append(episode_return)
                    recent_returns.append(episode_return)
                    running_returns[env_index] = 0.
                    env_states[env_index], _ = env.reset()
                else:
                    env_states[env_index] = next_state

    # world model update helper

    def update_world_model(num_updates: int):
        reward_scale = buffer.reward_rms(device)

        for _ in range(num_updates):
            states, actions, next_states, next_terminals, next_rewards = buffer.sample_sequences(batch_size, wm_horizon, device)

            wm_loss, _, _ = world_model.rollout(
                states[:, 0],
                actions,
                next_states,
                next_terminals = next_terminals,
                next_rewards = next_rewards / reward_scale
            )

            world_model_opt.zero_grad()
            (wm_loss / wm_horizon).backward()
            clip_grad_norm_(world_model.parameters(), max_grad_norm)
            world_model_opt.step()

        return reward_scale

    # warmup and initial world model fit

    collect_env_steps(warmup_steps_per_env, random_actions = True)
    update_world_model(wm_pretrain_steps)

    # training iterations

    for iteration in range(1, num_iterations + 1):
        collect_env_steps(env_steps_per_iter)

        reward_scale = update_world_model(wm_updates_per_iter)

        # update critic on real replay sequences

        for _ in range(critic_updates_per_iter):
            critic_states, _, _, critic_terminals, critic_rewards = buffer.sample_sequences(batch_size, critic_horizon, device, max_age = critic_max_age)
            critic_loss = supermimic.critic_loss(critic_states, critic_rewards / reward_scale, critic_terminals, lam = gae_lam)

            if terminal_anchor_weight > 0.:
                terminal_batch = buffer.sample_terminals(terminal_batch_size, device, min_age = critic_max_age)
                if exists(terminal_batch):
                    term_states, term_rewards = terminal_batch
                    critic_loss = critic_loss + terminal_anchor_weight * critic(term_states, returns = term_rewards / reward_scale)

            critic_opt.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_opt.step()
            supermimic.update_ema()

        # imagine rollout through frozen world model and update actor via bptt

        initial_states = buffer.sample_states(batch_size, device)
        rollout = supermimic(initial_states, num_rollout_timesteps = imagination_horizon)

        entropy = actor.entropy(rollout.states[:, :-1].detach()).mean()

        actor_loss = supermimic.actor_loss(
            rollout.rewards,
            rollout.terminals,
            final_state = rollout.states[:, -1]
        ) - entropy_weight * entropy

        actor_opt.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(actor.parameters(), max_grad_norm)
        actor_opt.step()

        ema_actor.update()
        actor_scheduler.step()

        # periodic evaluation and early stopping

        if divisible_by(iteration, eval_every) or iteration == num_iterations:
            eval_mean, eval_std = evaluate(env_name, ema_actor.ema_model, device)
            eval_steps.append(total_env_steps)
            eval_means.append(eval_mean)

            rolling_mean = sum(recent_returns) / len(recent_returns) if recent_returns else 0.
            print(f'iter {iteration:4d} | steps {total_env_steps:6d} | train{rolling_window} {rolling_mean:6.1f} | eval {eval_mean:6.1f} ± {eval_std:5.1f}', flush = True)

            torch.save(ema_actor.ema_model.state_dict(), out_path / 'actor.pt')
            save_returns_plot(train_returns, eval_steps, eval_means, out_path / 'returns.png')

            if exists(target_return) and eval_mean >= target_return:
                print(f'reached eval return {eval_mean:.1f} >= {target_return:.1f} at iter {iteration}, stopping early', flush = True)
                break

    for env in envs:
        env.close()

    # record final trajectories

    if num_record_episodes > 0:
        trajectories_dir = out_path / 'trajectories'
        final_mean, final_std = evaluate(env_name, ema_actor.ema_model, device, num_episodes = num_record_episodes, seed_base = 20_000, record_dir = trajectories_dir)
        print(f'saved {num_record_episodes} trajectories to {trajectories_dir} (return {final_mean:.1f} ± {final_std:.1f})', flush = True)

if __name__ == '__main__':
    fire.Fire(main)
