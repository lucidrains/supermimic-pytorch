# /// script
# requires-python = ">= 3.10"
# dependencies = [
#     "supermimic-pytorch",
#     "memmap-replay-buffer>=0.3.0",
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
from torch import as_tensor, cat, tensor
from torch.nn import Module, SiLU, init
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from einops import rearrange
from torch_einops_utils import temp_eval
from ema_pytorch import EMA
from mean_conc_beta import Beta
from memmap_replay_buffer import StreamReplayBuffer
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
    memmap = True,
    recent_window = 4_000,
    recent_frac = 0.35,
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

    # replay buffer

    buffer = StreamReplayBuffer(
        folder = out_path / 'replay' if memmap else None,
        num_streams = num_envs,
        capacity = buffer_capacity,
        fields = dict(
            state = ('float', dim_state),
            action = ('float', dim_action),
            next_state = ('float', dim_state),
            reward = 'float',
            terminal = 'float',
            is_real = ('bool', (), False)
        )
    )

    absorbing_steps = wm_horizon - 1
    real_only = dict(is_real = True)
    real_terminals = dict(is_real = True, terminal = lambda x: x > 0.5)
    recent_mixture = [(recent_frac, dict(max_age = recent_window)), (1. - recent_frac, dict())]

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

                buffer.store_step(
                    env_index,
                    state = env_states[env_index],
                    action = actions[env_index],
                    next_state = next_state,
                    reward = float(reward),
                    terminal = float(terminated),
                    is_real = True
                )

                if terminated:
                    for _ in range(absorbing_steps):
                        buffer.store_step(
                            env_index,
                            state = next_state,
                            action = np.random.uniform(action_low, action_high).astype(np.float32),
                            next_state = next_state,
                            reward = 0.,
                            terminal = 1.,
                            is_real = False
                        )

                total_env_steps += 1
                running_returns[env_index] += reward

                if terminated or truncated:
                    buffer.end_episode(env_index)
                    episode_return = float(running_returns[env_index])
                    train_returns.append(episode_return)
                    recent_returns.append(episode_return)
                    running_returns[env_index] = 0.
                    env_states[env_index], _ = env.reset()
                else:
                    env_states[env_index] = next_state

    # world model update helper

    def update_world_model(num_updates: int):
        r_stats = buffer.field_stats('reward', filter_fields = real_only)
        reward_scale = tensor(max(r_stats.rms, 1e-2) if r_stats.count >= 64 else 1., dtype = torch.float32, device = device)

        for _ in range(num_updates):
            batch = buffer.sample(
                batch_size,
                horizon = wm_horizon,
                mixture = recent_mixture,
                to_named_tuple = ('state', 'action', 'next_state', 'terminal', 'reward'),
                device = device
            )

            wm_loss, _, _ = world_model.rollout(
                batch.state[:, 0],
                batch.action,
                batch.next_state,
                next_terminals = batch.terminal,
                next_rewards = batch.reward / reward_scale
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
            batch = buffer.sample(
                batch_size,
                horizon = critic_horizon,
                max_age = critic_max_age,
                fallback_on_empty = True,
                to_named_tuple = ('state', 'last_next_state', 'terminal', 'reward'),
                device = device
            )

            critic_states = cat((batch.state, rearrange(batch.last_next_state, 'b d -> b 1 d')), dim = 1)
            critic_loss = supermimic.critic_loss(critic_states, batch.reward / reward_scale, batch.terminal, lam = gae_lam)

            if terminal_anchor_weight > 0.:
                term = buffer.sample(
                    terminal_batch_size,
                    min_age = critic_max_age,
                    filter_fields = real_terminals,
                    allow_empty = True,
                    to_named_tuple = ('state', 'reward'),
                    device = device
                )

                if exists(term):
                    critic_loss = critic_loss + terminal_anchor_weight * critic(term.state, returns = term.reward / reward_scale)

            critic_opt.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_opt.step()
            supermimic.update_ema()

        # imagine rollout through frozen world model and update actor via bptt

        initial_states = buffer.sample(
            batch_size,
            filter_fields = real_only,
            mixture = recent_mixture,
            fields = ('state',),
            device = device
        )['state']

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
