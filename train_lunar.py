# /// script
# requires-python = ">= 3.10"
# dependencies = [
#     "supermimic-pytorch",
#     "memmap-replay-buffer>=0.3.0",
#     "mean-conc-beta>=0.2.11",
#     "discrete-continuous-embed-readout>=0.2.12",
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

import math
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
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from einops import pack, rearrange
from torch_einops_utils import pad_right_at_dim, safe_cat, temp_eval
from torch_einops_utils.shape import size
from ema_pytorch import EMA
from mean_conc_beta import Beta
from discrete_continuous_embed_readout import MultiCategorical
from memmap_replay_buffer import StreamReplayBuffer
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP

from supermimic_pytorch import WorldModel, Critic, SuperMimic, HistoryWindow
from supermimic_pytorch.supermimic import repeat_batch

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def cosine_anneal(start: float, end: float, frac: float) -> float:
    return end + (start - end) * 0.5 * (1. + math.cos(math.pi * frac))

def pad_to_dim(x, dim: int):
    if isinstance(x, (list, tuple)):
        x = cat(x, dim = -1)
    pad = dim - size(x, '... [d]')
    return pad_right_at_dim(x, pad, dim = -1) if pad > 0 else x

# actor

def init_history(actor, dim_state, dim_action, obs_window, action_window, pad_mode, per_slot_pad):
    actor.obs_history = HistoryWindow(obs_window, dim_state, pad_mode = pad_mode, per_slot_pad = per_slot_pad) if obs_window > 1 else None
    actor.action_history = HistoryWindow(action_window, dim_action, lag = 1, pad_mode = pad_mode, per_slot_pad = per_slot_pad) if action_window > 0 else None
    dim_obs = actor.obs_history.dim_out if exists(actor.obs_history) else dim_state
    dim_act = actor.action_history.dim_out if exists(actor.action_history) else 0
    return dim_obs + dim_act

def encode_history(actor, states, actions = None, past_obs = None, past_actions = None, past_mask = None):
    obs = states
    if exists(actor.obs_history):
        obs = rearrange(actor.obs_history(states, past = past_obs, past_mask = past_mask), 'b t w d -> b t (w d)')

    act = None
    if exists(actor.action_history):
        act = rearrange(actor.action_history(actions, past = past_actions, past_mask = past_mask), 'b t w d -> b t (w d)')

    return safe_cat((obs, act), dim = -1)

class RolloutHistory:
    def __init__(self, actor, batch_size, device):
        self.obs_history, self.action_history = actor.obs_history, actor.action_history
        self.obs_cache = None
        self.action_cache = None

        if exists(self.action_history):
            self.action_cache = self.action_history.init_cache(batch_size, device = device)

    def observe(self, state):
        if exists(self.obs_history):
            state, self.obs_cache = self.obs_history.step(state, self.obs_cache)
            state = rearrange(state, 'b w d -> b (w d)')

        past_action = None
        if exists(self.action_history):
            past_action = rearrange(self.action_history.read(self.action_cache), 'b w d -> b (w d)')

        return safe_cat((state, past_action), dim = -1)

    def record_action(self, action):
        if exists(self.action_history):
            self.action_cache = self.action_history.push(action, self.action_cache)

    def reset(self, done):
        if exists(self.obs_history):
            self.obs_cache = self.obs_history.reset(self.obs_cache, done)
        if exists(self.action_history):
            self.action_cache = self.action_history.reset(self.action_cache, done)

class Actor(Module):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        bounds = (-1., 1.),
        dim_hidden = 256,
        depth = 3,
        obs_window = 1,
        action_window = 0,
        pad_mode = 'learned',
        per_slot_pad = True,
        dim_explorative_latent = 0
    ):
        super().__init__()
        self.dim_in = init_history(self, dim_state, dim_action, obs_window, action_window, pad_mode, per_slot_pad) + dim_explorative_latent
        self.net = NormedMLP(self.dim_in, *([dim_hidden] * depth), dim_action * 2, activation = SiLU())

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
        return rearrange(self.net(pad_to_dim(state, self.dim_in)), '... (d p) -> ... d p', p = 2)

    def entropy(self, state, actions = None, **kw):
        return self.beta.entropy(self.params(encode_history(self, state, actions, **kw)))

    def act(self, state):
        return self(state)

    def act_to_history(self, actions):
        return actions

    def forward(self, state):
        params = self.params(state)
        return self.beta.rsample(params) if self.training else self.beta.mean(params)

class DiscreteActor(Module):
    def __init__(
        self,
        dim_state: int,
        num_actions: int,
        dim_hidden = 256,
        depth = 3,
        action_noise = 0.05,
        obs_window = 1,
        action_window = 0,
        pad_mode = 'learned',
        per_slot_pad = True,
        dim_explorative_latent = 0
    ):
        super().__init__()
        self.num_actions = num_actions
        self.action_noise = action_noise
        self.dim_in = init_history(self, dim_state, num_actions, obs_window, action_window, pad_mode, per_slot_pad) + dim_explorative_latent
        self.net = NormedMLP(self.dim_in, *([dim_hidden] * depth), num_actions, activation = SiLU())

        last_linear = self.net.layers[-1][0]
        init.orthogonal_(last_linear.weight, gain = 0.01)
        init.zeros_(last_linear.bias)

    def dist(self, state):
        return MultiCategorical(self.forward(state))

    def entropy(self, state, actions = None, past_actions = None, **kw):
        if exists(past_actions) and not past_actions.is_floating_point():
            past_actions = F.one_hot(past_actions.long(), self.num_actions).to(state.dtype)
        return self.dist(encode_history(self, state, actions, past_actions = past_actions, **kw)).entropy()

    def act(self, state):
        actions = self.dist(state).sample(temperature = 0.)

        if self.training and self.action_noise > 0.:
            noise_mask = torch.rand(actions.shape, device = actions.device) < self.action_noise
            random_actions = torch.randint_like(actions, self.num_actions)
            actions = torch.where(noise_mask, random_actions, actions)

        return actions

    def act_to_history(self, actions):
        return F.one_hot(actions.long(), self.num_actions).float()

    def forward(self, state):
        return self.net(pad_to_dim(state, self.dim_in))

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

    hist = RolloutHistory(actor, num_episodes, device)

    with temp_eval(actor), torch.no_grad():
        while not dones.all():
            active = ~dones
            x = hist.observe(as_tensor(states, device = device))
            actions_t = actor.act(x)
            hist.record_action(actor.act_to_history(actions_t))
            actions = actions_t.cpu().numpy()

            for env_index in np.flatnonzero(active):
                next_state, reward, terminated, truncated, _ = envs[env_index].step(actions[env_index])
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

def save_returns_plot(
    train_returns: list[float],
    eval_steps: list[int],
    eval_means: list[float],
    out_path: Path,
    title: str = 'LunarLanderContinuous-v3 — SuperMimic'
):
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
    ax.set_title(title)
    ax.grid(True, alpha = 0.25)
    ax.legend(loc = 'lower right')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# main training

def main(
    continuous = True,
    env_name: str | None = None,
    out_dir: str | None = None,
    seed = 42,
    obs_window = 1,
    action_window = 0,
    history_pad_mode = 'learned',
    per_slot_pad_embed = True,
    action_smoothness_weight = 0.,
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
    num_explorative_rollouts = 2,
    dim_explorative_latent = 8,
    explorative_latent_scale = 0.17,
    explorative_latent_scale_end = 0.,
    discrete_temperature = 1.,
    discrete_action_noise = 0.05,
    rolling_window = 50,
    target_return: float | None = 250.,
    eval_every = 25,
    num_record_episodes = 8
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # environments

    env_name = default(env_name, 'LunarLanderContinuous-v3' if continuous else 'LunarLander-v3')
    envs = [gym.make(env_name) for _ in range(num_envs)]
    env_states = [env.reset(seed = seed + i)[0] for i, env in enumerate(envs)]
    env_is_first = [True] * num_envs

    is_discrete = isinstance(envs[0].action_space, gym.spaces.Discrete)
    dim_state = int(envs[0].observation_space.shape[0])

    out_dir = default(out_dir, './results_lunar_discrete' if is_discrete else './results_lunar')
    out_path = Path(out_dir)
    out_path.mkdir(parents = True, exist_ok = True)

    # models

    history_kwargs = dict(
        obs_window = obs_window,
        action_window = action_window,
        pad_mode = history_pad_mode,
        per_slot_pad = per_slot_pad_embed
    )

    active_latent_dim = dim_explorative_latent if num_explorative_rollouts > 1 else 0

    if is_discrete:
        num_actions, dim_action = int(envs[0].action_space.n), None
        actor = DiscreteActor(dim_state, num_actions, action_noise = discrete_action_noise, dim_explorative_latent = active_latent_dim, **history_kwargs).to(device)
    else:
        num_actions, dim_action = None, int(envs[0].action_space.shape[0])
        action_low, action_high = envs[0].action_space.low, envs[0].action_space.high
        actor = Actor(dim_state, dim_action, bounds = np.stack((action_low, action_high), axis = -1), dim_explorative_latent = active_latent_dim, **history_kwargs).to(device)

    ema_actor = EMA(actor, beta = ema_decay, include_online_model = False, update_after_step = 0, update_every = 1).to(device)

    world_model = WorldModel(
        dim_state = dim_state,
        dim_action = dim_action,
        num_discrete_actions = num_actions,
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
        obs_history = actor.obs_history,
        action_history = actor.action_history,
        action_smoothness_weight = action_smoothness_weight,
        discount_factor = discount_factor,
        discrete_action_temperature = discrete_temperature,
        num_explorative_rollouts = num_explorative_rollouts,
        dim_explorative_latent = active_latent_dim,
        explorative_latent_scale = explorative_latent_scale
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
            action = 'int' if is_discrete else ('float', dim_action),
            next_state = ('float', dim_state),
            reward = 'float',
            terminal = 'float',
            is_real = ('bool', (), False),
            is_first = ('bool', (), False)
        )
    )

    prefix_len = max(obs_window - 1, action_window)
    absorbing_steps = wm_horizon - 1
    real_only = dict(is_real = True)
    real_starts = dict(is_real = True, is_first = True)
    real_terminals = dict(is_real = True, terminal = lambda x: x > 0.5)
    recent_mixture = [(recent_frac, dict(max_age = recent_window)), (1. - recent_frac, dict())]

    running_returns = np.zeros(num_envs)
    train_returns, recent_returns = [], deque(maxlen = rolling_window)
    eval_steps, eval_means = [], []
    best_eval = -float('inf')
    total_env_steps = 0
    env_hist = RolloutHistory(actor, num_envs, device)

    # environment step helper

    def collect_env_steps(num_steps: int, random_actions = False):
        nonlocal total_env_steps
        actor.train()

        for _ in range(num_steps):
            states_tensor = as_tensor(np.asarray(env_states, dtype = np.float32), device = device)

            with torch.no_grad():
                x = env_hist.observe(states_tensor)
                if random_actions:
                    actions = np.array([env.action_space.sample() for env in envs])
                    actions_t = as_tensor(actions, device = device, dtype = torch.long if is_discrete else torch.float32)
                else:
                    actions_t = actor.act(x)
                    actions = actions_t.cpu().numpy()

                env_hist.record_action(actor.act_to_history(actions_t))

            done_flags = np.zeros(num_envs, dtype = bool)

            for env_index, env in enumerate(envs):
                next_state, reward, terminated, truncated, _ = env.step(actions[env_index])

                buffer.store_step(
                    env_index,
                    state = env_states[env_index],
                    action = actions[env_index],
                    next_state = next_state,
                    reward = float(reward),
                    terminal = float(terminated),
                    is_real = True,
                    is_first = env_is_first[env_index]
                )
                env_is_first[env_index] = False

                if terminated:
                    for _ in range(absorbing_steps):
                        buffer.store_step(
                            env_index,
                            state = next_state,
                            action = env.action_space.sample(),
                            next_state = next_state,
                            reward = 0.,
                            terminal = 1.,
                            is_real = False,
                            is_first = False
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
                    env_is_first[env_index] = True
                    done_flags[env_index] = True
                else:
                    env_states[env_index] = next_state

            if done_flags.any():
                env_hist.reset(as_tensor(done_flags, device = device))

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

            critic_states, _ = pack((batch.state, batch.last_next_state), 'b * d')
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

        past_obs = past_actions = past_mask = None

        if prefix_len > 0:
            starts = buffer.sample(max(1, batch_size // 32), filter_fields = real_starts, allow_empty = True, fields = ('state',), device = device)
            n_starts = size(starts['state'], '[b] ...') if exists(starts) else 0

            seed = buffer.sample(
                batch_size - n_starts,
                horizon = prefix_len + 1,
                filter_fields = real_only,
                window_filter_mode = 'all',
                mixture = recent_mixture,
                to_named_tuple = ('state', 'action'),
                device = device
            )

            initial_states, past_obs, past_actions = seed.state[:, -1], seed.state[:, :-1], seed.action[:, :-1]
            past_mask = torch.ones((size(initial_states, '[b] ...'), prefix_len), dtype = torch.bool, device = device)

            if n_starts > 0:
                initial_states = cat((initial_states, starts['state']), dim = 0)
                past_obs = pad_right_at_dim(past_obs, n_starts, dim = 0)
                past_actions = pad_right_at_dim(past_actions, n_starts, dim = 0)
                past_mask = pad_right_at_dim(past_mask, n_starts, dim = 0, value = False)
        else:
            initial_states = buffer.sample(
                batch_size,
                filter_fields = real_only,
                mixture = recent_mixture,
                fields = ('state',),
                device = device
            )['state']

        cur_latent_scale = cosine_anneal(
            explorative_latent_scale,
            explorative_latent_scale_end,
            (iteration - 1) / max(1, num_iterations - 1)
        )

        rollout = supermimic(
            initial_states,
            past_obs = past_obs,
            past_actions = past_actions,
            past_mask = past_mask,
            num_rollout_timesteps = imagination_horizon,
            explorative_latent_scale = cur_latent_scale
        )

        past_obs_k, past_actions_k, past_mask_k = repeat_batch((past_obs, past_actions, past_mask), num_explorative_rollouts)

        entropy = actor.entropy(
            rollout.states[:, :-1].detach(),
            rollout.actions.detach(),
            past_obs = past_obs_k,
            past_actions = past_actions_k,
            past_mask = past_mask_k
        ).mean()

        actor_loss = supermimic.actor_loss(
            rollout.rewards,
            rollout.terminals,
            final_state = rollout.states[:, -1],
            actions = rollout.actions,
            past_actions = past_actions,
            past_mask = past_mask
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

            if eval_mean >= best_eval:
                best_eval = eval_mean
                torch.save(ema_actor.ema_model.state_dict(), out_path / 'actor.pt')

            tags = ([f'obs={obs_window}', f'act={action_window}', history_pad_mode] if prefix_len > 0 else []) + ([f'smooth={action_smoothness_weight}'] if action_smoothness_weight > 0 else [])
            history_suffix = f' ({", ".join(tags)})' if tags else ''
            save_returns_plot(train_returns, eval_steps, eval_means, out_path / 'returns.png', title = f'{env_name} — SuperMimic{history_suffix}')

            if exists(target_return) and eval_mean >= target_return:
                print(f'reached eval return {eval_mean:.1f} >= {target_return:.1f} at iter {iteration}, stopping early', flush = True)
                break

    for env in envs:
        env.close()

    # record final trajectories from best checkpoint

    if num_record_episodes > 0:
        ckpt_path = out_path / 'actor.pt'
        if ckpt_path.exists():
            ema_actor.ema_model.load_state_dict(torch.load(ckpt_path, map_location = device, weights_only = True))

        trajectories_dir = out_path / 'trajectories'
        final_mean, final_std = evaluate(env_name, ema_actor.ema_model, device, num_episodes = num_record_episodes, seed_base = 20_000, record_dir = trajectories_dir)
        print(f'saved {num_record_episodes} trajectories to {trajectories_dir} (return {final_mean:.1f} ± {final_std:.1f})', flush = True)

if __name__ == '__main__':
    fire.Fire(main)
