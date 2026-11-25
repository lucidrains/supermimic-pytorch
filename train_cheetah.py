# /// script
# requires-python = ">= 3.10"
# dependencies = [
#     "supermimic-pytorch",
#     "x-ppo",
#     "memmap-replay-buffer>=0.3.0",
#     "mean-conc-beta>=0.2.11",
#     "gymnasium[mujoco]",
#     "ema-pytorch>=0.8.3",
#     "einops>=0.8.1",
#     "einx>=0.3.0",
#     "x-mlps-pytorch>=0.6.14",
#     "fire",
#     "imageio",
#     "imageio-ffmpeg",
#     "matplotlib",
#     "numpy",
#     "pillow",
# ]
# ///

from __future__ import annotations

import copy
import math
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import time
from pathlib import Path

import fire
import imageio
import einx
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.utils import RunningMeanStd
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import Tensor, as_tensor
from torch.nn import Module, SiLU, Tanh, init
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from einops import rearrange
from torch_einops_utils import pad_right_at_dim, safe_cat, temp_eval
from torch_einops_utils.shape import shape, size
from ema_pytorch import EMA
from mean_conc_beta import Beta
from memmap_replay_buffer import StreamReplayBuffer
from x_mlps_pytorch import MLP
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP
from x_ppo import calc_gae, clipped_value_loss_fn, ppo_actor_loss

from supermimic_pytorch import (
    ActionSmoothnessTerm,
    AdversarialCritic,
    AdversarialSuperMimic,
    AnalyticalReward,
    Critic,
    HistoryWindow,
    SuperMimic,
    TrackingTerm,
    WorldModel
)

# constants

REF_DIM = 8          # [root_z, torso_pitch, 6 joint angles] — all 9 velocities hidden
GOAL_WINDOW = 8
MAX_TRACK_REWARD = 4.

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def cosine_anneal(start: float, end: float, frac: float) -> float:
    return end + (start - end) * 0.5 * (1. + math.cos(math.pi * frac))

def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def step_optimizer(opt, loss: Tensor, params, max_norm = 1., clip_value = None):
    params = list(params)
    opt.zero_grad()
    loss.backward()
    clip_grad_norm_(params, max_norm)

    if exists(clip_value):
        clip_grad_value_(params, clip_value)

    opt.step()

def make_reward(mean: Tensor | None = None, std: Tensor | None = None, smoothness = -0.01) -> AnalyticalReward:
    state_transform = None

    if exists(mean) and exists(std):
        def state_transform(state):
            return state * std.to(state) + mean.to(state)

    terms = dict(
        root_height = TrackingTerm(0, std = 0.25, weight = 1.),
        torso_pitch = TrackingTerm(1, metric = 'angle', std = 0.35, weight = 1.),
        joint_pos = TrackingTerm(slice(2, REF_DIM), std = 0.75, weight = 2.),
        action_smoothness = ActionSmoothnessTerm(weight = smoothness)
    )

    return AnalyticalReward(terms, state_transform = state_transform)

def gather_refs(refs: Tensor, clip_ids: Tensor, starts: Tensor, length: int) -> Tensor:
    offsets = torch.arange(length, device = refs.device)
    frames = einx.add('b, l -> b l', starts, offsets).clamp(max = size(refs, 'c [t] d') - 1)
    return einx.get_at('[c] [t] d, b, b l -> b l d', refs, clip_ids, frames)

# parallel mocap envs with reference state initialization (RSI)

class MocapEnvs:
    def __init__(
        self,
        dataset: dict,
        split: str,
        device,
        num_envs: int | None = None,
        horizon: int | None = None,
        render = False
    ):
        self.is_eval = split == 'eval'
        self.device = device

        self.qpos = dataset[f'{split}_qpos']
        self.qvel = dataset[f'{split}_qvel']
        self.refs = as_tensor(dataset[f'{split}_refs'], device = device)

        self.num_clips, self.total_frames = shape(self.refs, '[c t] d')
        self.num_envs = default(num_envs, self.num_clips)
        self.horizon = default(horizon, self.total_frames - 1)
        self.max_start = max(1, self.total_frames - self.horizon - 2)

        # state normalization stats from training clips

        flat_train = rearrange(dataset['train_states'], 'c t d -> (c t) d')
        self.mean = flat_train.mean(axis = 0).astype(np.float32)
        self.std = np.maximum(flat_train.std(axis = 0), 1e-2).astype(np.float32)

        # precompute future goal windows for each clip frame

        goal_window = HistoryWindow(GOAL_WINDOW, REF_DIM, future = True, pad_mode = 'replicate')
        self.goal_windows = goal_window(self.refs[:, 1:])

        # create gymnasium envs

        env_name = dataset['env_name']
        self.envs = [
            gym.make(env_name, render_mode = 'rgb_array', camera_name = 'track') if (render and i == 0) else gym.make(env_name)
            for i in range(self.num_envs)
        ]

        self.dim_state = int(self.envs[0].observation_space.shape[0])
        self.dim_action = int(self.envs[0].action_space.shape[0])
        self.bounds = np.stack((self.envs[0].action_space.low, self.envs[0].action_space.high), axis = -1)

        self.clips = np.zeros(self.num_envs, dtype = np.int64)
        self.frames = np.zeros(self.num_envs, dtype = np.int64)
        self.episode_steps = np.zeros(self.num_envs, dtype = np.int64)
        self.states = np.zeros((self.num_envs, self.dim_state), dtype = np.float32)

        self.reset_all()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for env in self.envs:
            env.close()

    def reset_all(self):
        for i in range(self.num_envs):
            clip = i if self.is_eval else None
            frame = 0 if self.is_eval else None
            self.reset(i, clip = clip, frame = frame)

    def reset(self, index: int, clip: int | None = None, frame: int | None = None):
        clip = default(clip, int(np.random.randint(self.num_clips)))
        frame = default(frame, int(np.random.randint(self.max_start)))

        env = self.envs[index]
        env.reset()
        env.unwrapped.set_state(self.qpos[clip, frame], self.qvel[clip, frame])

        self.clips[index] = clip
        self.frames[index] = frame
        self.episode_steps[index] = 0
        self.states[index] = env.unwrapped._get_obs()

    def normalize(self, states: np.ndarray | None = None) -> np.ndarray:
        return (default(states, self.states) - self.mean) / self.std

    def goals(self) -> Tensor:
        clips = as_tensor(self.clips, device = self.device)
        frames = as_tensor(self.frames, device = self.device)
        return einx.get_at('[c] [t] w d, e, e -> e w d', self.goal_windows, clips, frames)

    def step(self, actions: np.ndarray, auto_reset = True):
        next_states = np.empty_like(self.states)
        rewards = np.empty(self.num_envs)
        dones = np.zeros(self.num_envs, dtype = bool)

        for i, env in enumerate(self.envs):
            next_state, reward, terminated, truncated, _ = env.step(actions[i])
            next_states[i] = next_state
            rewards[i] = reward

            self.frames[i] += 1
            self.episode_steps[i] += 1

            reached_end = (
                terminated
                or truncated
                or self.episode_steps[i] >= self.horizon
                or self.frames[i] >= self.total_frames - 2
            )

            if auto_reset and reached_end:
                dones[i] = True
                self.reset(i)
            else:
                self.states[i] = next_state

        return next_states, rewards, dones

@torch.no_grad()
def evaluate_tracker(
    eval_envs: MocapEnvs,
    policy,
    reward_fn: AnalyticalReward,
    device,
    render = False,
    stride = 4
) -> dict:
    eval_envs.reset_all()
    frames = [eval_envs.envs[0].render()] if render else None

    next_states = []
    actions = []
    env_returns = np.zeros(eval_envs.num_envs)

    for t in range(eval_envs.total_frames - 1):
        obs = as_tensor(eval_envs.normalize(), device = device)
        action = policy(obs, eval_envs.goal_windows[:, t]).cpu().numpy()

        next_state, reward, _ = eval_envs.step(action, auto_reset = False)

        actions.append(action)
        next_states.append(next_state)
        env_returns += reward

        if exists(frames) and divisible_by(t + 1, stride):
            frames.append(eval_envs.envs[0].render())

    next_states_t = as_tensor(np.stack(next_states, axis = 1), device = device)
    actions_t = as_tensor(np.stack(actions, axis = 1), device = device)
    refs = eval_envs.refs[:, 1:]

    _, breakdown = reward_fn(next_states_t, refs, actions = actions_t, return_breakdown = True)
    track_reward = sum(value for name, value in breakdown.items() if name != 'action_smoothness')

    return dict(
        norm_return = float(track_reward.mean() / MAX_TRACK_REWARD),
        pose_rmse = float((next_states_t[..., :REF_DIM] - refs).square().mean().sqrt()),
        env_return = float(env_returns.mean()),
        env_return_std = float(env_returns.std()),
        frames = frames
    )

# shared x-ppo agent & update

class PPOAgent(Module):
    def __init__(self, dim_in: int, dim_action: int, bounds: np.ndarray, dim_hidden = 256):
        super().__init__()
        self.actor = MLP(dim_in, dim_hidden, dim_hidden, dim_action * 2, activation = Tanh())
        self.critic = MLP(dim_in, dim_hidden, dim_hidden, 1, activation = Tanh())

        for net, gain in ((self.actor, 0.01), (self.critic, 1.)):
            init.orthogonal_(net.layers[-1].weight, gain = gain)
            init.zeros_(net.layers[-1].bias)

        self.beta = Beta(
            bounds = bounds,
            pos_fn = 'softplus',
            init_conc = 2.,
            detach_entropy_mean = True
        )

    def dist(self, x: Tensor):
        params = rearrange(self.actor(x), '... (d p) -> ... d p', p = 2)
        return self.beta(params)

    def value(self, x: Tensor) -> Tensor:
        return rearrange(self.critic(x), '... 1 -> ...')

def ppo_update(agent: PPOAgent, opt: AdamW, get_obs, step_env, rollout_steps: int, device):
    # collect rollout

    trajectory = []

    for _ in range(rollout_steps):
        obs = get_obs()

        with torch.no_grad():
            dist = agent.dist(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim = -1)
            value = agent.value(obs)

        reward, done = step_env(action.cpu().numpy())
        reward_t = as_tensor(reward, dtype = torch.float32, device = device)
        done_t = as_tensor(done, dtype = torch.float32, device = device)

        trajectory.append((obs, action, log_prob, reward_t, value, done_t))

    # compute gae returns and advantages

    obs_seq, action_seq, log_prob_seq, reward_seq, value_seq, done_seq = map(list, zip(*trajectory))

    rewards = rearrange(reward_seq, 't e -> e t')
    values = rearrange(value_seq, 't e -> e t')
    dones = rearrange(done_seq, 't e -> e t')

    with torch.no_grad():
        gae = calc_gae(
            rewards,
            values,
            masks = 1. - dones,
            next_value = agent.value(get_obs()),
            gamma = 0.99,
            lam = 0.95,
            use_accelerated = False,
            return_advantages = True
        )

    # flatten rollout and run ppo epochs

    flat_obs = rearrange(obs_seq, 't e d -> (e t) d')
    flat_actions = rearrange(action_seq, 't e d -> (e t) d')
    flat_log_probs = rearrange(log_prob_seq, 't e -> (e t)')
    flat_values = rearrange(values, 'e t -> (e t)')
    flat_advantages = rearrange(gae.advantages, 'e t -> (e t)')
    flat_returns = rearrange(gae.returns, 'e t -> (e t)')

    num_samples = size(flat_obs, '[n] ...')

    for _ in range(10):
        perm = torch.randperm(num_samples, device = device)

        for start in range(0, num_samples, 256):
            idx = perm[start : start + 256]

            dist = agent.dist(flat_obs[idx])
            new_log_probs = dist.log_prob(flat_actions[idx]).sum(dim = -1)

            actor_loss = ppo_actor_loss(
                new_log_probs,
                flat_log_probs[idx],
                flat_advantages[idx],
                eps_clip = 0.2,
                normalize_advantages = True
            ).mean()

            critic_loss = clipped_value_loss_fn(
                agent.value(flat_obs[idx]),
                flat_values[idx],
                flat_returns[idx],
                clip = 0.2
            )

            entropy = dist.entropy().sum(dim = -1).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.005 * entropy

            step_optimizer(opt, loss, agent.parameters(), max_norm = 0.5)

# stage 1: x-ppo expert

@torch.no_grad()
def rollout_expert(env, agent: PPOAgent, normalize_fn, device, seed: int):
    obs, _ = env.reset(seed = seed)
    trajectory = [(env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy(), obs.copy())]
    total_return = 0.

    while True:
        action = agent.dist(as_tensor(normalize_fn(obs), device = device)).mean.cpu().numpy()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_return += float(reward)
        trajectory.append((env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy(), obs.copy()))

        if terminated or truncated:
            qpos, qvel, states = (np.asarray(col, dtype = np.float32) for col in zip(*trajectory))
            return total_return, qpos, qvel, states

def train_xppo_expert(
    env_name: str,
    mocap_path: Path,
    device,
    seed = 42,
    total_steps = 400_000,
    target_return = 3500.,
    num_clips = 12,
    eval_clips = 4
) -> dict:
    print(f'\n=== Stage 1: Training x-ppo Expert on {env_name} ===', flush = True)
    seed_everything(seed)
    start_time = time.time()

    num_envs = 8
    rollout_steps = 256

    envs = gym.make_vec(
        env_name,
        num_envs = num_envs,
        vectorization_mode = 'sync',
        vector_kwargs = dict(autoreset_mode = gym.vector.AutoresetMode.SAME_STEP)
    )
    eval_env = gym.make(env_name)

    dim_obs = envs.single_observation_space.shape[0]
    dim_action = envs.single_action_space.shape[0]
    bounds = np.stack((envs.single_action_space.low, envs.single_action_space.high), axis = -1)

    agent = PPOAgent(dim_obs, dim_action, bounds).to(device)
    rms = RunningMeanStd(shape = (dim_obs,))

    def normalize(x: np.ndarray) -> np.ndarray:
        return np.clip((x - rms.mean) / np.sqrt(rms.var + 1e-8), -10., 10.).astype(np.float32)

    num_updates = max(1, total_steps // (num_envs * rollout_steps))
    opt = AdamW(agent.parameters(), lr = 3e-4, eps = 1e-5)
    scheduler = CosineAnnealingLR(opt, T_max = num_updates, eta_min = 3e-5)

    obs, _ = envs.reset(seed = seed)
    rms.update(obs)

    best_eval = -float('inf')
    best_state = None
    steps_done = 0

    def get_obs() -> Tensor:
        return as_tensor(normalize(obs), device = device)

    def step_env(actions: np.ndarray):
        nonlocal obs, steps_done
        obs, rewards, terminated, truncated, _ = envs.step(actions)
        rms.update(obs)
        steps_done += num_envs
        return rewards, (terminated | truncated).astype(np.float32)

    for update in range(1, num_updates + 1):
        ppo_update(agent, opt, get_obs, step_env, rollout_steps, device)
        scheduler.step()

        if divisible_by(update, 10) or update == num_updates:
            with temp_eval(agent):
                eval_returns = [rollout_expert(eval_env, agent, normalize, device, 10_000 + i)[0] for i in range(5)]

            mean_ret = float(np.mean(eval_returns))
            std_ret = float(np.std(eval_returns))
            elapsed = time.time() - start_time

            print(
                f'[x-ppo expert] update {update:3d}/{num_updates} | steps {steps_done:7d} | '
                f'eval {mean_ret:7.1f} ± {std_ret:5.1f} | {elapsed:.1f}s',
                flush = True
            )

            if mean_ret > best_eval:
                best_eval = mean_ret
                best_state = (copy.deepcopy(agent.state_dict()), copy.deepcopy(rms))

            if mean_ret >= target_return:
                break

    envs.close()

    if exists(best_state):
        agent.load_state_dict(best_state[0])
        rms = best_state[1]

    # record train and held-out eval mocap clips

    with temp_eval(agent):
        clips = [rollout_expert(eval_env, agent, normalize, device, 50_000 + i) for i in range(num_clips + eval_clips)]

    eval_env.close()

    returns, qpos, qvel, states = zip(*clips)
    qpos = np.asarray(qpos, dtype = np.float32)
    qvel = np.asarray(qvel, dtype = np.float32)
    states = np.asarray(states, dtype = np.float32)
    refs = states[..., :REF_DIM]

    dataset = dict(
        env_name = env_name,
        expert_return = float(np.mean(returns[num_clips:])),
        expert_steps = steps_done,
        train_qpos = qpos[:num_clips],
        train_qvel = qvel[:num_clips],
        train_states = states[:num_clips],
        train_refs = refs[:num_clips],
        eval_qpos = qpos[num_clips:],
        eval_qvel = qvel[num_clips:],
        eval_states = states[num_clips:],
        eval_refs = refs[num_clips:]
    )

    torch.save(dataset, mocap_path)
    print(
        f'[x-ppo expert] saved {num_clips} train + {eval_clips} eval clips (return {dataset["expert_return"]:.1f})',
        flush = True
    )
    return dataset

# stage 2: supermimic tracker (world-model BPTT + analytical reward on 8-dim pose subset)

class TrackingActor(Module):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        dim_ref: int,
        obs_window = 2,
        action_window = 1,
        goal_window = GOAL_WINDOW,
        dim_hidden = 256,
        depth = 3,
        dim_explorative_latent = 0
    ):
        super().__init__()
        self.dim_action = dim_action

        self.obs_history = HistoryWindow(obs_window, dim_state, pad_mode = 'replicate')
        self.action_history = HistoryWindow(action_window, dim_action, lag = 1, pad_mode = 'zero')
        self.goal_window = HistoryWindow(goal_window, dim_ref, future = True, pad_mode = 'replicate')

        self.dim_in = obs_window * dim_state + action_window * dim_action + goal_window * dim_ref + dim_explorative_latent
        self.net = NormedMLP(self.dim_in, *([dim_hidden] * depth), dim_action, activation = SiLU())

        init.orthogonal_(self.net.layers[-1][0].weight, gain = 0.01)
        init.zeros_(self.net.layers[-1][0].bias)

    def forward(self, x: Tensor | tuple[Tensor, ...]) -> Tensor:
        if isinstance(x, (list, tuple)):
            x = torch.cat(x, dim = -1)
        pad = self.dim_in - size(x, '... [d]')
        if pad > 0:
            x = pad_right_at_dim(x, pad, dim = -1)
        return self.net(x).tanh()

class RolloutPolicy:
    def __init__(self, actor: TrackingActor, batch_size: int, device, noise = 0.):
        self.actor = actor
        self.batch_size = batch_size
        self.device = device
        self.noise = noise

        self.obs_cache = None
        self.action_cache = actor.action_history.init_cache(batch_size, device = device)

    def __call__(self, state: Tensor, goal_window: Tensor, random = False) -> Tensor:
        obs, self.obs_cache = self.actor.obs_history.step(state, self.obs_cache)
        past_actions = self.actor.action_history.read(self.action_cache)

        if random:
            action = torch.empty((self.batch_size, self.actor.dim_action), device = self.device).uniform_(-1., 1.)
        else:
            features = [rearrange(t, 'b w d -> b (w d)') for t in (obs, past_actions, goal_window)]
            action = self.actor(safe_cat(features, dim = -1))

            if self.noise > 0.:
                action = (action + torch.randn_like(action) * self.noise).clamp(-1., 1.)

        self.action_cache = self.actor.action_history.push(action, self.action_cache)
        return action

    def reset(self, done: Tensor):
        self.obs_cache = self.actor.obs_history.reset(self.obs_cache, done)
        self.action_cache = self.actor.action_history.reset(self.action_cache, done)

def train_supermimic_tracker(
    dataset: dict,
    out_path: Path,
    device,
    seed = 42,
    num_iterations = 40,
    num_explorative_rollouts = 2,
    dim_explorative_latent = 4,
    explorative_latent_scale = 0.28,
    explorative_latent_scale_end = 0.03,
    wm_num_explorative_rollouts: int | None = None,
    wm_dim_explorative_latent = 0,
    wm_explorative_latent_scale = 0.25,
    wm_explorative_latent_scale_end = 0.,
    wm_antithetic_explorative_latent = True,
    wm_imag_explorative_latent_scale = 0.,
    amp_weight = 0.,
    amp_every = 4,
    amp_prob = 1.,
    amp_warmup_steps = 0,
    amp_full_state = False,
    amp_gp_weight = 0.5,
    amp_disc_steps = 2
) -> dict:
    print(f'\n=== Stage 2: Training SuperMimic Tracker (Analytical Rewards on {REF_DIM}-d Pose Subset) ===', flush = True)
    seed_everything(seed)
    start_time = time.time()

    num_envs = 16
    wm_horizon = 8
    imag_horizon = 24
    policy_updates = 4
    batch_size = 256
    active_latent_dim = dim_explorative_latent if num_explorative_rollouts > 1 else 0
    wm_k = default(wm_num_explorative_rollouts, 2 if wm_dim_explorative_latent > 0 else 1)
    active_wm_latent_dim = wm_dim_explorative_latent if wm_k > 1 else 0

    with (
        MocapEnvs(dataset, 'train', device, num_envs = num_envs, horizon = 128) as train_envs,
        MocapEnvs(dataset, 'eval', device, render = True) as eval_envs
    ):
        # models

        actor = TrackingActor(
            train_envs.dim_state,
            train_envs.dim_action,
            REF_DIM,
            dim_explorative_latent = active_latent_dim
        ).to(device)
        ema_actor = EMA(actor, beta = 0.98, include_online_model = False, update_after_step = 0, update_every = 1).to(device)

        world_model = WorldModel(
            dim_state = train_envs.dim_state,
            dim_action = train_envs.dim_action,
            dim_sim_context = 0,
            dim_hiddens = (512, 512, 512),
            activation = SiLU(),
            normed_mlp = True,
            pred_residual_state = True,
            pred_reward = False,
            pred_terminal = False,
            num_explorative_rollouts = wm_k,
            dim_explorative_latent = active_wm_latent_dim,
            explorative_latent_scale = wm_explorative_latent_scale,
            antithetic_explorative_latent = wm_antithetic_explorative_latent
        ).to(device)

        critic = Critic(
            dim_state = train_envs.dim_state,
            dim_goal = REF_DIM,
            dim_sim_context = 0,
            dim_hiddens = (256, 256),
            activation = SiLU(),
            normed_mlp = True
        ).to(device)

        reward_fn = make_reward(
            as_tensor(train_envs.mean, device = device),
            as_tensor(train_envs.std, device = device)
        ).to(device)
        raw_reward_fn = make_reward().to(device)

        supermimic = SuperMimic(
            actor = actor,
            world_model = world_model,
            critic = critic,
            use_ema_critic = True,
            critic_ema_decay = 0.99,
            obs_history = actor.obs_history,
            action_history = actor.action_history,
            goal_window = actor.goal_window,
            reward_fn = reward_fn,
            discount_factor = 0.99,
            num_explorative_rollouts = num_explorative_rollouts,
            dim_explorative_latent = active_latent_dim,
            explorative_latent_scale = explorative_latent_scale,
            wm_explorative_latent_scale = wm_imag_explorative_latent_scale
        ).to(device)

        motion_critic = motion_critic_opt = norm_train_states = None
        model = supermimic

        if amp_weight > 0.:
            motion_critic = AdversarialCritic(
                dim_feature = train_envs.dim_state if amp_full_state else REF_DIM,
                feature_fn = None if amp_full_state else slice(0, REF_DIM),
                dim_hiddens = (256, 256),
                activation = SiLU(),
                r1_weight = amp_gp_weight,
                r2_weight = amp_gp_weight
            ).to(device)
            model = AdversarialSuperMimic(supermimic, motion_critic = motion_critic, motion_reward_weight = 0.).to(device)
            motion_critic_opt = AdamW(motion_critic.parameters(), lr = 5e-4, weight_decay = 1e-5)
            norm_train_states = as_tensor(train_envs.normalize(dataset['train_states']), device = device)

        # optimizers

        world_model_opt = AdamW(world_model.parameters(), lr = 5e-4, weight_decay = 1e-5)
        actor_opt = AdamW(actor.parameters(), lr = 3e-4, betas = (0.9, 0.95), weight_decay = 1e-5)
        actor_scheduler = CosineAnnealingLR(actor_opt, T_max = num_iterations * policy_updates, eta_min = 3e-5)
        critic_opt = AdamW(critic.parameters(), lr = 3e-4, weight_decay = 1e-5)

        # replay buffer

        buffer = StreamReplayBuffer(
            num_streams = num_envs,
            capacity = 64_000,
            fields = dict(
                state = ('float', train_envs.dim_state),
                action = ('float', train_envs.dim_action),
                next_state = ('float', train_envs.dim_state),
                clip_id = 'int',
                frame_idx = 'int'
            )
        )

        train_policy = RolloutPolicy(actor, num_envs, device, noise = 0.1)
        total_env_steps = 0
        history = []

        def collect_steps(num_steps: int, random = False):
            nonlocal total_env_steps

            with temp_eval(actor), torch.no_grad():
                for _ in range(num_steps):
                    norm_states = train_envs.normalize()
                    clips = train_envs.clips.copy()
                    frames = train_envs.frames.copy()

                    actions = train_policy(
                        as_tensor(norm_states, device = device),
                        train_envs.goals(),
                        random = random
                    ).cpu().numpy()

                    next_states, _, dones = train_envs.step(actions)
                    norm_next_states = train_envs.normalize(next_states)

                    for i in range(num_envs):
                        buffer.store_step(
                            i,
                            state = norm_states[i],
                            action = actions[i],
                            next_state = norm_next_states[i],
                            clip_id = int(clips[i]),
                            frame_idx = int(frames[i])
                        )

                        if dones[i]:
                            buffer.end_episode(i)

                    total_env_steps += num_envs

                    if dones.any():
                        train_policy.reset(as_tensor(dones, device = device))

        def fit_world_model(num_updates: int, latent_scale: float | None = None) -> float:
            world_model.train()
            fields = ('state', 'action', 'next_state')

            for _ in range(num_updates):
                batch = buffer.sample(batch_size, horizon = wm_horizon, to_named_tuple = fields, device = device)
                wm_loss, _, _ = world_model.rollout(
                    batch.state[:, 0],
                    batch.action,
                    batch.next_state,
                    explorative_latent_scale = latent_scale
                )
                loss = wm_loss / wm_horizon
                step_optimizer(world_model_opt, loss, world_model.parameters())

            return float(loss.item())

        def eval_ema(iteration: int | None = None, wm_loss = 0., render = False) -> dict:
            eval_policy = RolloutPolicy(ema_actor.ema_model, eval_envs.num_envs, device)

            with temp_eval(ema_actor.ema_model):
                metrics = evaluate_tracker(eval_envs, eval_policy, raw_reward_fn, device, render = render)

            if exists(iteration):
                history.append((total_env_steps, metrics['norm_return'], metrics['env_return']))
                elapsed = time.time() - start_time
                print(
                    f'[SuperMimic] iter {iteration:3d} | steps {total_env_steps:6d} | wm {wm_loss:.5f} | '
                    f'track {metrics["norm_return"]:.3f} | rmse {metrics["pose_rmse"]:.3f} | '
                    f'env_ret {metrics["env_return"]:7.1f} (expert {dataset["expert_return"]:.1f}) | {elapsed:.1f}s',
                    flush = True
                )

            return metrics

        # warmup with random exploration and pretrain world model

        collect_steps(160, random = True)
        wm_loss = fit_world_model(300)
        eval_ema(0, wm_loss)

        ckpt_path = out_path / 'supermimic_tracker_actor.pt'
        best_norm = -float('inf')
        seed_fields = ('state', 'action', 'clip_id', 'frame_idx')

        # main training loop

        for iteration in range(1, num_iterations + 1):
            # collect on-policy transitions and refit world model

            frac = (iteration - 1) / max(1, num_iterations - 1)
            cur_wm_latent_scale = cosine_anneal(wm_explorative_latent_scale, wm_explorative_latent_scale_end, frac)

            collect_steps(16)
            wm_loss = fit_world_model(10, latent_scale = cur_wm_latent_scale)

            # imagine through frozen world model and update actor via bptt

            actor.train()
            cur_latent_scale = cosine_anneal(explorative_latent_scale, explorative_latent_scale_end, frac)

            for update_idx in range(policy_updates):
                global_update = (iteration - 1) * policy_updates + update_idx + 1

                seed_batch = buffer.sample(batch_size, horizon = 2, to_named_tuple = seed_fields, device = device)
                clip_ids = seed_batch.clip_id[:, -1].long()
                frame_ids = seed_batch.frame_idx[:, -1].long()

                goals = gather_refs(train_envs.refs, clip_ids, frame_ids + 1, imag_horizon + GOAL_WINDOW)
                critic_goals = goals[:, :imag_horizon + 1]

                amp_kwargs = dict()
                if exists(motion_critic):
                    active = divisible_by(global_update, amp_every) and (amp_prob >= 1. or np.random.rand() < amp_prob)
                    ramp = min(1., global_update / amp_warmup_steps) if amp_warmup_steps > 0 else 1.
                    amp_kwargs['motion_reward_weight'] = amp_weight * ramp if active else 0.

                rollout = model(
                    seed_batch.state[:, -1],
                    actor_goals = goals,
                    references = goals[:, :imag_horizon],
                    past_obs = seed_batch.state[:, :-1],
                    past_actions = seed_batch.action[:, :-1],
                    num_rollout_timesteps = imag_horizon,
                    explorative_latent_scale = cur_latent_scale,
                    **amp_kwargs
                )

                actor_loss = model.actor_loss(
                    rollout.rewards,
                    final_state = rollout.states[:, -1],
                    final_goal = critic_goals[:, -1]
                )

                step_optimizer(actor_opt, actor_loss, actor.parameters(), clip_value = 0.1)
                actor_scheduler.step()
                ema_actor.update()

                # fit critic on task rewards (stationary when AMP is titrated in occasionally)

                critic_rewards = getattr(rollout, 'task_rewards', rollout.rewards)

                for _ in range(4):
                    critic_loss = model.critic_loss(rollout.states, critic_rewards, goals = critic_goals, lam = 0.95)
                    step_optimizer(critic_opt, critic_loss, critic.parameters())
                    model.update_ema()

                # update AMP discriminator on active steps (after actor backward)

                if exists(motion_critic) and exists(rollout.motion_rewards):
                    motion_critic.train()
                    fake_states = rollout.states.detach()

                    for _ in range(amp_disc_steps):
                        num_ref_clips, num_ref_frames = shape(norm_train_states, '[c t] d')
                        ref_clips = torch.randint(0, num_ref_clips, (batch_size,), device = device)
                        ref_starts = torch.randint(0, num_ref_frames - imag_horizon - 1, (batch_size,), device = device)
                        ref_states = gather_refs(norm_train_states, ref_clips, ref_starts, imag_horizon + 1)
                        step_optimizer(motion_critic_opt, model.motion_critic_loss(ref_states, fake_states), motion_critic.parameters())

            if divisible_by(iteration, 2) or iteration == num_iterations:
                metrics = eval_ema(iteration, wm_loss)

                if metrics['norm_return'] > best_norm:
                    best_norm = metrics['norm_return']
                    torch.save(ema_actor.ema_model.state_dict(), ckpt_path)

        if ckpt_path.exists():
            ema_actor.ema_model.load_state_dict(torch.load(ckpt_path, map_location = device, weights_only = True))

        final = eval_ema(render = True)

    print(
        f'[SuperMimic final] track={final["norm_return"]:.3f} | rmse={final["pose_rmse"]:.3f} | '
        f'env_return={final["env_return"]:.1f} ± {final["env_return_std"]:.1f} '
        f'(expert {dataset["expert_return"]:.1f})',
        flush = True
    )
    return dict(history = history, final = final)

# stage 3 (optional): x-ppo tracker baseline on identical 8-dim analytical reward

def train_ppo_tracker_baseline(dataset: dict, device, total_steps: int, seed = 42) -> list:
    print(f'\n=== Stage 3: Baseline x-ppo Tracker ({REF_DIM}-d Analytical Reward, {total_steps} steps) ===', flush = True)
    seed_everything(seed)
    start_time = time.time()

    num_envs = 16
    rollout_steps = 128

    def encode(states: Tensor, goal_windows: Tensor) -> Tensor:
        return safe_cat((states, rearrange(goal_windows, 'b w d -> b (w d)')), dim = -1)

    def to_seq(arr: np.ndarray) -> Tensor:
        return rearrange(as_tensor(arr, device = device), 'b ... -> b 1 ...')

    with (
        MocapEnvs(dataset, 'train', device, num_envs = num_envs, horizon = 128) as train_envs,
        MocapEnvs(dataset, 'eval', device) as eval_envs
    ):
        agent = PPOAgent(train_envs.dim_state + GOAL_WINDOW * REF_DIM, train_envs.dim_action, train_envs.bounds).to(device)
        opt = AdamW(agent.parameters(), lr = 3e-4, eps = 1e-5)
        reward_fn = make_reward().to(device)

        prev_actions = np.zeros((num_envs, train_envs.dim_action), dtype = np.float32)
        steps_done = 0
        history = []

        def get_obs() -> Tensor:
            return encode(as_tensor(train_envs.normalize(), device = device), train_envs.goals())

        def step_env(actions: np.ndarray):
            nonlocal steps_done

            clips = train_envs.clips.copy()
            frames = train_envs.frames.copy()
            has_prev = train_envs.episode_steps > 0

            next_states, _, dones = train_envs.step(actions)

            with torch.no_grad():
                step_rewards = reward_fn(
                    to_seq(next_states),
                    to_seq(dataset['train_refs'][clips, frames + 1]),
                    actions = to_seq(actions),
                    past_actions = to_seq(prev_actions),
                    past_mask = to_seq(has_prev)
                )
                step_rewards = rearrange(step_rewards, 'b 1 -> b').cpu().numpy()

            prev_actions[:] = einx.where('b, , b d -> b d', dones, np.float32(0.), actions)
            steps_done += num_envs
            return step_rewards, dones.astype(np.float32)

        def eval_policy(obs: Tensor, goal_window: Tensor) -> Tensor:
            return agent.dist(encode(obs, goal_window)).mean

        def log_eval(label: str):
            with temp_eval(agent):
                metrics = evaluate_tracker(eval_envs, eval_policy, reward_fn, device)

            history.append((steps_done, metrics['norm_return'], metrics['env_return']))

            if label:
                elapsed = time.time() - start_time
                print(
                    f'[x-ppo tracker] {label} | steps {steps_done:6d} | track {metrics["norm_return"]:.3f} | '
                    f'env_ret {metrics["env_return"]:7.1f} | {elapsed:.1f}s',
                    flush = True
                )

        num_updates = max(1, total_steps // (num_envs * rollout_steps))
        eval_every = max(1, num_updates // 8)
        log_eval('')

        for update in range(1, num_updates + 1):
            ppo_update(agent, opt, get_obs, step_env, rollout_steps, device)

            if divisible_by(update, eval_every) or update == num_updates:
                log_eval(f'update {update:3d}/{num_updates}')

    return history

# plotting & comparison videos

def add_banner(frames: list[np.ndarray], title: str, subtitle: str, accent: tuple[int, int, int]) -> list[np.ndarray]:
    bar = Image.new('RGB', (frames[0].shape[1], 50), (15, 23, 42))
    draw = ImageDraw.Draw(bar)
    draw.text((12, 6), title, fill = (248, 250, 252), font = ImageFont.load_default(size = 15))
    draw.text((12, 27), subtitle, fill = (203, 213, 225), font = ImageFont.load_default(size = 12))
    draw.rectangle((0, 47, bar.width, 50), fill = accent)

    bar_arr = np.asarray(bar)
    return [np.concatenate((bar_arr, frame), axis = 0) for frame in frames]

def save_mp4(path: Path, frames: list[np.ndarray], fps = 25):
    imageio.mimsave(
        path,
        frames,
        fps = fps,
        codec = 'libx264',
        pixelformat = 'yuv420p',
        macro_block_size = 1,
        output_params = ['-movflags', '+faststart']
    )

def save_comparison_videos(dataset: dict, final: dict, out_path: Path, fps = 25):
    env = gym.make(dataset['env_name'], render_mode = 'rgb_array', camera_name = 'track')
    env.reset()

    expert_raw = []
    for qpos, qvel in zip(dataset['eval_qpos'][0, ::4], dataset['eval_qvel'][0, ::4]):
        env.unwrapped.set_state(qpos, qvel)
        expert_raw.append(env.render())

    env.close()

    expert_frames = add_banner(
        expert_raw,
        'x-ppo Expert (Task Reward)',
        f'HalfCheetah-v5 held-out clip  |  Return: {dataset["expert_return"]:.0f}',
        accent = (22, 163, 74)
    )

    mimic_subtitle = (
        f'Track: {final["norm_return"]:.3f}  |  RMSE: {final["pose_rmse"]:.3f}  |  '
        f'Unseen Return: {final["env_return"]:.0f}'
    )
    mimic_frames = add_banner(
        final['frames'],
        f'SuperMimic Tracker ({REF_DIM}-d Pose Only, No Task Reward)',
        mimic_subtitle,
        accent = (37, 99, 235)
    )

    divider = np.full((expert_frames[0].shape[0], 4, 3), (15, 23, 42), dtype = np.uint8)
    side_by_side = [np.concatenate((left, divider, right), axis = 1) for left, right in zip(expert_frames, mimic_frames)]

    save_mp4(out_path / 'xppo_expert_cheetah.mp4', expert_frames, fps = fps)
    save_mp4(out_path / 'supermimic_tracker_cheetah.mp4', mimic_frames, fps = fps)
    save_mp4(out_path / 'expert_vs_supermimic_side_by_side.mp4', side_by_side, fps = fps)

def annotate_end(ax, x: float, y: float, text: str, color: str, offset: tuple[int, int], ha = 'left'):
    ax.annotate(
        text,
        xy = (x, y),
        xytext = offset,
        textcoords = 'offset points',
        ha = ha,
        color = color,
        fontsize = 8.5,
        fontweight = 'bold'
    )

def save_convergence_plot(sm_history: list, ppo_history: list | None, expert_return: float, out_path: Path):
    fig, (ax_track, ax_return) = plt.subplots(1, 2, figsize = (12.2, 4.9), dpi = 150)

    suffix = 'SuperMimic BPTT vs x-ppo Tracker' if exists(ppo_history) else 'SuperMimic World-Model BPTT'
    fig.suptitle(
        f'HalfCheetah-v5: x-ppo Expert → {REF_DIM}-d Pose-Only Reference Subset → {suffix}',
        fontsize = 11.5,
        fontweight = 'bold'
    )

    sm_steps, sm_track, sm_ret = zip(*sm_history)
    sm_k = [step / 1e3 for step in sm_steps]

    ppo_k = ppo_track = ppo_ret = None
    if exists(ppo_history):
        ppo_steps, ppo_track, ppo_ret = zip(*ppo_history)
        ppo_k = [step / 1e3 for step in ppo_steps]

    max_k = max(sm_k[-1], ppo_k[-1] if exists(ppo_k) else 0.)

    panels = (
        (ax_track, sm_track, ppo_track, '{:.3f}', 'Normalized Tracking Return (0–1)', 'Pose-Subset Tracking (Held-Out)', ''),
        (ax_return, sm_ret, ppo_ret, '{:.0f}', 'HalfCheetah-v5 Return', 'Recovered Task Return (Unseen)', ' — no task reward')
    )

    for ax, sm_y, ppo_y, fmt, ylabel, title, note in panels:
        sm_label = f'SuperMimic BPTT{note} ({fmt.format(sm_y[-1])} @ {sm_k[-1]:.1f}k)'
        ax.plot(sm_k, sm_y, color = '#1d4ed8', marker = 'o', markersize = 4, linewidth = 2.2, label = sm_label)
        annotate_end(ax, sm_k[-1], sm_y[-1], fmt.format(sm_y[-1]), '#1d4ed8', (6, -10 if ax is ax_track else -4))

        if exists(ppo_y):
            ppo_label = f'x-ppo Tracker{note} ({fmt.format(ppo_y[-1])} @ {ppo_k[-1]:.1f}k)'
            ax.plot(
                ppo_k,
                ppo_y,
                color = '#dc2626',
                marker = 's',
                markersize = 4,
                linewidth = 2.,
                linestyle = '--',
                label = ppo_label
            )
            annotate_end(ax, ppo_k[-1], ppo_y[-1], fmt.format(ppo_y[-1]), '#dc2626', (-4, 8), ha = 'right')

        if ax is ax_track:
            ax.set_ylim(0., 1.08)
        else:
            expert_label = f'x-ppo Expert Reference ({expert_return:.0f})'
            ax.axhline(expert_return, color = '#16a34a', linestyle = ':', linewidth = 1.9, label = expert_label)

        ax.set_xlim(0., max_k * 1.12)
        ax.set_xlabel('Environment Steps (×1,000)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha = 0.25)
        ax.legend(loc = 'center right' if (ax is ax_return and exists(ppo_history)) else 'lower right', fontsize = 8.2)

    fig.tight_layout(rect = (0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)

# main

def main(
    env_name = 'HalfCheetah-v5',
    out_dir = './results_supermimic_cheetah',
    seed = 42,
    reuse_expert = True,
    expert_steps = 400_000,
    expert_target_return = 3500.,
    num_iterations = 40,
    num_explorative_rollouts = 2,
    dim_explorative_latent = 4,
    explorative_latent_scale = 0.28,
    explorative_latent_scale_end = 0.03,
    wm_num_explorative_rollouts: int | None = None,
    wm_dim_explorative_latent = 0,
    wm_explorative_latent_scale = 0.25,
    wm_explorative_latent_scale_end = 0.,
    wm_antithetic_explorative_latent = True,
    wm_imag_explorative_latent_scale = 0.,
    amp_weight = 0.,
    amp_every = 4,
    amp_prob = 1.,
    amp_warmup_steps = 0,
    amp_full_state = False,
    amp_gp_weight = 0.5,
    amp_disc_steps = 2,
    compare_ppo = True,
    ppo_tracker_steps = 50_000
):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f'using device: {device}', flush = True)

    out_path = Path(out_dir)
    out_path.mkdir(parents = True, exist_ok = True)

    mocap_path = out_path / 'xppo_expert_mocap.pt'

    if reuse_expert and mocap_path.exists():
        print(f'loading cached expert mocap from {mocap_path}', flush = True)
        dataset = torch.load(mocap_path, weights_only = False)
    else:
        dataset = train_xppo_expert(
            env_name,
            mocap_path,
            device,
            seed = seed,
            total_steps = expert_steps,
            target_return = expert_target_return
        )

    sm_result = train_supermimic_tracker(
        dataset,
        out_path,
        device,
        seed = seed,
        num_iterations = num_iterations,
        num_explorative_rollouts = num_explorative_rollouts,
        dim_explorative_latent = dim_explorative_latent,
        explorative_latent_scale = explorative_latent_scale,
        explorative_latent_scale_end = explorative_latent_scale_end,
        wm_num_explorative_rollouts = wm_num_explorative_rollouts,
        wm_dim_explorative_latent = wm_dim_explorative_latent,
        wm_explorative_latent_scale = wm_explorative_latent_scale,
        wm_explorative_latent_scale_end = wm_explorative_latent_scale_end,
        wm_antithetic_explorative_latent = wm_antithetic_explorative_latent,
        wm_imag_explorative_latent_scale = wm_imag_explorative_latent_scale,
        amp_weight = amp_weight,
        amp_every = amp_every,
        amp_prob = amp_prob,
        amp_warmup_steps = amp_warmup_steps,
        amp_full_state = amp_full_state,
        amp_gp_weight = amp_gp_weight,
        amp_disc_steps = amp_disc_steps
    )

    ppo_history = None
    if compare_ppo:
        ppo_steps = max(sm_result['history'][-1][0], ppo_tracker_steps)
        ppo_history = train_ppo_tracker_baseline(dataset, device, total_steps = ppo_steps, seed = seed)

    plot_path = out_path / 'supermimic_vs_xppo_convergence.png'
    save_convergence_plot(sm_result['history'], ppo_history, dataset['expert_return'], plot_path)
    save_comparison_videos(dataset, sm_result['final'], out_path)
    print(f'\nsaved plot & comparison videos to {out_path}', flush = True)

if __name__ == '__main__':
    fire.Fire(main)
