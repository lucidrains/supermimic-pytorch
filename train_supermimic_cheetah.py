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
#     "matplotlib",
#     "numpy",
#     "pillow",
# ]
# ///

from __future__ import annotations

import copy
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
from torch_einops_utils import safe_cat, temp_eval
from ema_pytorch import EMA
from mean_conc_beta import Beta
from memmap_replay_buffer import StreamReplayBuffer
from x_mlps_pytorch import MLP
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP
from x_ppo import calc_gae, clipped_value_loss_fn, ppo_actor_loss

from supermimic_pytorch import (
    ActionSmoothnessTerm,
    AnalyticalReward,
    Critic,
    HistoryWindow,
    SuperMimic,
    TrackingTerm,
    WorldModel
)

# constants & helpers

REF_DIM = 8  # [root_z, torso_pitch, 6 joint angles] — all 9 velocities hidden
GOAL_WINDOW = 8
MAX_TRACK_REWARD = 4.0

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def step_opt(opt, loss: Tensor, params, max_norm = 1., clip_val = None):
    params = list(params)
    opt.zero_grad()
    loss.backward()
    clip_grad_norm_(params, max_norm)
    if exists(clip_val):
        clip_grad_value_(params, clip_val)
    opt.step()

def make_reward(mean: Tensor | None = None, std: Tensor | None = None, smoothness = -0.01) -> AnalyticalReward:
    unnorm = (lambda s: s * std.to(s) + mean.to(s)) if exists(mean) and exists(std) else None
    terms = dict(
        root_height = TrackingTerm(0, std = 0.25, weight = 1.),
        torso_pitch = TrackingTerm(1, metric = 'angle', std = 0.35, weight = 1.),
        joint_pos = TrackingTerm(slice(2, REF_DIM), std = 0.75, weight = 2.),
        action_smoothness = ActionSmoothnessTerm(weight = smoothness)
    )
    return AnalyticalReward(terms, state_transform = unnorm)

def gather_refs(refs: Tensor, clip_ids: Tensor, starts: Tensor, length: int) -> Tensor:
    offsets = torch.arange(length, device = refs.device)
    frames = einx.add('b, l -> b l', starts, offsets).clamp(max = refs.shape[1] - 1)
    return einx.get_at('[c] [t] d, b, b l -> b l d', refs, clip_ids, frames)

# parallel mocap envs with reference state initialization (RSI)

class MocapEnvs:
    def __init__(self, dataset: dict, split: str, device, num_envs = None, horizon = None, render = False):
        self.qpos, self.qvel, self.refs = dataset[f'{split}_qpos'], dataset[f'{split}_qvel'], dataset[f'{split}_refs']
        self.num_clips, self.total_frames = self.refs.shape[:2]
        self.num_envs = default(num_envs, self.num_clips)
        self.horizon = default(horizon, self.total_frames - 1)
        self.max_start = max(1, self.total_frames - self.horizon - 2)
        self.device = device

        flat = rearrange(dataset['train_states'], 'c t d -> (c t) d')
        self.mean = flat.mean(0).astype(np.float32)
        self.std = np.maximum(flat.std(0), 1e-2).astype(np.float32)

        self.refs_t = as_tensor(self.refs, device = device)
        gw_mod = HistoryWindow(GOAL_WINDOW, REF_DIM, future = True, pad_mode = 'replicate')
        self.goal_windows = gw_mod(self.refs_t[:, 1:])

        name = dataset['env_name']
        self.envs = [
            gym.make(name, render_mode = 'rgb_array', camera_name = 'track') if (render and i == 0) else gym.make(name)
            for i in range(self.num_envs)
        ]
        self.dim_state = int(self.envs[0].observation_space.shape[0])
        self.dim_action = int(self.envs[0].action_space.shape[0])
        self.bounds = np.stack((self.envs[0].action_space.low, self.envs[0].action_space.high), axis = -1)

        self.clip, self.frame, self.ep_step = (np.zeros(self.num_envs, np.int64) for _ in range(3))
        self.state = np.zeros((self.num_envs, self.dim_state), np.float32)

        for i in range(self.num_envs):
            self.reset(i, clip = i if split == 'eval' else None, frame = 0 if split == 'eval' else None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for env in self.envs:
            env.close()

    def reset(self, i: int, clip = None, frame = None):
        c = default(clip, int(np.random.randint(self.num_clips)))
        f = default(frame, int(np.random.randint(self.max_start)))
        self.envs[i].reset()
        self.envs[i].unwrapped.set_state(self.qpos[c, f], self.qvel[c, f])
        self.clip[i], self.frame[i], self.ep_step[i] = c, f, 0
        self.state[i] = self.envs[i].unwrapped._get_obs()

    def norm(self, state = None) -> np.ndarray:
        return (default(state, self.state) - self.mean) / self.std

    def goals(self) -> Tensor:
        clips = as_tensor(self.clip, device = self.device)
        frames = as_tensor(self.frame, device = self.device)
        return einx.get_at('[c] [t] w d, e, e -> e w d', self.goal_windows, clips, frames)

    def step(self, actions: np.ndarray, auto_reset = True):
        next_s = np.empty_like(self.state)
        rews = np.empty(self.num_envs)
        dones = np.zeros(self.num_envs, bool)

        for i, env in enumerate(self.envs):
            ns, r, term, trunc, _ = env.step(actions[i])
            next_s[i], rews[i] = ns, r
            self.frame[i] += 1
            self.ep_step[i] += 1

            reached_end = term or trunc or self.ep_step[i] >= self.horizon or self.frame[i] >= self.total_frames - 2
            if auto_reset and reached_end:
                dones[i] = True
                self.reset(i)
            else:
                self.state[i] = next_s[i]

        return next_s, rews, dones

@torch.no_grad()
def evaluate_tracker(dataset: dict, policy_fn, reward_fn: AnalyticalReward, device, render = False, stride = 4) -> dict:
    with MocapEnvs(dataset, 'eval', device, render = render) as vec:
        frames = [vec.envs[0].render()] if render else None
        next_states, actions = [], []
        env_rets = np.zeros(vec.num_envs)

        for t in range(vec.total_frames - 1):
            act = policy_fn(as_tensor(vec.norm(), device = device), vec.goal_windows[:, t]).cpu().numpy()
            ns, r, _ = vec.step(act, auto_reset = False)
            actions.append(act)
            next_states.append(ns)
            env_rets += r
            if exists(frames) and divisible_by(t + 1, stride):
                frames.append(vec.envs[0].render())

        refs_t = vec.refs_t

    ns_t = as_tensor(np.stack(next_states, axis = 1), device = device)
    act_t = as_tensor(np.stack(actions, axis = 1), device = device)
    _, bd = reward_fn(ns_t, refs_t[:, 1:], actions = act_t, return_breakdown = True)

    track = sum(v for k, v in bd.items() if k != 'action_smoothness')
    return dict(
        norm_return = float(track.mean() / MAX_TRACK_REWARD),
        pose_rmse = float((ns_t[..., :REF_DIM] - refs_t[:, 1:]).square().mean().sqrt()),
        env_return = float(env_rets.mean()),
        env_return_std = float(env_rets.std()),
        frames = frames
    )

# shared x-ppo agent & rollout/update step

class PPOAgent(Module):
    def __init__(self, dim_in: int, dim_action: int, bounds: np.ndarray, dim_hidden = 256):
        super().__init__()
        self.actor = MLP(dim_in, dim_hidden, dim_hidden, dim_action * 2, activation = Tanh())
        self.critic = MLP(dim_in, dim_hidden, dim_hidden, 1, activation = Tanh())
        for net, gain in ((self.actor, 0.01), (self.critic, 1.)):
            init.orthogonal_(net.layers[-1].weight, gain = gain)
            init.zeros_(net.layers[-1].bias)
        self.beta = Beta(bounds = bounds, pos_fn = 'softplus', init_conc = 2., detach_entropy_mean = True)

    def dist(self, x: Tensor):
        return self.beta(rearrange(self.actor(x), '... (d p) -> ... d p', p = 2))

    def value(self, x: Tensor) -> Tensor:
        return rearrange(self.critic(x), '... 1 -> ...')

def ppo_step(agent: PPOAgent, opt: AdamW, get_obs, step_env, rollout_steps: int, device):
    traj = []
    for _ in range(rollout_steps):
        x = get_obs()
        with torch.no_grad():
            dist = agent.dist(x)
            a = dist.sample()
            lp, v = dist.log_prob(a).sum(dim = -1), agent.value(x)

        r, d = step_env(a.cpu().numpy())
        r_t = as_tensor(r, dtype = torch.float32, device = device)
        d_t = as_tensor(d, dtype = torch.float32, device = device)
        traj.append((x, a, lp, r_t, v, d_t))

    obs, act, logp, rew, val, done = (list(col) for col in zip(*traj))
    rew_b, val_b, done_b = (rearrange(b, 't e -> e t') for b in (rew, val, done))

    with torch.no_grad():
        gae = calc_gae(
            rew_b, val_b, masks = 1. - done_b, next_value = agent.value(get_obs()),
            gamma = 0.99, lam = 0.95, use_accelerated = False, return_advantages = True
        )

    f_obs, f_act = (rearrange(b, 't e d -> (e t) d') for b in (obs, act))
    f_logp = rearrange(logp, 't e -> (e t)')
    f_val, f_adv, f_ret = (rearrange(x, 'e t -> (e t)') for x in (val_b, gae.advantages, gae.returns))

    for _ in range(10):
        perm = torch.randperm(len(f_obs), device = device)
        for s in range(0, len(f_obs), 256):
            idx = perm[s : s + 256]
            dist = agent.dist(f_obs[idx])
            log_probs = dist.log_prob(f_act[idx]).sum(dim = -1)
            a_loss = ppo_actor_loss(log_probs, f_logp[idx], f_adv[idx], eps_clip = 0.2, normalize_advantages = True).mean()
            v_loss = clipped_value_loss_fn(agent.value(f_obs[idx]), f_val[idx], f_ret[idx], clip = 0.2)
            loss = a_loss + 0.5 * v_loss - 0.005 * dist.entropy().sum(dim = -1).mean()
            step_opt(opt, loss, agent.parameters(), max_norm = 0.5)

# stage 1: x-ppo expert

@torch.no_grad()
def rollout_expert(env, agent: PPOAgent, norm_fn, device, seed: int):
    obs, _ = env.reset(seed = seed)
    traj = [(env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy(), obs.copy())]
    ep_ret = 0.

    while True:
        action = agent.dist(as_tensor(norm_fn(obs), device = device)).mean.cpu().numpy()
        obs, reward, term, trunc, _ = env.step(action)
        ep_ret += float(reward)
        traj.append((env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy(), obs.copy()))
        if term or trunc:
            return ep_ret, *(np.asarray(x, np.float32) for x in zip(*traj))

def train_xppo_expert(
    env_name: str, mocap_path: Path, device, seed = 42,
    total_steps = 400_000, target_return = 3500., num_clips = 12, eval_clips = 4
) -> dict:
    print(f'\n=== Stage 1: Training x-ppo Expert on {env_name} ===', flush = True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    num_envs = 8
    rollout_steps = 256
    envs = gym.make_vec(
        env_name, num_envs = num_envs, vectorization_mode = 'sync',
        vector_kwargs = dict(autoreset_mode = gym.vector.AutoresetMode.SAME_STEP)
    )
    eval_env = gym.make(env_name)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    bounds = np.stack((envs.single_action_space.low, envs.single_action_space.high), axis = -1)

    agent = PPOAgent(obs_dim, act_dim, bounds).to(device)
    rms = RunningMeanStd(shape = (obs_dim,))
    norm_obs = lambda x: np.clip((x - rms.mean) / np.sqrt(rms.var + 1e-8), -10., 10.).astype(np.float32)

    num_updates = max(1, total_steps // (num_envs * rollout_steps))
    opt = AdamW(agent.parameters(), lr = 3e-4, eps = 1e-5)
    scheduler = CosineAnnealingLR(opt, T_max = num_updates, eta_min = 3e-5)

    obs, _ = envs.reset(seed = seed)
    rms.update(obs)
    best_eval, best_state, steps_done = -float('inf'), None, 0

    def step_env(actions: np.ndarray):
        nonlocal obs, steps_done
        obs, rews, term, trunc, _ = envs.step(actions)
        rms.update(obs)
        steps_done += num_envs
        return rews, (term | trunc).astype(np.float32)

    for update in range(1, num_updates + 1):
        ppo_step(agent, opt, lambda: as_tensor(norm_obs(obs), device = device), step_env, rollout_steps, device)
        scheduler.step()

        if divisible_by(update, 10) or update == num_updates:
            with temp_eval(agent):
                rets = [rollout_expert(eval_env, agent, norm_obs, device, 10_000 + i)[0] for i in range(5)]
            mean_ret, std_ret = float(np.mean(rets)), float(np.std(rets))
            print(
                f'[x-ppo expert] update {update:3d}/{num_updates} | steps {steps_done:7d} | '
                f'eval {mean_ret:7.1f} ± {std_ret:5.1f} | {time.time() - t0:.1f}s',
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

    with temp_eval(agent):
        clips = [rollout_expert(eval_env, agent, norm_obs, device, 50_000 + c) for c in range(num_clips + eval_clips)]
    eval_env.close()

    returns, qpos, qvel, states = zip(*clips)
    arrays = dict(
        qpos = np.asarray(qpos, np.float32),
        qvel = np.asarray(qvel, np.float32),
        states = np.asarray(states, np.float32)
    )
    arrays['refs'] = arrays['states'][..., :REF_DIM]

    dataset = dict(
        env_name = env_name,
        expert_return = float(np.mean(returns[num_clips:])),
        expert_steps = steps_done,
        **{f'train_{k}': v[:num_clips] for k, v in arrays.items()},
        **{f'eval_{k}': v[num_clips:] for k, v in arrays.items()}
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
        self, dim_state: int, dim_action: int, dim_ref: int,
        obs_window = 2, action_window = 1, goal_window = GOAL_WINDOW, dim_hidden = 256, depth = 3
    ):
        super().__init__()
        self.dim_action = dim_action
        self.obs_history = HistoryWindow(obs_window, dim_state, pad_mode = 'replicate')
        self.action_history = HistoryWindow(action_window, dim_action, lag = 1, pad_mode = 'zero')
        self.goal_window = HistoryWindow(goal_window, dim_ref, future = True, pad_mode = 'replicate')

        dim_in = obs_window * dim_state + action_window * dim_action + goal_window * dim_ref
        self.net = NormedMLP(dim_in, *([dim_hidden] * depth), dim_action, activation = SiLU())
        init.orthogonal_(self.net.layers[-1][0].weight, gain = 0.01)
        init.zeros_(self.net.layers[-1][0].bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).tanh()

    def rollout_policy(self, batch_size: int, device, noise = 0.):
        obs_c = None
        act_c = self.action_history.init_cache(batch_size, device = device)

        def step(state: Tensor, goal_win: Tensor, random = False) -> Tensor:
            nonlocal obs_c, act_c
            obs, obs_c = self.obs_history.step(state, obs_c)
            parts = (obs, self.action_history.read(act_c), goal_win)
            x = safe_cat([rearrange(t, 'b w d -> b (w d)') for t in parts], dim = -1)

            if random:
                act = torch.empty((batch_size, self.dim_action), device = device).uniform_(-1., 1.)
            else:
                act = self(x)
                if noise > 0.:
                    act = (act + torch.randn_like(act) * noise).clamp(-1., 1.)

            act_c = self.action_history.push(act, act_c)
            return act

        def reset(done: Tensor):
            nonlocal obs_c, act_c
            obs_c = self.obs_history.reset(obs_c, done)
            act_c = self.action_history.reset(act_c, done)

        return step, reset

def train_supermimic_tracker(dataset: dict, out_path: Path, device, seed = 42, num_iterations = 40) -> dict:
    print(f'\n=== Stage 2: Training SuperMimic Tracker (Analytical Rewards on {REF_DIM}-d Pose Subset) ===', flush = True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    num_envs = 16
    wm_horizon = 8
    imag_horizon = 24
    policy_updates = 4
    batch_size = 256

    with MocapEnvs(dataset, 'train', device, num_envs = num_envs, horizon = 128) as vec:
        actor = TrackingActor(vec.dim_state, vec.dim_action, REF_DIM).to(device)
        ema_actor = EMA(actor, beta = 0.98, include_online_model = False, update_after_step = 0, update_every = 1).to(device)

        world_model = WorldModel(
            dim_state = vec.dim_state,
            dim_action = vec.dim_action,
            dim_sim_context = 0,
            dim_hiddens = (512, 512, 512),
            activation = SiLU(),
            normed_mlp = True,
            pred_residual_state = True,
            pred_reward = False,
            pred_terminal = False
        ).to(device)

        critic = Critic(
            dim_state = vec.dim_state,
            dim_goal = REF_DIM,
            dim_sim_context = 0,
            dim_hiddens = (256, 256),
            activation = SiLU(),
            normed_mlp = True
        ).to(device)

        reward_fn = make_reward(as_tensor(vec.mean, device = device), as_tensor(vec.std, device = device)).to(device)
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
            discount_factor = 0.99
        ).to(device)

        wm_opt = AdamW(world_model.parameters(), lr = 5e-4, weight_decay = 1e-5)
        actor_opt = AdamW(actor.parameters(), lr = 3e-4, betas = (0.9, 0.95), weight_decay = 1e-5)
        actor_sched = CosineAnnealingLR(actor_opt, T_max = num_iterations * policy_updates, eta_min = 3e-5)
        critic_opt = AdamW(critic.parameters(), lr = 3e-4, weight_decay = 1e-5)

        buffer = StreamReplayBuffer(
            num_streams = num_envs,
            capacity = 64_000,
            fields = dict(
                state = ('float', vec.dim_state),
                action = ('float', vec.dim_action),
                next_state = ('float', vec.dim_state),
                clip_id = 'int',
                frame_idx = 'int'
            )
        )

        step_policy, reset_policy = actor.rollout_policy(num_envs, device, noise = 0.1)
        total_env_steps = 0
        history = []

        def collect_steps(num_steps: int, random = False):
            nonlocal total_env_steps
            with temp_eval(actor), torch.no_grad():
                for _ in range(num_steps):
                    norm_s = vec.norm()
                    clips, frames = vec.clip.copy(), vec.frame.copy()
                    actions = step_policy(as_tensor(norm_s, device = device), vec.goals(), random = random).cpu().numpy()
                    next_s, _, dones = vec.step(actions)
                    norm_ns = vec.norm(next_s)

                    for i in range(num_envs):
                        buffer.store_step(
                            i, state = norm_s[i], action = actions[i], next_state = norm_ns[i],
                            clip_id = int(clips[i]), frame_idx = int(frames[i])
                        )
                        if dones[i]:
                            buffer.end_episode(i)

                    total_env_steps += num_envs
                    if dones.any():
                        reset_policy(as_tensor(dones, device = device))

        def fit_world_model(num_updates: int) -> float:
            world_model.train()
            fields = ('state', 'action', 'next_state')
            for _ in range(num_updates):
                b = buffer.sample(batch_size, horizon = wm_horizon, to_named_tuple = fields, device = device)
                loss = world_model.rollout(b.state[:, 0], b.action, b.next_state)[0] / wm_horizon
                step_opt(wm_opt, loss, world_model.parameters())
            return float(loss.item())

        def eval_ema(iteration = None, wm_loss = 0., render = False) -> dict:
            step_fn, _ = ema_actor.ema_model.rollout_policy(dataset['eval_refs'].shape[0], device)
            with temp_eval(ema_actor.ema_model):
                m = evaluate_tracker(dataset, step_fn, raw_reward_fn, device, render = render)
            if exists(iteration):
                history.append((total_env_steps, m['norm_return'], m['env_return']))
                print(
                    f'[SuperMimic] iter {iteration:3d} | steps {total_env_steps:6d} | wm {wm_loss:.5f} | '
                    f'track {m["norm_return"]:.3f} | rmse {m["pose_rmse"]:.3f} | '
                    f'env_ret {m["env_return"]:7.1f} (expert {dataset["expert_return"]:.1f}) | {time.time() - t0:.1f}s',
                    flush = True
                )
            return m

        collect_steps(160, random = True)
        eval_ema(0, fit_world_model(300))

        ckpt_path = out_path / 'supermimic_tracker_actor.pt'
        best_norm = -float('inf')
        seed_fields = ('state', 'action', 'clip_id', 'frame_idx')

        for iteration in range(1, num_iterations + 1):
            collect_steps(16)
            wm_loss = fit_world_model(10)

            actor.train()
            for _ in range(policy_updates):
                seed_b = buffer.sample(batch_size, horizon = 2, to_named_tuple = seed_fields, device = device)
                clip_ids = seed_b.clip_id[:, -1].long()
                frame_ids = seed_b.frame_idx[:, -1].long()
                goals = gather_refs(vec.refs_t, clip_ids, frame_ids + 1, imag_horizon + GOAL_WINDOW)
                critic_goals = goals[:, :imag_horizon + 1]

                rollout = supermimic(
                    seed_b.state[:, -1],
                    actor_goals = goals,
                    references = goals[:, :imag_horizon],
                    past_obs = seed_b.state[:, :-1],
                    past_actions = seed_b.action[:, :-1],
                    num_rollout_timesteps = imag_horizon
                )

                actor_loss = supermimic.actor_loss(
                    rollout.rewards,
                    final_state = rollout.states[:, -1],
                    final_goal = critic_goals[:, -1]
                )
                step_opt(actor_opt, actor_loss, actor.parameters(), clip_val = 0.1)
                actor_sched.step()
                ema_actor.update()

                for _ in range(4):
                    c_loss = supermimic.critic_loss(rollout.states, rollout.rewards, goals = critic_goals, lam = 0.95)
                    step_opt(critic_opt, c_loss, critic.parameters())
                    supermimic.update_ema()

            if divisible_by(iteration, 2) or iteration == num_iterations:
                m = eval_ema(iteration, wm_loss)
                if m['norm_return'] > best_norm:
                    best_norm = m['norm_return']
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    num_envs = 16
    rollout_steps = 128

    with MocapEnvs(dataset, 'train', device, num_envs = num_envs, horizon = 128) as vec:
        agent = PPOAgent(vec.dim_state + GOAL_WINDOW * REF_DIM, vec.dim_action, vec.bounds).to(device)
        opt = AdamW(agent.parameters(), lr = 3e-4, eps = 1e-5)
        reward_fn = make_reward().to(device)
        prev_act = np.zeros((num_envs, vec.dim_action), np.float32)
        steps_done = 0
        history = []

        encode = lambda s, gw: safe_cat((s, rearrange(gw, 'b w d -> b (w d)')), dim = -1)
        to_seq = lambda a: rearrange(as_tensor(a, device = device), 'b ... -> b 1 ...')

        def step_env(actions: np.ndarray):
            nonlocal steps_done
            clips, frames = vec.clip.copy(), vec.frame.copy()
            has_prev = (vec.ep_step > 0)
            next_s, _, dones = vec.step(actions)

            with torch.no_grad():
                step_r = reward_fn(
                    to_seq(next_s),
                    to_seq(dataset['train_refs'][clips, frames + 1]),
                    actions = to_seq(actions),
                    past_actions = to_seq(prev_act),
                    past_mask = to_seq(has_prev)
                )
                step_r = rearrange(step_r, 'b 1 -> b').cpu().numpy()

            prev_act[:] = einx.where('b, , b d -> b d', dones, np.float32(0.), actions)
            steps_done += num_envs
            return step_r, dones.astype(np.float32)

        def log_eval(label: str):
            with temp_eval(agent):
                m = evaluate_tracker(dataset, lambda s, gw: agent.dist(encode(s, gw)).mean, reward_fn, device)
            history.append((steps_done, m['norm_return'], m['env_return']))
            if label:
                print(
                    f'[x-ppo tracker] {label} | steps {steps_done:6d} | track {m["norm_return"]:.3f} | '
                    f'env_ret {m["env_return"]:7.1f} | {time.time() - t0:.1f}s',
                    flush = True
                )

        num_updates = max(1, total_steps // (num_envs * rollout_steps))
        log_eval('')

        for update in range(1, num_updates + 1):
            get_obs = lambda: encode(as_tensor(vec.norm(), device = device), vec.goals())
            ppo_step(agent, opt, get_obs, step_env, rollout_steps, device)
            if divisible_by(update, max(1, num_updates // 8)) or update == num_updates:
                log_eval(f'update {update:3d}/{num_updates}')

    return history

# self-explanatory plots & labeled gifs

def add_banner(frames: list[np.ndarray], title: str, subtitle: str, accent: tuple[int, int, int]) -> list[np.ndarray]:
    bar = Image.new('RGB', (frames[0].shape[1], 50), (15, 23, 42))
    draw = ImageDraw.Draw(bar)
    draw.text((12, 6), title, fill = (248, 250, 252), font = ImageFont.load_default(size = 15))
    draw.text((12, 27), subtitle, fill = (203, 213, 225), font = ImageFont.load_default(size = 12))
    draw.rectangle((0, 47, bar.width, 50), fill = accent)
    bar_arr = np.asarray(bar)
    return [np.concatenate((bar_arr, f), axis = 0) for f in frames]

def save_comparison_gifs(dataset: dict, sm_final: dict, out_path: Path, fps = 25):
    env = gym.make(dataset['env_name'], render_mode = 'rgb_array', camera_name = 'track')
    env.reset()
    expert_raw = []
    for qp, qv in zip(dataset['eval_qpos'][0, ::4], dataset['eval_qvel'][0, ::4]):
        env.unwrapped.set_state(qp, qv)
        expert_raw.append(env.render())
    env.close()

    expert_frames = add_banner(
        expert_raw,
        'x-ppo Expert (Task Reward)',
        f'HalfCheetah-v5 held-out clip  |  Return: {dataset["expert_return"]:.0f}',
        accent = (22, 163, 74)
    )
    sm_sub = (
        f'Track: {sm_final["norm_return"]:.3f}  |  RMSE: {sm_final["pose_rmse"]:.3f}  |  '
        f'Unseen Return: {sm_final["env_return"]:.0f}'
    )
    mimic_frames = add_banner(
        sm_final['frames'],
        f'SuperMimic Tracker ({REF_DIM}-d Pose Only, No Task Reward)',
        sm_sub,
        accent = (37, 99, 235)
    )

    divider = np.full((expert_frames[0].shape[0], 4, 3), (15, 23, 42), np.uint8)
    side_by_side = [np.concatenate((l, divider, r), axis = 1) for l, r in zip(expert_frames, mimic_frames)]

    for name, frames in (
        ('xppo_expert_cheetah.gif', expert_frames),
        ('supermimic_tracker_cheetah.gif', mimic_frames),
        ('expert_vs_supermimic_side_by_side.gif', side_by_side)
    ):
        imageio.mimsave(out_path / name, frames, fps = fps, loop = 0)

def annotate_end(ax, x: float, y: float, text: str, color: str, offset: tuple[int, int], ha = 'left'):
    ax.annotate(
        text, xy = (x, y), xytext = offset, textcoords = 'offset points',
        ha = ha, color = color, fontsize = 8.5, fontweight = 'bold'
    )

def save_convergence_plot(sm_hist: list, ppo_hist: list | None, expert_return: float, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12.2, 4.9), dpi = 150)
    suffix = 'SuperMimic BPTT vs x-ppo Tracker' if exists(ppo_hist) else 'SuperMimic World-Model BPTT'
    fig.suptitle(
        f'HalfCheetah-v5: x-ppo Expert → {REF_DIM}-d Pose-Only Reference Subset → {suffix}',
        fontsize = 11.5, fontweight = 'bold'
    )

    sm_steps, sm_track, sm_ret = zip(*sm_hist)
    sm_k = [s / 1e3 for s in sm_steps]
    ppo_steps, ppo_track, ppo_ret = zip(*ppo_hist) if exists(ppo_hist) else (None, None, None)
    ppo_k = [s / 1e3 for s in ppo_steps] if exists(ppo_hist) else None
    max_k = max(sm_k[-1], ppo_k[-1] if exists(ppo_k) else 0.)

    panels = (
        (ax1, sm_track, ppo_track, '{:.3f}', 'Normalized Tracking Return (0–1)', 'Pose-Subset Tracking (Held-Out)', ''),
        (ax2, sm_ret, ppo_ret, '{:.0f}', 'HalfCheetah-v5 Return', 'Recovered Task Return (Unseen)', ' — no task reward')
    )

    for ax, sm_y, ppo_y, fmt, ylabel, title, note in panels:
        sm_lbl = f'SuperMimic BPTT{note} ({fmt.format(sm_y[-1])} @ {sm_k[-1]:.1f}k)'
        ax.plot(sm_k, sm_y, color = '#1d4ed8', marker = 'o', markersize = 4, linewidth = 2.2, label = sm_lbl)
        annotate_end(ax, sm_k[-1], sm_y[-1], fmt.format(sm_y[-1]), '#1d4ed8', (6, -10 if ax is ax1 else -4))

        if exists(ppo_y):
            ppo_lbl = f'x-ppo Tracker{note} ({fmt.format(ppo_y[-1])} @ {ppo_k[-1]:.1f}k)'
            ax.plot(
                ppo_k, ppo_y,
                color = '#dc2626', marker = 's', markersize = 4, linewidth = 2., linestyle = '--', label = ppo_lbl
            )
            annotate_end(ax, ppo_k[-1], ppo_y[-1], fmt.format(ppo_y[-1]), '#dc2626', (-4, 8), ha = 'right')

        if ax is ax1:
            ax.set_ylim(0., 1.08)
        else:
            exp_lbl = f'x-ppo Expert Reference ({expert_return:.0f})'
            ax.axhline(expert_return, color = '#16a34a', linestyle = ':', linewidth = 1.9, label = exp_lbl)

        ax.set_xlim(0., max_k * 1.12)
        ax.set_xlabel('Environment Steps (×1,000)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha = 0.25)
        ax.legend(loc = 'center right' if (ax is ax2 and exists(ppo_hist)) else 'lower right', fontsize = 8.2)

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
            env_name, mocap_path, device,
            seed = seed, total_steps = expert_steps, target_return = expert_target_return
        )

    sm_result = train_supermimic_tracker(dataset, out_path, device, seed = seed, num_iterations = num_iterations)

    ppo_hist = None
    if compare_ppo:
        ppo_steps = max(sm_result['history'][-1][0], ppo_tracker_steps)
        ppo_hist = train_ppo_tracker_baseline(dataset, device, total_steps = ppo_steps, seed = seed)

    plot_path = out_path / 'supermimic_vs_xppo_convergence.png'
    save_convergence_plot(sm_result['history'], ppo_hist, dataset['expert_return'], plot_path)
    save_comparison_gifs(dataset, sm_result['final'], out_path)
    print(f'\nsaved plot & labeled gifs to {out_path}', flush = True)

if __name__ == '__main__':
    fire.Fire(main)
