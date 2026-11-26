# /// script
# requires-python = ">= 3.10"
# dependencies = [
#     "supermimic-pytorch",
#     "env-ssl-wrapper>=0.7.2",
#     "x-ppo",
#     "mean-conc-beta>=0.2.11",
#     "memmap-replay-buffer>=0.3.0",
#     "mujoco>=3.2",
#     "gymnasium>=1.0",
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

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import copy
import json
import math
import time
from pathlib import Path

import fire
import gymnasium as gym
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import Tensor, as_tensor
from torch.nn import Module, SiLU, Tanh, init
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from einops import rearrange
from ema_pytorch import EMA
from mean_conc_beta import Beta
from memmap_replay_buffer import StreamReplayBuffer
from torch_einops_utils import temp_eval
from x_mlps_pytorch import MLP
from x_ppo import calc_gae, calc_smdp_gae, clipped_value_loss_fn, ppo_actor_loss

from env_ssl_wrapper import LatentActionWrapper, StandardizeEnvWrapper
from supermimic_pytorch import Critic, HistoryWindow, LatentMotionActor, SuperMimic, WorldModel
from train_ruka_hand import (
    FPS, LOCKED_JOINTS, SUBSTEPS, TIP_BODIES,
    build_ruka_model, cosine_anneal, default, exists,
    make_reward, norm_stats, retarget_human_hand, select_device, step_optimizer,
)

# constants

SKILL_SPECS = (
    # name, short, start, end, hex, rgba
    ('Open Palm',    'Open',   286, 299, '#0ea5e9', (0.12, 0.65, 0.95, 0.55)),
    ('Power Fist',   'Fist',   254, 274, '#f59e0b', (0.96, 0.66, 0.08, 0.90)),
    ('Index Pinch',  'Index',  308, 328, '#ef4444', (0.94, 0.27, 0.27, 0.95)),
    ('Middle Pinch', 'Middle', 356, 376, '#10b981', (0.06, 0.73, 0.51, 0.95)),
    ('Ring Pinch',   'Ring',   406, 426, '#8b5cf6', (0.55, 0.36, 0.96, 0.95)),
    ('Pinky Pinch',  'Pinky',  452, 472, '#ec4899', (0.93, 0.28, 0.60, 0.95)),
)

NUM_SKILLS = len(SKILL_SPECS)
CLIP_LEN = 20
ENC_SUBSAMPLE = (0, 5, 10, 15, 19)
DEMO_SEQ = (2, 0, 1, 3, 4, 5)
SCRIPT_DIR = Path(__file__).resolve().parent

# helpers

def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def lookup_grid_skills(z_arr: np.ndarray, grid: dict) -> np.ndarray:
    xs, ys, table = grid['xs'], grid['ys'], grid['nearest_skill']
    ix = np.clip(np.round((z_arr[:, 0] - xs[0]) / (xs[-1] - xs[0]) * (len(xs) - 1)).astype(int), 0, len(xs) - 1)
    iy = np.clip(np.round((z_arr[:, 1] - ys[0]) / (ys[-1] - ys[0]) * (len(ys) - 1)).astype(int), 0, len(ys) - 1)
    return table[iy, ix]

def extract_skill_clips(mocap: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    qpos_all, qvel_all, tips_all = mocap['qpos'], mocap['qvel'], mocap['tips']
    nq = qpos_all.shape[-1]

    clip_qpos = np.zeros((NUM_SKILLS, CLIP_LEN, nq), dtype = np.float32)
    clip_qvel = np.zeros((NUM_SKILLS, CLIP_LEN, nq), dtype = np.float32)
    clip_tips = np.zeros((NUM_SKILLS, CLIP_LEN, 5, 3), dtype = np.float32)

    for m, (_, _, start, end, _, _) in enumerate(SKILL_SPECS):
        raw_q, raw_v, raw_t = qpos_all[start:end], qvel_all[start:end], tips_all[start:end]
        pad = max(0, CLIP_LEN - len(raw_q))
        clip_qpos[m] = np.concatenate((raw_q[:CLIP_LEN], np.repeat(raw_q[-1:], pad, axis = 0))) if pad else raw_q[:CLIP_LEN]
        clip_qvel[m] = np.concatenate((raw_v[:CLIP_LEN], np.zeros((pad, nq), dtype = np.float32))) if pad else raw_v[:CLIP_LEN]
        clip_tips[m] = np.concatenate((raw_t[:CLIP_LEN], np.repeat(raw_t[-1:], pad, axis = 0))) if pad else raw_t[:CLIP_LEN]

    peak_qpos = clip_qpos[:, 8:18].mean(axis = 1).astype(np.float32)
    peak_tips = clip_tips[:, 8:18].mean(axis = 1).astype(np.float32)
    mean, std = norm_stats(clip_qpos.reshape(-1, nq), clip_qvel.reshape(-1, nq))

    tip_mean = clip_tips.mean(axis = (0, 1, 2), keepdims = True)
    enc_q = (clip_qpos[:, ENC_SUBSAMPLE] - mean[:nq]) / std[:nq]
    enc_t = ((clip_tips[:, ENC_SUBSAMPLE] - tip_mean) / 0.05).reshape(NUM_SKILLS, len(ENC_SUBSAMPLE), -1)
    motion_feats = np.concatenate((enc_q, enc_t), axis = -1).reshape(NUM_SKILLS, -1).astype(np.float32)

    return dict(
        qpos = clip_qpos,
        qvel = clip_qvel,
        tips = clip_tips,
        peak_qpos = peak_qpos,
        peak_tips = peak_tips,
        mean = mean,
        std = std,
        motion_feats = motion_feats,
        palm_ball_pos = peak_tips[1].mean(axis = 0).astype(np.float32),
        bead_positions = np.stack([(0.30 * peak_tips[2 + k, 0] + 0.70 * peak_tips[2 + k, 1 + k]).astype(np.float32) for k in range(4)]),
        release_pos = np.array([0.025, 0.010, 0.228], dtype = np.float32),
    )

# stage 1: supermimic world model + latent-conditioned motor policy

class RukaSkillCollectEnvs:
    def __init__(self, model: mujoco.MjModel, skills: dict, num_envs = 16, horizon = 96, switch_every = 24, rsi_pos_noise = 0.04):
        self.model, self.nq = model, model.nq
        self.clip_qpos, self.clip_qvel = skills['qpos'], skills['qvel']
        self.mean, self.std = skills['mean'], skills['std']
        self.num_envs, self.horizon, self.switch_every, self.rsi_pos_noise = num_envs, horizon, switch_every, rsi_pos_noise

        self.datas = [mujoco.MjData(model) for _ in range(num_envs)]
        self.skill_ids = np.zeros(num_envs, dtype = np.int64)
        self.steps = np.zeros(num_envs, dtype = np.int64)
        self.states = np.zeros((num_envs, 2 * self.nq), dtype = np.float32)
        for i in range(num_envs):
            self.reset(i)

    def reset(self, index: int):
        s, f, cmd = int(np.random.randint(NUM_SKILLS)), int(np.random.randint(CLIP_LEN)), int(np.random.randint(NUM_SKILLS))
        data = self.datas[index]
        mujoco.mj_resetData(self.model, data)
        noise = np.random.normal(0., self.rsi_pos_noise, self.nq)
        data.qpos[:] = np.clip(self.clip_qpos[s, f] + noise, self.model.jnt_range[:, 0], self.model.jnt_range[:, 1])
        data.qvel[:] = self.clip_qvel[s, f] * np.random.uniform(0.3, 1.0)
        mujoco.mj_forward(self.model, data)

        self.skill_ids[index] = cmd
        self.steps[index] = 0
        self.states[index] = np.concatenate((data.qpos, data.qvel))

    def normalize(self, states: np.ndarray | None = None) -> np.ndarray:
        return (default(states, self.states) - self.mean) / self.std

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        next_states = np.empty_like(self.states)
        dones = np.zeros(self.num_envs, dtype = bool)

        for i, data in enumerate(self.datas):
            data.ctrl[:] = np.clip(actions[i], -1., 1.)
            mujoco.mj_step(self.model, data, nstep = SUBSTEPS)
            next_states[i] = np.concatenate((data.qpos, data.qvel))
            self.steps[i] += 1

            if self.steps[i] >= self.horizon:
                dones[i] = True
                self.reset(i)
            else:
                self.states[i] = next_states[i]
                if self.steps[i] % self.switch_every == 0:
                    self.skill_ids[i] = int(np.random.choice([s for s in range(NUM_SKILLS) if s != self.skill_ids[i]]))

        return next_states, dones

@torch.no_grad()
def _rollout_stage1(model, data, tip_ids, init_qpos, z_cmd, actor, mean, std, obs_hist, act_hist, device, steps, tail = 1):
    mujoco.mj_resetData(model, data)
    data.qpos[:], data.qvel[:] = init_qpos, 0.
    mujoco.mj_forward(model, data)

    obs_cache, act_cache = None, act_hist.init_cache(1, device = device)
    qs, tips = [], []
    for t in range(steps):
        st = np.concatenate((data.qpos, data.qvel)).astype(np.float32)
        obs_w, obs_cache = obs_hist.step(as_tensor((st - mean) / std, device = device)[None], obs_cache)
        act = actor.act(obs_w, z_cmd, past_action = act_hist.read(act_cache))
        act_cache = act_hist.push(act, act_cache)
        data.ctrl[:] = np.clip(act[0].cpu().numpy(), -1., 1.)
        mujoco.mj_step(model, data, nstep = SUBSTEPS)
        if t >= steps - tail:
            qs.append(data.qpos.copy())
            tips.append(np.stack([data.xpos[bid].copy() for bid in tip_ids]))
    return np.mean(qs, axis = 0), np.mean(tips, axis = 0)

@torch.no_grad()
def eval_stage1(model: mujoco.MjModel, skills: dict, actor: LatentMotionActor, device: torch.device, *, with_grid = False, obs_window = 2, action_window = 1) -> dict:
    nq, mean, std, peak_qpos, peak_tips = model.nq, skills['mean'], skills['std'], skills['peak_qpos'], skills['peak_tips']
    z_all = actor.encode(as_tensor(skills['motion_feats'], device = device), add_noise = False)
    tip_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in TIP_BODIES]
    data = mujoco.MjData(model)
    obs_hist = HistoryWindow(obs_window, 2 * nq, pad_mode = 'replicate')
    act_hist = HistoryWindow(action_window, nq, lag = 1, pad_mode = 'zero')

    tip_err = np.zeros((NUM_SKILLS, NUM_SKILLS), dtype = np.float32)
    q_rmse = np.zeros((NUM_SKILLS, NUM_SKILLS), dtype = np.float32)
    match = np.zeros((NUM_SKILLS, NUM_SKILLS), dtype = np.float32)

    for src in range(NUM_SKILLS):
        for dst in range(NUM_SKILLS):
            mq, mt = _rollout_stage1(model, data, tip_ids, peak_qpos[src], z_all[dst : dst + 1], actor, mean, std, obs_hist, act_hist, device, steps = 24, tail = 10)
            tip_err[src, dst] = float(np.linalg.norm(mt - peak_tips[dst], axis = -1).mean() * 1000.)
            q_rmse[src, dst] = float(np.sqrt(np.mean((mq - peak_qpos[dst]) ** 2)))
            match[src, dst] = float(int(np.argmin(np.linalg.norm(peak_tips - mt[None], axis = -1).mean(axis = -1))) == dst)

    z_np = z_all.cpu().numpy()
    pdist = np.linalg.norm(z_np[:, None] - z_np[None, :], axis = -1)
    np.fill_diagonal(pdist, np.inf)

    res = dict(
        tip_mm_mean = float(tip_err.mean()),
        tip_mm_diag = float(np.diag(tip_err).mean()),
        q_rmse_mean = float(q_rmse.mean()),
        transition_acc = float(match.mean()),
        min_latent_dist = float(pdist.min()),
        tip_err_mat = tip_err,
        latents = z_np,
    )

    if with_grid:
        xs = ys = np.linspace(-1., 1., 21)
        nearest = np.zeros((21, 21), dtype = np.int64)
        neutral_q = peak_qpos.mean(axis = 0)
        for iy, zy in enumerate(ys):
            for ix, zx in enumerate(xs):
                _, tips_end = _rollout_stage1(model, data, tip_ids, neutral_q, torch.tensor([[zx, zy]], dtype = torch.float32, device = device), actor, mean, std, obs_hist, act_hist, device, steps = 22)
                nearest[iy, ix] = int(np.argmin(np.linalg.norm(peak_tips - tips_end[None], axis = -1).mean(axis = -1)))
        res['grid'] = dict(xs = xs, ys = ys, nearest_skill = nearest)

    return res

def train_stage1(model: mujoco.MjModel, skills: dict, out_path: Path, device: torch.device, *, seed = 42, dim_latent = 2, num_iterations = 95, obs_window = 2, action_window = 1):
    seed_everything(seed)
    nq, dim_state, dim_motion = model.nq, 2 * model.nq, skills['motion_feats'].shape[-1]
    batch_size, wm_horizon, imag_horizon = 256, 8, 16
    warmup_steps, warmup_wm_updates, collect_steps = 128, 350, 16
    wm_updates, policy_updates, critic_updates = 20, 8, 4

    envs = RukaSkillCollectEnvs(model, skills)
    motion_feats_t = as_tensor(skills['motion_feats'], device = device)
    ref_qpos_t = as_tensor(skills['qpos'][:, 4 : 4 + imag_horizon], device = device)
    ref_tips_t = as_tensor(skills['tips'][:, 4 : 4 + imag_horizon], device = device)

    actor_kwargs = dict(
        dim_obs = dim_state, dim_action = nq, dim_motion = dim_motion, dim_latent = dim_latent,
        num_obs_frames = obs_window, num_action_frames = action_window, dim_explorative_latent = 4,
        latent_space = 'box', encoder_hiddens = (128, 128), policy_hiddens = (256, 256, 256),
        normed_mlp = True, squash_action = True,
    )
    actor = LatentMotionActor(**actor_kwargs, latent_noise_scale = 0.14, activation = SiLU()).to(device)
    ema_actor = EMA(actor, beta = 0.96, include_online_model = False, update_after_step = 0, update_every = 1).to(device)
    world_model = WorldModel(dim_state = dim_state, dim_action = nq, dim_sim_context = 0, dim_hiddens = (512, 512, 512), activation = SiLU(), normed_mlp = True, pred_terminal = False).to(device)
    critic = Critic(dim_state = dim_state, dim_goal = dim_latent, dim_sim_context = 0, dim_hiddens = (256, 256), activation = SiLU(), normed_mlp = True).to(device)

    reward_fn = make_reward(
        model, as_tensor(skills['mean'], device = device), as_tensor(skills['std'], device = device),
        joint_std = 0.25, joint_weight = 2.0, tip_std = 0.018, tip_weight = 3.0,
        joint_limit_margin = 0.02, joint_limit_weight = -0.5, action_smoothness_weight = -0.01,
    ).to(device)

    obs_history = HistoryWindow(obs_window, dim_state, pad_mode = 'replicate')
    action_history = HistoryWindow(action_window, nq, lag = 1, pad_mode = 'zero')
    supermimic = SuperMimic(
        actor = actor, world_model = world_model, critic = critic,
        use_ema_critic = True, critic_ema_decay = 0.99,
        obs_history = obs_history, action_history = action_history, reward_fn = reward_fn,
        discount_factor = 0.99, num_explorative_rollouts = 2, dim_explorative_latent = 4, explorative_latent_scale = 0.20,
    ).to(device)

    wm_opt = AdamW(world_model.parameters(), lr = 6e-4, weight_decay = 1e-5)
    wm_sched = CosineAnnealingLR(wm_opt, T_max = max(1, warmup_wm_updates + num_iterations * wm_updates), eta_min = 1e-4)
    actor_opt = AdamW(actor.parameters(), lr = 5e-4, betas = (0.9, 0.95), weight_decay = 1e-5)
    actor_sched = CosineAnnealingLR(actor_opt, T_max = max(1, num_iterations * policy_updates), eta_min = 5e-5)
    critic_opt = AdamW(critic.parameters(), lr = 3e-4, weight_decay = 1e-5)

    buffer = StreamReplayBuffer(num_streams = envs.num_envs, capacity = 64_000, fields = dict(state = ('float', dim_state), action = ('float', nq), next_state = ('float', dim_state)))
    obs_cache, act_cache = None, action_history.init_cache(envs.num_envs, device = device)
    smooth_rand = np.zeros((envs.num_envs, nq), dtype = np.float32)
    total_steps, history, best_score, t0 = 0, [], float('inf'), time.time()
    ckpt_path = out_path / 'stage1_latent_actor.pt'

    def collect(steps: int, action_noise = 0., latent_noise = 0.08, random = False):
        nonlocal obs_cache, act_cache, smooth_rand, total_steps
        with temp_eval(actor), torch.no_grad():
            z_bank = actor.encode(motion_feats_t, add_noise = False)
            for _ in range(steps):
                norm_st = envs.normalize()
                obs_w, obs_cache = obs_history.step(as_tensor(norm_st, device = device), obs_cache)
                past_a = action_history.read(act_cache)
                if random:
                    smooth_rand = 0.65 * smooth_rand + 0.35 * np.random.uniform(-1., 1., size = (envs.num_envs, nq)).astype(np.float32)
                    act_np = np.clip(smooth_rand * 1.4, -1., 1.)
                    act_t = as_tensor(act_np, device = device)
                else:
                    z_cmd = z_bank[as_tensor(envs.skill_ids, device = device)]
                    if latent_noise > 0.:
                        z_cmd = actor.project_latent(z_cmd + torch.randn_like(z_cmd) * latent_noise)
                    act_t = actor.act(obs_w, z_cmd, past_action = past_a)
                    if action_noise > 0.:
                        act_t = (act_t + torch.randn_like(act_t) * action_noise).clamp(-1., 1.)
                    act_np = act_t.cpu().numpy()

                act_cache = action_history.push(act_t, act_cache)
                next_states, dones = envs.step(act_np)
                norm_next = envs.normalize(next_states)
                for i in range(envs.num_envs):
                    buffer.store_step(i, state = norm_st[i], action = act_np[i], next_state = norm_next[i])
                    if dones[i]:
                        buffer.end_episode(i)
                total_steps += envs.num_envs
                if dones.any():
                    done_t = as_tensor(dones, device = device)
                    obs_cache = obs_history.reset(obs_cache, done_t)
                    act_cache = action_history.reset(act_cache, done_t)
                    smooth_rand[dones] = 0.

    def fit_wm(updates: int) -> float:
        world_model.train()
        for _ in range(updates):
            b = buffer.sample(batch_size, horizon = wm_horizon, to_named_tuple = ('state', 'action', 'next_state'), device = device)
            loss = world_model.rollout(b.state[:, 0], b.action, b.next_state)[0] / wm_horizon
            step_optimizer(wm_opt, loss, world_model.parameters(), max_norm = 1.0)
            wm_sched.step()
        return float(loss.item())

    print(f'[stage 1] supermimic latent actor ({NUM_SKILLS} skills -> {dim_latent}d) on {device}', flush = True)
    collect(warmup_steps, random = True)
    wm_loss = fit_wm(warmup_wm_updates)

    for it in range(1, num_iterations + 1):
        frac = (it - 1) / max(1, num_iterations - 1)
        cur_lat_noise = cosine_anneal(0.14, 0.10, frac)
        collect(collect_steps, action_noise = 0.08 - 0.06 * frac, latent_noise = cur_lat_noise * 0.75)
        wm_loss = fit_wm(wm_updates)

        actor.train()
        for _ in range(policy_updates):
            seed_b = buffer.sample(batch_size, horizon = 2, to_named_tuple = ('state', 'action'), device = device)
            targets = torch.randint(0, NUM_SKILLS, (batch_size,), device = device)
            z_all = actor.encode(motion_feats_t, add_noise = False)
            z_noisy = actor.project_latent(z_all[targets] + torch.randn((batch_size, dim_latent), device = device) * cur_lat_noise)

            rollout = supermimic(
                seed_b.state[:, -1],
                actor_cond = z_noisy,
                references = dict(joint_pos = ref_qpos_t[targets], body_pos = ref_tips_t[targets]),
                past_obs = seed_b.state[:, :-1],
                past_actions = seed_b.action[:, :-1],
                num_rollout_timesteps = imag_horizon,
                explorative_latent_scale = cosine_anneal(0.20, 0.03, frac),
            )
            z_c = z_noisy.detach()
            loss = (
                supermimic.actor_loss(rollout.rewards, final_state = rollout.states[:, -1], final_goal = z_c)
                + 8.0 * actor.uniformity_loss(z_all, is_encoded = True, temperature = 2.5, max_bound = 0.78, bound_weight = 15.0)
            )
            step_optimizer(actor_opt, loss, actor.parameters(), max_norm = 1.0, clip_value = 0.1)
            actor_sched.step()
            ema_actor.update()

            for _ in range(critic_updates):
                c_loss = supermimic.critic_loss(rollout.states, rollout.rewards, goals = z_c, lam = 0.95)
                step_optimizer(critic_opt, c_loss, critic.parameters(), max_norm = 1.0)
                supermimic.update_ema()

        if it % 5 == 0 or it == num_iterations:
            with temp_eval(ema_actor.ema_model):
                ev = eval_stage1(model, skills, ema_actor.ema_model, device, obs_window = obs_window, action_window = action_window)
            history.append(dict(iter = it, steps = total_steps, wm_loss = wm_loss, tip_mm_mean = ev['tip_mm_mean'], transition_acc = ev['transition_acc'], min_latent_dist = ev['min_latent_dist']))
            print(f'[stage 1] {it:2d}/{num_iterations} | wm {wm_loss:.5f} | 36-trans {ev["transition_acc"]*100:5.1f}% | tips {ev["tip_mm_mean"]:4.1f}mm | min|dz| {ev["min_latent_dist"]:.2f} | {time.time() - t0:.1f}s', flush = True)
            score = ev['tip_mm_mean'] + (1.0 - ev['transition_acc']) * 100.0
            if score < best_score:
                best_score = score
                torch.save(dict(actor_state_dict = ema_actor.ema_model.state_dict(), actor_kwargs = actor_kwargs, skills = skills, history = history), ckpt_path)

    ckpt = torch.load(ckpt_path, map_location = device, weights_only = False)
    best_actor = LatentMotionActor(**ckpt['actor_kwargs'], activation = SiLU()).to(device)
    best_actor.load_state_dict(ckpt['actor_state_dict'])
    best_actor.eval()
    final_ev = eval_stage1(model, skills, best_actor, device, with_grid = True, obs_window = obs_window, action_window = action_window)
    grid = final_ev.pop('grid')
    return best_actor, dict(history = history, eval = final_ev, grid = grid)

# stage 2: pure 3d-geometry multi-stage ruka-v2 task (no skill ids in obs or reward)

def build_ruka_task_model(cache_dir: Path) -> tuple[mujoco.MjModel, int]:
    base_model, _, _ = build_ruka_model(cache_dir)
    spec = mujoco.MjSpec.from_file(str(cache_dir / 'ruka_v2' / 'robot_mj.urdf'))

    for j in list(spec.joints):
        if j.name in LOCKED_JOINTS:
            spec.delete(j)
    for g in list(spec.geoms):
        if g.group == 0:
            spec.delete(g)
        else:
            g.contype = g.conaffinity = g.group = 0

    spec.option.timestep = 1.0 / (FPS * SUBSTEPS)
    spec.option.gravity = [0.0, 0.0, -9.81]
    for j in spec.joints:
        spec.add_actuator(name = f'act_{j.name}', target = j.name, trntype = mujoco.mjtTrn.mjTRN_JOINT, gear = [0.32, 0, 0, 0, 0, 0], ctrllimited = True, ctrlrange = [-1.0, 1.0])

    # non-colliding visual target sphere
    target = spec.worldbody.add_body(name = 'target_object', mocap = True, pos = [0.03, -0.02, 0.16])
    target.add_geom(name = 'target_geom', type = mujoco.mjtGeom.mjGEOM_SPHERE, size = [0.011] * 3, rgba = [0.95, 0.30, 0.30, 0.90], contype = 0, conaffinity = 0)
    spec.worldbody.add_light(pos = [0.2, -0.6, 0.5], dir = [-0.2, 0.8, -0.5], diffuse = [0.85, 0.85, 0.85], specular = [0.25, 0.25, 0.25], castshadow = False)
    spec.worldbody.add_light(pos = [-0.3, -0.2, 0.6], dir = [0.4, 0.3, -0.7], diffuse = [0.4, 0.42, 0.46], castshadow = False)
    spec.add_texture(name = 'sky', type = mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1 = [0.93, 0.95, 0.98], rgb2 = [0.80, 0.84, 0.90], width = 64, height = 64)

    model = spec.compile()
    model.dof_damping[:], model.dof_armature[:] = base_model.dof_damping, base_model.dof_armature
    return model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'target_geom')

class RukaDexterousTaskEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': FPS}

    def __init__(self, model: mujoco.MjModel, target_geom_id: int, skills: dict, *, num_stages = 4, max_steps = 128, hold_steps = 4, jitter_m = 0.0025, render_mode: str | None = None, render_size = 360, forced_sequence: tuple[int, ...] | None = None):
        super().__init__()
        self.model, self.target_geom_id, self.nq, self.skills = model, target_geom_id, model.nq, skills
        self.num_stages, self.max_steps, self.hold_steps, self.jitter_m = int(num_stages), int(max_steps), int(hold_steps), float(jitter_m)
        self.render_mode, self.render_size, self.forced_sequence = render_mode, int(render_size), forced_sequence

        self.tip_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in TIP_BODIES]
        self.data = mujoco.MjData(model)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape = (2 * self.nq + 21,), dtype = np.float32)
        self.action_space = gym.spaces.Box(-1., 1., shape = (self.nq,), dtype = np.float32)
        self._renderer = self._cam = None

    def _tips(self) -> np.ndarray:
        return np.stack([self.data.xpos[bid] for bid in self.tip_ids]).astype(np.float32)

    def _configure_stage(self, task_id: int, rng: np.random.Generator):
        jitter = rng.uniform(-self.jitter_m, self.jitter_m, size = 3).astype(np.float32) if self.jitter_m > 0. else np.zeros(3, dtype = np.float32)
        if task_id == 0:
            self._pos, self._rad, self._rel, vis = self.skills['release_pos'] + 0.5 * jitter, 0.0, 1.0, 0.016
        elif task_id == 1:
            self._pos, self._rad, self._rel, vis = self.skills['palm_ball_pos'] + jitter, 0.024, 0.0, 0.023
        else:
            self._pos, self._rad, self._rel, vis = self.skills['bead_positions'][task_id - 2] + jitter, 0.011, 0.0, 0.011

        self.data.mocap_pos[0] = self._pos
        if self.render_mode == 'rgb_array':
            self.model.geom_size[self.target_geom_id, :] = vis
            self.model.geom_rgba[self.target_geom_id, :] = SKILL_SPECS[task_id][5]

    def _eval_geometry(self, tips: np.ndarray) -> tuple[float, bool]:
        d_th = np.linalg.norm(tips[1:] - tips[0:1], axis = -1)
        if self._rel > 0.5:
            min_z, min_d = float(tips[1:, 2].min()), float(d_th.min())
            return float(np.exp(-max(0., 0.198 - min_z) / 0.016) * np.exp(-max(0., 0.090 - min_d) / 0.020)), (min_z > 0.192 and min_d > 0.083)

        if self._rad > 0.018:
            d = np.linalg.norm(tips - self._pos[None], axis = -1)
            mean_d, max_d = float(d.mean()), float(d.max())
            return float(np.exp(-max(0., mean_d - 0.028) / 0.025) * np.exp(-max(0., max_d - 0.044) / 0.025)), (mean_d < 0.037 and max_d < 0.055)

        best_f = int(np.argmin(np.linalg.norm(tips[1:] - self._pos[None], axis = -1)))
        d_pt = float(np.linalg.norm(0.30 * tips[0] + 0.70 * tips[1 + best_f] - self._pos))
        gap = float(np.linalg.norm(tips[0] - tips[1 + best_f]))
        other = np.arange(4) != best_f
        oz, od = float(tips[1:][other, 2].min()), float(d_th[other].min())
        score = float(np.exp(-d_pt / 0.016) * np.exp(-max(0., gap - 0.058) / 0.018) * np.exp(-max(0., 0.196 - oz) / 0.016) * np.exp(-max(0., 0.066 - od) / 0.016))
        return score, (d_pt < 0.013 and gap < 0.066 and oz > 0.189 and od > 0.062)

    def _obs(self) -> np.ndarray:
        return np.concatenate((self.data.qpos, self.data.qvel, self._tips().reshape(-1), self._pos, [self._rad * 20.0, self._rel, self._stage_idx / max(1, self.num_stages)])).astype(np.float32)

    def reset(self, *, seed = None, options = None):
        super().reset(seed = seed)
        rng = self.np_random
        self._seq = np.asarray(self.forced_sequence, dtype = np.int64) if exists(self.forced_sequence) else rng.choice(NUM_SKILLS, size = self.num_stages, replace = False).astype(np.int64)
        self.num_stages = len(self._seq)

        init_skill = int(rng.choice([s for s in range(NUM_SKILLS) if s != int(self._seq[0])]))
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.clip(self.skills['peak_qpos'][init_skill] + rng.normal(0., 0.03, self.nq), self.model.jnt_range[:, 0], self.model.jnt_range[:, 1])
        self.data.qvel[:] = 0.
        self._stage_idx = self._hold = self._steps = self._done_stages = 0

        self._configure_stage(int(self._seq[0]), rng)
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), dict(current_task_id = int(self._seq[0]), next_task_id = int(self._seq[0]), stages_completed = 0, task_success = False)

    def step(self, action):
        self.data.ctrl[:] = np.clip(np.asarray(action, dtype = np.float32), -1., 1.)
        mujoco.mj_step(self.model, self.data, nstep = SUBSTEPS)
        self._steps += 1

        score, ok = self._eval_geometry(self._tips())
        reward, cur_id = 0.10 * score, int(self._seq[self._stage_idx])
        self._hold = (self._hold + 1) if ok else 0

        if self.render_mode == 'rgb_array':
            self.model.geom_rgba[self.target_geom_id, :] = (0.15, 0.92, 0.30, 0.98) if ok else SKILL_SPECS[cur_id][5]

        terminated = False
        if self._hold >= self.hold_steps:
            reward += 3.0
            self._done_stages += 1
            self._hold = 0
            if self._done_stages >= self.num_stages:
                reward += 5.0
                terminated = True
            else:
                self._stage_idx += 1
                self._configure_stage(int(self._seq[self._stage_idx]), self.np_random)
                mujoco.mj_forward(self.model, self.data)

        truncated = (not terminated) and (self._steps >= self.max_steps)
        info = dict(
            current_task_id = cur_id,
            next_task_id = int(self._seq[min(self._stage_idx, self.num_stages - 1)]),
            stages_completed = self._done_stages,
            task_success = self._done_stages >= self.num_stages,
        )
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != 'rgb_array':
            return None
        if not exists(self._renderer):
            self._renderer = mujoco.Renderer(self.model, height = self.render_size, width = self.render_size)
            self._cam = mujoco.MjvCamera()
            self._cam.lookat[:], self._cam.distance, self._cam.elevation, self._cam.azimuth = [0.020, -0.008, 0.138], 0.33, -8, 90
        self._renderer.update_scene(self.data, camera = self._cam)
        return self._renderer.render()

    def close(self):
        if exists(self._renderer):
            self._renderer.close()
            self._renderer = None

# stage 2: latent action wrapper (env-ssl-wrapper>=0.7.2) + x-ppo

class PPOAgent(Module):
    def __init__(self, dim_in: int, dim_action: int, dim_hidden = 256):
        super().__init__()
        self.actor = MLP(dim_in, dim_hidden, dim_hidden, dim_action * 2, activation = Tanh())
        self.critic = MLP(dim_in, dim_hidden, dim_hidden, 1, activation = Tanh())
        for net, gain in ((self.actor, 0.01), (self.critic, 1.)):
            init.orthogonal_(net.layers[-1].weight, gain = gain)
            init.zeros_(net.layers[-1].bias)
        bounds = np.stack((-np.ones(dim_action, dtype = np.float32), np.ones(dim_action, dtype = np.float32)), axis = -1)
        self.beta = Beta(bounds = bounds, pos_fn = 'softplus', init_conc = 2.0, detach_entropy_mean = True)

    def dist(self, x: Tensor):
        return self.beta(rearrange(self.actor(x), '... (d p) -> ... d p', p = 2))

    def value(self, x: Tensor) -> Tensor:
        return rearrange(self.critic(x), '... 1 -> ...')

def wrap_task_env(raw_env, *, actor: LatentMotionActor | None = None, skills: dict | None = None, horizon = 8, gamma = 0.99, device = 'cpu', eval_mode = False, record_frames = False, norm_mode = 'deferred', norm_state = None) -> StandardizeEnvWrapper:
    if exists(actor):
        mean_t = as_tensor(skills['mean'], dtype = torch.float32, device = device)
        std_t = as_tensor(skills['std'], dtype = torch.float32, device = device)
        dim_proprio = len(skills['mean'])

        raw_env = LatentActionWrapper(
            raw_env,
            actor,
            horizon = horizon,
            obs_fn = lambda obs: (obs[..., :dim_proprio] - mean_t.to(obs)) / std_t.to(obs),
            obs_history_len = 2,
            action_history_len = 1,
            obs_pad_mode = 'replicate',
            freeze = True,
            gamma = 1.0 if eval_mode else gamma,
            eval_mode = eval_mode,
            clip_lower_actions = True,
            record_frames = record_frames,
        )

    env = StandardizeEnvWrapper(raw_env, device = device, normalize_state = dict(update_mode = norm_mode))
    if exists(norm_state):
        env.load_state_dict(norm_state)
    return env

@torch.no_grad()
def eval_stage2(agent: PPOAgent | None, make_single_env, *, actor: LatentMotionActor | None = None, skills: dict | None = None, norm_state = None, horizon = 8, episodes = 24, seed = 2000, device = torch.device('cpu'), random_latent_dim: int | None = None, stage1_latents: np.ndarray | None = None, stage1_grid: dict | None = None) -> dict:
    env = wrap_task_env(make_single_env(), actor = actor, skills = skills, horizon = horizon, device = device, eval_mode = True, norm_mode = 'frozen', norm_state = norm_state)
    if exists(agent):
        agent.eval()

    returns, stages, successes, emitted_z, emitted_task = [], [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed = seed + ep)
        ret, done, st_done, ok = 0.0, False, 0, False
        while not done:
            cur_task = int(info.get('next_task_id', info.get('current_task_id', -1)))
            action = agent.dist(obs).mean if exists(agent) else torch.empty((1, random_latent_dim), device = device).uniform_(-1., 1.)
            if exists(actor):
                emitted_z.append(action[0].cpu().numpy())
                emitted_task.append(cur_task)
            obs, rew, term, trunc, info = env.step(action)
            ret += float(rew.item())
            done = bool((term | trunc).item())
            st_done = int(np.asarray(info.get('stages_completed', st_done)).item())
            ok = bool(np.asarray(info.get('task_success', ok)).item())
        returns.append(ret)
        stages.append(st_done)
        successes.append(float(ok))
    env.close()

    ctr_acc = basin_acc = None
    if emitted_z and exists(stage1_latents):
        z_arr, t_arr = np.asarray(emitted_z, dtype = np.float32), np.asarray(emitted_task, dtype = np.int64)
        valid = t_arr >= 0
        if valid.any():
            nearest = np.argmin(np.linalg.norm(z_arr[valid, None] - stage1_latents[None], axis = -1), axis = -1)
            ctr_acc = float(np.mean([float((nearest[t_arr[valid] == m] == m).mean()) for m in range(NUM_SKILLS) if (t_arr[valid] == m).any()]))
            if exists(stage1_grid):
                decoded = lookup_grid_skills(z_arr[valid], stage1_grid)
                basin_acc = float(np.mean([float((decoded[t_arr[valid] == m] == m).mean()) for m in range(NUM_SKILLS) if (t_arr[valid] == m).any()]))

    return dict(
        mean_return = float(np.mean(returns)),
        mean_stages = float(np.mean(stages)),
        success_rate = float(np.mean(successes)),
        balanced_retrieval_acc = ctr_acc,
        basin_retrieval_acc = basin_acc,
        emitted_z = np.asarray(emitted_z, dtype = np.float32) if emitted_z else None,
        emitted_task = np.asarray(emitted_task, dtype = np.int64) if emitted_task else None,
    )

def train_stage2_ppo(label: str, make_single_env, *, actor: LatentMotionActor | None = None, skills: dict | None = None, stage1_latents: np.ndarray | None = None, stage1_grid: dict | None = None, num_envs = 16, total_prim_steps = 300_000, horizon = 8, rollout_steps = 24, gamma = 0.99, gae_lam = 0.95, lr = 6e-4, ppo_epochs = 10, minibatch_size = 128, ent_coef = 0.005, eval_every = 10, num_stages = 4, seed = 42, device = torch.device('cpu')):
    seed_everything(seed)
    use_latent = exists(actor)

    env = wrap_task_env([make_single_env() for _ in range(num_envs)], actor = actor, skills = skills, horizon = horizon, gamma = gamma, device = device)
    obs, _ = env.reset(seed = seed)

    agent = PPOAgent(int(obs.shape[-1]), int(env.action_space.shape[0])).to(device)
    num_updates = max(1, int(math.ceil(total_prim_steps / (num_envs * rollout_steps * (horizon if use_latent else 1)))))
    opt = AdamW(agent.parameters(), lr = lr, eps = 1e-5)
    sched = CosineAnnealingLR(opt, T_max = num_updates, eta_min = lr * 0.08)

    history, prim_steps, t0 = [], 0, time.time()

    def log_eval(upd: int):
        ev = eval_stage2(agent, make_single_env, actor = actor, skills = skills, norm_state = env.state_dict(), horizon = horizon, episodes = 24, seed = 2000, device = device, stage1_latents = stage1_latents, stage1_grid = stage1_grid)
        history.append(dict(prim_steps = prim_steps, **{k: v for k, v in ev.items() if k not in ('emitted_z', 'emitted_task')}))
        ret_s = f' | basin {ev["basin_retrieval_acc"]*100:5.1f}%' if exists(ev['basin_retrieval_acc']) else ''
        print(f'[{label}] {upd:3d}/{num_updates:3d} | steps {prim_steps:6d} | ret {ev["mean_return"]:5.2f} | stages {ev["mean_stages"]:.2f}/{num_stages} | succ {ev["success_rate"]*100:5.1f}%{ret_s} | {time.time() - t0:.1f}s', flush = True)

    log_eval(0)

    for upd in range(1, num_updates + 1):
        agent.eval()
        buf = {k: [] for k in ('obs', 'act', 'lp', 'rew', 'val', 'next_val', 'term', 'trunc', 'dur')}

        for _ in range(rollout_steps):
            with torch.no_grad():
                dist = agent.dist(obs)
                action = dist.sample()
                lp = dist.log_prob(action).sum(dim = -1)
                val = agent.value(obs)

            next_obs, rew, term, trunc, info = env.step(action)
            done = term | trunc
            with torch.no_grad():
                boot_obs = torch.where(done.unsqueeze(-1), info['final_observation'], next_obs) if ('final_observation' in info and bool(done.any())) else next_obs
                next_val = agent.value(boot_obs)

            dur = info['duration'] if use_latent else torch.ones((num_envs,), device = device)
            prim_steps += int(dur.sum().item())

            for k, v in (('obs', obs), ('act', action), ('lp', lp), ('rew', rew), ('val', val), ('next_val', next_val), ('term', term), ('trunc', trunc), ('dur', dur)):
                buf[k].append(v)
            obs = next_obs

        rew_t, val_t, nval_t, term_t, trunc_t, dur_t = (rearrange(buf[k], 't e -> e t') for k in ('rew', 'val', 'next_val', 'term', 'trunc', 'dur'))

        with torch.no_grad():
            gae = (
                calc_smdp_gae(rew_t, val_t, dur_t, next_values = nval_t, terminated = term_t, truncated = trunc_t, gamma = gamma, lam = gae_lam, trace_power = 0., use_accelerated = False)
                if use_latent else
                calc_gae(rew_t, val_t, next_values = nval_t, terminated = term_t, truncated = trunc_t, gamma = gamma, lam = gae_lam, use_accelerated = False, return_advantages = True)
            )

        flat_obs, flat_act = rearrange(buf['obs'], 't e d -> (e t) d'), rearrange(buf['act'], 't e d -> (e t) d')
        flat_lp = rearrange(buf['lp'], 't e -> (e t)')
        flat_val, flat_adv, flat_ret = (rearrange(x, 'e t -> (e t)') for x in (val_t, gae.advantages, gae.returns))

        agent.train()
        for _ in range(ppo_epochs):
            perm = torch.randperm(flat_obs.shape[0], device = device)
            for s in range(0, flat_obs.shape[0], minibatch_size):
                idx = perm[s : s + minibatch_size]
                dist = agent.dist(flat_obs[idx])
                new_lp = dist.log_prob(flat_act[idx]).sum(dim = -1)
                loss = (
                    ppo_actor_loss(new_lp, flat_lp[idx], flat_adv[idx], eps_clip = 0.2, normalize_advantages = True).mean()
                    + 0.5 * clipped_value_loss_fn(agent.value(flat_obs[idx]), flat_val[idx], flat_ret[idx], clip = 0.2)
                    - ent_coef * dist.entropy().sum(dim = -1).mean()
                )
                step_optimizer(opt, loss, agent.parameters(), max_norm = 0.5)

        sched.step()
        env.commit_stats()
        if upd % eval_every == 0 or upd == num_updates:
            log_eval(upd)

    return agent, env, history

# plots & comparison video

def plot_basin_map(ax, grid: dict, latents: np.ndarray, alpha = 0.30, annotate = False):
    cmap = matplotlib.colors.ListedColormap([s[4] for s in SKILL_SPECS])
    ax.imshow(grid['nearest_skill'], origin = 'lower', extent = (-1., 1., -1., 1.), cmap = cmap, vmin = -0.5, vmax = NUM_SKILLS - 0.5, alpha = alpha, interpolation = 'nearest')
    for m, (_, short, _, _, color, _) in enumerate(SKILL_SPECS):
        ax.scatter(latents[m, 0], latents[m, 1], s = 180, marker = '*', color = color, edgecolors = '#0f172a', linewidths = 1.1, zorder = 6)
        if annotate:
            ax.annotate(short, xy = (latents[m, 0], latents[m, 1]), xytext = (0, 9), textcoords = 'offset points', ha = 'center', fontsize = 8, fontweight = 'bold', bbox = dict(boxstyle = 'round,pad=0.14', fc = 'white', ec = color, alpha = 0.88))
    ax.set(xlim = (-1.05, 1.05), ylim = (-1.05, 1.05), xlabel = 'Latent $z_1$', ylabel = 'Latent $z_2$')
    ax.grid(True, alpha = 0.25)

def add_banner(frame: np.ndarray, title: str, subtitle: str, accent: tuple[int, int, int]) -> np.ndarray:
    bar = Image.new('RGB', (frame.shape[1], 48), (15, 23, 42))
    draw = ImageDraw.Draw(bar)
    draw.text((10, 6), title, fill = (248, 250, 252), font = ImageFont.load_default(size = 13))
    draw.text((10, 25), subtitle, fill = (203, 213, 225), font = ImageFont.load_default(size = 11))
    draw.rectangle((0, 45, bar.width, 48), fill = accent)
    return np.concatenate((np.asarray(bar), frame), axis = 0)

def write_video_pair(stem: Path, frames: list[np.ndarray], fps = 15):
    imageio.mimsave(stem.with_suffix('.mp4'), frames, fps = fps, codec = 'libx264', pixelformat = 'yuv420p', macro_block_size = 1, output_params = ['-movflags', '+faststart'])
    Image.fromarray(frames[0]).save(stem.with_suffix('.webp'), save_all = True, append_images = [Image.fromarray(f) for f in frames[1::2]], duration = int(2000 / fps), loop = 0, quality = 82)

def save_figures(out_path: Path, stage1_res: dict, hl_eval: dict, hl_hist: list[dict], flat_hist: list[dict], rand_eval: dict, num_stages = 4):
    grid, ev = stage1_res['grid'], stage1_res['eval']

    fig, axes = plt.subplots(1, 3, figsize = (15.5, 4.8), dpi = 160)
    fig.suptitle('SuperMimic 2D Latent Skill Space & High-Level Retrieval (RUKA-v2 Hand)', fontsize = 11.5, fontweight = 'bold')

    plot_basin_map(axes[0], grid, ev['latents'], alpha = 0.32, annotate = True)
    axes[0].set_aspect('equal')
    axes[0].set_title('Stage 1: Learned 2D Skill Basins')

    im = axes[1].imshow(ev['tip_err_mat'], cmap = 'YlGnBu', vmin = 0., vmax = max(12.0, float(ev['tip_err_mat'].max())))
    shorts = [s[1] for s in SKILL_SPECS]
    axes[1].set_xticks(range(NUM_SKILLS), shorts, rotation = 30, ha = 'right', fontsize = 8.5)
    axes[1].set_yticks(range(NUM_SKILLS), shorts, fontsize = 8.5)
    axes[1].set(xlabel = 'Commanded Skill', ylabel = 'Initial Pose', title = f'Stage 1: 6×6 Error (Acc {ev["transition_acc"]*100:.0f}%)')
    for i in range(NUM_SKILLS):
        for j in range(NUM_SKILLS):
            v = ev['tip_err_mat'][i, j]
            axes[1].text(j, i, f'{v:.1f}', ha = 'center', va = 'center', fontsize = 7.8, color = 'white' if v > 7.5 else '#0f172a', fontweight = 'bold')
    fig.colorbar(im, ax = axes[1], fraction = 0.046, pad = 0.04, label = 'Fingertip Error (mm)')

    plot_basin_map(axes[2], grid, ev['latents'], alpha = 0.24)
    if exists(hl_eval['emitted_z']) and exists(hl_eval['emitted_task']):
        for m, (_, short, _, _, color, _) in enumerate(SKILL_SPECS):
            mk = hl_eval['emitted_task'] == m
            if mk.any():
                axes[2].scatter(hl_eval['emitted_z'][mk, 0], hl_eval['emitted_z'][mk, 1], s = 26, color = color, alpha = 0.72, edgecolors = 'white', linewidths = 0.3, label = short, zorder = 4)
    axes[2].set_aspect('equal')
    axes[2].set_title(f'Stage 2 (x-ppo): Emitted $z$ (Basin {(hl_eval.get("basin_retrieval_acc") or 0.)*100:.1f}%)')
    axes[2].legend(fontsize = 7.5, loc = 'upper right', framealpha = 0.9, ncol = 2)
    fig.tight_layout(rect = (0, 0, 1, 0.93))
    fig.savefig(out_path / 'ruka_latent_space_and_retrieval.png')
    plt.close(fig)

    if flat_hist:
        fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize = (12.0, 4.6), dpi = 160)
        fig.suptitle(f'Stage 2 RUKA-v2 {num_stages}-Stage Task: Latent Skills vs From-Scratch Flat 18-DOF x-ppo', fontsize = 11.0, fontweight = 'bold')
        for name, color, ls, hist in (('HL x-ppo + SuperMimic 2D Latent', '#1d4ed8', '-', hl_hist), ('Flat 18-DOF x-ppo (From Scratch)', '#dc2626', '--', flat_hist)):
            xs = [r['prim_steps'] / 1e3 for r in hist]
            rets, succs = [r['mean_return'] for r in hist], [r['success_rate'] * 100. for r in hist]
            ax_r.plot(xs, rets, color = color, linestyle = ls, linewidth = 2.2, marker = 'o', markersize = 3.5, label = f'{name} ({rets[-1]:.1f})')
            ax_s.plot(xs, succs, color = color, linestyle = ls, linewidth = 2.2, marker = 'o', markersize = 3.5, label = f'{name} ({succs[-1]:.0f}%)')
        ax_r.axhline(rand_eval['mean_return'], color = '#475569', linestyle = ':', linewidth = 1.6, label = f'Random Latent ({rand_eval["mean_return"]:.1f})')
        ax_s.axhline(rand_eval['success_rate'] * 100., color = '#475569', linestyle = ':', linewidth = 1.6, label = f'Random Latent ({rand_eval["success_rate"]*100:.0f}%)')
        ax_r.set(xlabel = 'Primitive MuJoCo Steps (×1,000)', ylabel = 'Mean Return', title = 'Held-Out Episode Return')
        ax_s.set(xlabel = 'Primitive MuJoCo Steps (×1,000)', ylabel = 'Success Rate (%)', title = f'{num_stages}-Stage Completion (%)', ylim = (-4, 106))
        for ax in (ax_r, ax_s):
            ax.grid(True, alpha = 0.25)
            ax.legend(fontsize = 8.0, loc = 'best')
        fig.tight_layout(rect = (0, 0, 1, 0.92))
        fig.savefig(out_path / 'ruka_hl_learning_curves.png')
        plt.close(fig)

@torch.no_grad()
def save_comparison_videos(out_path: Path, model: mujoco.MjModel, target_geom_id: int, skills: dict, stage1_res: dict, hl_agent: PPOAgent, hl_norm, actor: LatentMotionActor, flat_agent: PPOAgent | None, flat_norm, device: torch.device, horizon = 8, render_size = 360):
    fig = plt.figure(figsize = (render_size / 100., render_size / 100.), dpi = 100)
    ax = fig.add_axes([0.13, 0.13, 0.82, 0.80])
    plot_basin_map(ax, stage1_res['grid'], stage1_res['eval']['latents'], alpha = 0.30, annotate = True)
    ax.tick_params(labelsize = 8)
    fig.canvas.draw()
    base_map = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    bbox = ax.get_window_extent()
    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    plt.close(fig)

    def rollout_policy(agent: PPOAgent, norm_state, use_latent: bool):
        raw = RukaDexterousTaskEnv(copy.deepcopy(model), target_geom_id, skills, num_stages = len(DEMO_SEQ), max_steps = 150, hold_steps = 4, jitter_m = 0., render_mode = 'rgb_array', render_size = render_size, forced_sequence = DEMO_SEQ)
        env = wrap_task_env(raw, actor = actor if use_latent else None, skills = skills, horizon = horizon, device = device, eval_mode = True, record_frames = use_latent, norm_mode = 'frozen', norm_state = norm_state)
        agent.eval()
        obs, info = env.reset(seed = 777)
        hand_panels, lat_panels, z_trail, done, stages = [], [], [], False, 0

        while not done:
            cur_task = int(info.get('next_task_id', info.get('current_task_id', 0)))
            act = agent.dist(obs).mean
            obs, _, term, trunc, info = env.step(act)
            done = bool((term | trunc).item())
            stages = int(np.asarray(info.get('stages_completed', 0)).item())
            frames = info.get('substep_frames', []) if use_latent else [raw.render()]

            if use_latent:
                z_np = act[0].cpu().numpy()
                basin = int(lookup_grid_skills(z_np[None], stage1_res['grid'])[0])
                px = int(round(x0 + (float(z_np[0]) + 1.05) / 2.10 * (x1 - x0)))
                py = int(round(render_size - (y0 + (float(z_np[1]) + 1.05) / 2.10 * (y1 - y0))))
                z_trail.append((px, py))

            for fr in frames:
                if not exists(fr):
                    continue
                if use_latent:
                    panel_r = Image.fromarray(base_map.copy())
                    draw = ImageDraw.Draw(panel_r)
                    for i in range(1, len(z_trail)):
                        draw.line((z_trail[i - 1], z_trail[i]), fill = (30, 41, 59), width = 2)
                    col = tuple(int(c * 255) for c in SKILL_SPECS[cur_task][5][:3])
                    draw.ellipse((px - 9, py - 9, px + 9, py + 9), fill = col, outline = (15, 23, 42), width = 3)
                    draw.line((px - 15, py, px + 15, py), fill = (15, 23, 42), width = 2)
                    draw.line((px, py - 15, px, py + 15), fill = (15, 23, 42), width = 2)
                    tag = 'ALL 6/6 DONE!' if stages >= len(DEMO_SEQ) else f'Target: {SKILL_SPECS[cur_task][0]}'
                    hand_panels.append(add_banner(fr, 'WITH Latent Action (HL x-ppo, 2D z)', f'{tag} | Completed: {stages}/{len(DEMO_SEQ)}', (22, 163, 74) if stages >= len(DEMO_SEQ) else (37, 99, 235)))
                    lat_panels.append(add_banner(np.asarray(panel_r), f'Emitted Latent z = [{z_np[0]:+.2f}, {z_np[1]:+.2f}]', f'Basin: {SKILL_SPECS[basin][0]} (env-ssl-wrapper)', (22, 163, 74) if basin == cur_task else (217, 119, 6)))
                else:
                    hand_panels.append(add_banner(fr, 'WITHOUT Latent Action (Flat 18-DOF x-ppo)', f'Target: {SKILL_SPECS[cur_task][0]} | Completed: {stages}/{len(DEMO_SEQ)}', (220, 38, 38)))

        env.close()
        return hand_panels, lat_panels, stages

    hl_panels, lat_panels, hl_stages = rollout_policy(hl_agent, hl_norm, use_latent = True)
    divider = np.full((hl_panels[0].shape[0], 4, 3), (15, 23, 42), dtype = np.uint8)
    write_video_pair(out_path / 'ruka_with_latent_action', [np.concatenate((h, divider, l), axis = 1) for h, l in zip(hl_panels, lat_panels)])

    flat_stages = 0
    if exists(flat_agent):
        flat_panels, _, flat_stages = rollout_policy(flat_agent, flat_norm, use_latent = False)
        n = min(max(len(hl_panels) + 18, len(flat_panels)), 120)
        combined = [np.concatenate((flat_panels[min(i, len(flat_panels) - 1)], divider, hl_panels[min(i, len(hl_panels) - 1)], divider, lat_panels[min(i, len(lat_panels) - 1)]), axis = 1) for i in range(n)]
        write_video_pair(out_path / 'ruka_with_vs_without_latent', combined)
        write_video_pair(out_path / 'ruka_without_latent_action', flat_panels[:n])
        Image.fromarray(np.concatenate([combined[p] for p in np.linspace(0, min(len(hl_panels), n) - 1, 5).astype(int)], axis = 0)).save(out_path / 'ruka_with_vs_without_contact_sheet.png')

    print(f'[videos] with latent: {hl_stages}/{len(DEMO_SEQ)} in {len(hl_panels)} steps | without: {flat_stages}/{len(DEMO_SEQ)} -> {out_path}', flush = True)
    return hl_stages, flat_stages

# main

def main(out_dir = None, cache_root = None, seed = 42, dim_latent = 2, stage1_iters = 95, stage2_steps = 300_000, num_stages = 4, max_task_steps = 128, macro_horizon = 8, num_envs = 16, skip_stage1 = False, skip_flat_baseline = False):
    out_path = Path(default(out_dir, SCRIPT_DIR / 'results_ruka_latent_skills'))
    shared = Path(default(cache_root, SCRIPT_DIR / 'results_ruka_hand'))
    device = select_device()
    out_path.mkdir(parents = True, exist_ok = True)

    base_model, pkl_path, _ = build_ruka_model(shared / 'cache')
    skills = extract_skill_clips(retarget_human_hand(base_model, pkl_path, shared / 'ruka_human_mocap.npz'))

    ckpt_path = out_path / 'stage1_latent_actor.pt'
    if skip_stage1 and ckpt_path.exists():
        print(f'[stage 1] loading {ckpt_path}', flush = True)
        ckpt = torch.load(ckpt_path, map_location = device, weights_only = False)
        latent_actor = LatentMotionActor(**ckpt['actor_kwargs'], activation = SiLU()).to(device)
        latent_actor.load_state_dict(ckpt['actor_state_dict'])
        latent_actor.eval()
        ev = eval_stage1(base_model, skills, latent_actor, device, with_grid = True)
        stage1_res = dict(history = ckpt['history'], grid = ev.pop('grid'), eval = ev)
    else:
        latent_actor, stage1_res = train_stage1(base_model, skills, out_path, device, seed = seed, dim_latent = dim_latent, num_iterations = stage1_iters)

    task_model, target_geom_id = build_ruka_task_model(shared / 'cache')
    make_task_env = lambda: RukaDexterousTaskEnv(task_model, target_geom_id, skills, num_stages = num_stages, max_steps = max_task_steps, hold_steps = 4)

    hl_device = torch.device('cpu')
    frozen_actor = copy.deepcopy(latent_actor).to(hl_device).eval()
    stage1_latents, stage1_grid = stage1_res['eval']['latents'], stage1_res['grid']

    rand_eval = eval_stage2(None, make_task_env, actor = frozen_actor, skills = skills, horizon = macro_horizon, episodes = 40, seed = 2000, device = hl_device, random_latent_dim = dim_latent, stage1_latents = stage1_latents, stage1_grid = stage1_grid)
    print(f'[random latent] ret {rand_eval["mean_return"]:.2f} | stages {rand_eval["mean_stages"]:.2f}/{num_stages} | succ {rand_eval["success_rate"]*100:.1f}%', flush = True)

    hl_agent, hl_env, hl_hist = train_stage2_ppo('hl x-ppo (2d latent)', make_task_env, actor = frozen_actor, skills = skills, stage1_latents = stage1_latents, stage1_grid = stage1_grid, num_envs = num_envs, total_prim_steps = stage2_steps, horizon = macro_horizon, rollout_steps = 24, num_stages = num_stages, seed = seed, device = hl_device)
    hl_final = eval_stage2(hl_agent, make_task_env, actor = frozen_actor, skills = skills, norm_state = hl_env.state_dict(), horizon = macro_horizon, episodes = 48, seed = 3000, device = hl_device, stage1_latents = stage1_latents, stage1_grid = stage1_grid)
    torch.save(dict(agent = hl_agent.state_dict(), norm = hl_env.state_dict()), out_path / 'stage2_hl_ppo.pt')

    flat_agent = flat_env = None
    flat_hist, flat_final = [], dict(mean_return = 0., mean_stages = 0., success_rate = 0.)
    if not skip_flat_baseline:
        flat_agent, flat_env, flat_hist = train_stage2_ppo('flat x-ppo (18-dof)', make_task_env, num_envs = num_envs, total_prim_steps = stage2_steps, horizon = 1, rollout_steps = 24 * macro_horizon, num_stages = num_stages, seed = seed, device = hl_device)
        flat_final = eval_stage2(flat_agent, make_task_env, norm_state = flat_env.state_dict(), episodes = 48, seed = 3000, device = hl_device)
        torch.save(dict(agent = flat_agent.state_dict(), norm = flat_env.state_dict()), out_path / 'stage2_flat_ppo.pt')

    save_figures(out_path, stage1_res, hl_final, hl_hist, flat_hist, rand_eval, num_stages = num_stages)
    hl_demo, flat_demo = save_comparison_videos(out_path, task_model, target_geom_id, skills, stage1_res, hl_agent, hl_env.state_dict(), frozen_actor, flat_agent, flat_env.state_dict() if exists(flat_env) else None, hl_device, horizon = macro_horizon)
    hl_env.close()
    if exists(flat_env):
        flat_env.close()

    summary = dict(
        stage1 = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stage1_res['eval'].items()},
        stage2 = dict(
            hl_ppo = {k: v for k, v in hl_final.items() if k not in ('emitted_z', 'emitted_task')},
            flat_ppo = {k: v for k, v in flat_final.items() if k not in ('emitted_z', 'emitted_task')},
            random_latent = {k: v for k, v in rand_eval.items() if k not in ('emitted_z', 'emitted_task')},
            demo_stages = dict(with_latent = f'{hl_demo}/{len(DEMO_SEQ)}', without_latent = f'{flat_demo}/{len(DEMO_SEQ)}'),
        ),
    )
    (out_path / 'results.json').write_text(json.dumps(summary, indent = 2))

    print('\n=== final summary (18-dof ruka-v2, env-ssl-wrapper 0.7.2) ===', flush = True)
    print(f'  stage 1 (2d skills)          : 36-trans {stage1_res["eval"]["transition_acc"]*100:5.1f}% | tips {stage1_res["eval"]["tip_mm_mean"]:.2f}mm | min|dz| {stage1_res["eval"]["min_latent_dist"]:.2f}', flush = True)
    print(f'  stage 2 with latent (hl)     : succ {hl_final["success_rate"]*100:5.1f}% | stages {hl_final["mean_stages"]:.2f}/{num_stages} | ret {hl_final["mean_return"]:5.2f} | basin {hl_final["basin_retrieval_acc"]*100:5.1f}%', flush = True)
    if not skip_flat_baseline:
        print(f'  stage 2 without latent (flat): succ {flat_final["success_rate"]*100:5.1f}% | stages {flat_final["mean_stages"]:.2f}/{num_stages} | ret {flat_final["mean_return"]:5.2f}', flush = True)
    print(f'  stage 2 random latent        : succ {rand_eval["success_rate"]*100:5.1f}% | stages {rand_eval["mean_stages"]:.2f}/{num_stages} | ret {rand_eval["mean_return"]:5.2f}', flush = True)
    print(f'  6-skill video                : with {hl_demo}/{len(DEMO_SEQ)} vs without {flat_demo}/{len(DEMO_SEQ)} -> {out_path}', flush = True)

if __name__ == '__main__':
    fire.Fire(main)
