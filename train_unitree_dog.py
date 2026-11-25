# /// script
# requires-python = ">= 3.10"
# dependencies = [
#     "supermimic-pytorch",
#     "memmap-replay-buffer>=0.3.0",
#     "mujoco>=3.2",
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

import json
import math
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import einx
import fire
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from torch import Tensor, as_tensor
from torch.nn import Module, SiLU, init
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from einops import rearrange
from ema_pytorch import EMA
from memmap_replay_buffer import StreamReplayBuffer
from torch_einops_utils import pad_right_at_dim, safe_cat, temp_eval
from torch_einops_utils.shape import size
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP

from supermimic_pytorch import (
    ActionSmoothnessTerm, AnalyticalReward, Critic, ForwardKinematics,
    HistoryWindow, JointLimitPenalty, SuperMimic, TrackingTerm, WorldModel,
    fall_termination
)

# constants

GO2_RAW = 'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_go2/'
DOG_RAW = 'https://raw.githubusercontent.com/erwincoumans/motion_imitation/master/retarget_motion/data/'

# real German Shepherd mocap (Peng et al. 2020)

TRAIN_CLIPS = (
    ('dog_walk00_joint_pos.txt', 110, 593),   # pace
    ('dog_walk01_joint_pos.txt', 310, 1072),  # walk + S-curves
    ('dog_walk03_joint_pos.txt', 140, 548),   # stand -> walk -> trot
    ('dog_walk04_joint_pos.txt', 240, 548),   # diagonal trot
    ('dog_walk09_joint_pos.txt', 500, 1600),  # sweeping turns
)
EVAL_CLIP = ('dog_walk02_joint_pos.txt', 420, 911)

LEGS = ('FL', 'FR', 'RL', 'RR')
FOOT_BODIES = tuple(f'{leg}_foot' for leg in LEGS)
DOG_HIP_IDS = (6, 11, 16, 20)
DOG_PAW_IDS = (10, 15, 19, 23)
CLIP_KEYS = ('dog_pts', 'qpos', 'qvel', 'states', 'feet')

DOG_CHAINS = (
    ((0, 2, 3, 4, 5),         (0.92, 0.72, 0.42, 1.0), 0.028),
    ((0, 24, 25, 26),         (0.85, 0.62, 0.34, 1.0), 0.014),
    ((3, 6, 7, 8, 9, 10),     (0.22, 0.56, 0.96, 1.0), 0.017),
    ((3, 11, 12, 13, 14, 15), (0.94, 0.32, 0.32, 1.0), 0.017),
    ((0, 16, 17, 18, 19),     (0.10, 0.74, 0.80, 1.0), 0.017),
    ((0, 20, 21, 22, 23),     (0.96, 0.56, 0.16, 1.0), 0.017),
)

GO2_HIPS_LOCAL = np.array(
    [[0.1934, 0.1420, 0.], [0.1934, -0.1420, 0.], [-0.1934, 0.1420, 0.], [-0.1934, -0.1420, 0.]],
    dtype = np.float32
)
Q_DEFAULT = np.array([0., 0.85, -1.70] * 4, dtype = np.float32)
ACTION_SCALE = np.array([0.50, 0.90, 1.00] * 4, dtype = np.float32)

# heading-frame state (35-d): [root_z(1), root_tilt(4), q(12), v_heading(3), w_heading(3), dq(12)]

NUM_JOINTS = 12
DIM_STATE = 35
DIM_REF = 23

FPS = 30
SUBSTEPS = 8
FOOT_RADIUS = 0.022
MIN_HEIGHT = 0.16
MAX_TILT = 0.50
MAX_EVAL_REWARD = 8.0

DOG_SCENE_XML = """<mujoco model="dog"><visual>
  <headlight diffuse="0.65 0.65 0.65" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
  <rgba haze="0.15 0.25 0.35 1"/></visual><asset>
  <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
  <texture type="2d" name="gp" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
  <material name="gp" texture="gp" texuniform="true" texrepeat="5 5" reflectance="0.2"/></asset>
  <worldbody><light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
  <geom name="floor" size="0 0 0.05" type="plane" material="gp"/></worldbody></mujoco>"""

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cosine_anneal(start: float, end: float, frac: float) -> float:
    return end + (start - end) * 0.5 * (1. + math.cos(math.pi * frac))

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def fetch(url: str, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents = True, exist_ok = True)
        urllib.request.urlretrieve(url, dst)
    return dst

def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis = -1, keepdims = True) + 1e-8)

def smooth_1d(arr: np.ndarray, passes = 2) -> np.ndarray:
    pad = ((1, 1),) + ((0, 0),) * (arr.ndim - 1)
    for _ in range(passes):
        p = np.pad(arr, pad, mode = 'edge')
        arr = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
    return arr

def step_optimizer(opt, loss: Tensor, params, max_norm = 1., clip_value = None):
    params = list(params)
    opt.zero_grad()
    loss.backward()
    clip_grad_norm_(params, max_norm)
    if exists(clip_value):
        clip_grad_value_(params, clip_value)
    opt.step()

def gather_clip_seq(seq: Tensor, clip_ends: Tensor, starts: Tensor, length: int) -> Tensor:
    ends = clip_ends[starts.clamp(max = len(clip_ends) - 1)]
    frames = einx.add('b, l -> b l', starts, torch.arange(length, device = seq.device))
    return seq[torch.minimum(frames, ends[:, None])]

# heading-frame SE(2) / SO(3) geometry

def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ), axis = -1)

def quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, u = q[..., :1], q[..., 1:]
    t = 2. * np.cross(u, v)
    return v + w * t + np.cross(u, t)

def quat_yaw_np(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))

def qpos_qvel_to_heading_state(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    q_world = unit(qpos[..., 3:7])
    half = -0.5 * quat_yaw_np(q_world)
    zeros = np.zeros_like(half)
    q_yaw_inv = np.stack((np.cos(half), zeros, zeros, np.sin(half)), axis = -1)

    q_tilt = quat_mul_np(q_yaw_inv, q_world)
    q_tilt = np.where(q_tilt[..., :1] < 0., -q_tilt, q_tilt)

    v_heading = quat_rotate_np(q_yaw_inv, qvel[..., :3])
    w_heading = quat_rotate_np(q_tilt, qvel[..., 3:6])
    return np.concatenate((qpos[..., 2:3], q_tilt, qpos[..., 7:], v_heading, w_heading, qvel[..., 6:]), axis = -1).astype(np.float32)

def norm_stats(states_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_s = np.concatenate(states_list, axis = 0)
    min_std = np.array([0.05] + [0.08] * 4 + [0.15] * 12 + [0.30] * 3 + [0.50] * 3 + [1.50] * 12, dtype = np.float32)
    return all_s.mean(0).astype(np.float32), np.maximum(all_s.std(0).astype(np.float32), min_std)

# Unitree Go2 MJCF & differentiable FK dog retargeting

def build_go2_model(cache_dir: Path, kp = 40.0, kd = 0.5) -> mujoco.MjModel:
    go2_dir = cache_dir / 'unitree_go2'
    fetch(GO2_RAW + 'scene.xml', go2_dir / 'scene.xml')
    go2_xml = fetch(GO2_RAW + 'go2.xml', go2_dir / 'go2.xml').read_text()

    for el in ET.fromstring(go2_xml).findall('.//mesh'):
        fetch(GO2_RAW + 'assets/' + el.attrib['file'], go2_dir / 'assets' / el.attrib['file'])

    spec = mujoco.MjSpec.from_file(str(go2_dir / 'scene.xml'))

    for leg in LEGS:
        foot = spec.body(f'{leg}_calf').add_body(name = f'{leg}_foot', pos = [-0.002, 0., -0.213])
        foot.mass, foot.inertia = 1e-6, [1e-9, 1e-9, 1e-9]

    for act in spec.actuators:
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = kp
        act.biasprm[:3] = [0., -kp, -kd]
        act.forcelimited, act.forcerange, act.ctrllimited = True, act.ctrlrange.copy(), False

    spec.option.timestep = 1. / (FPS * SUBSTEPS)
    return spec.compile()

def load_dog_clip(dog_dir: Path, fname: str, start: int, end: int | None, stride = 2) -> np.ndarray:
    rows = [[float(x) for x in line.split(',')] for line in fetch(DOG_RAW + fname, dog_dir / fname).read_text().splitlines() if line.strip()]
    pts = np.asarray(rows, dtype = np.float32).reshape(-1, 27, 3)[start:end:stride]

    # Y-up -> Z-up rotation, grounded at z = 0
    pts_zup = np.stack((pts[..., 0], -pts[..., 2], pts[..., 1]), axis = -1)
    pts_zup[..., :2] -= pts_zup[0, 0, :2]
    pts_zup[..., 2] -= np.percentile(pts_zup[:, list(DOG_PAW_IDS), 2], 2)
    return pts_zup

def retarget_dog_clip(model: mujoco.MjModel, dog_pts: np.ndarray, scale = 0.72, steps = 300, lr = 0.05) -> dict[str, np.ndarray]:
    scaled = dog_pts * scale
    pelvis, chest = smooth_1d(scaled[:, 0]), smooth_1d(scaled[:, 3])

    # horizontal left axis (Peng et al. 2020) avoids scapula-bobbing roll
    fwd = unit(chest - pelvis)
    left = unit(np.cross(np.array([0., 0., 1.], dtype = np.float32), fwd))
    R = np.stack((fwd, left, unit(np.cross(fwd, left))), axis = -1)

    root_quat = np.zeros((len(R), 4), dtype = np.float64)
    for t, mat in enumerate(R):
        mujoco.mju_mat2Quat(root_quat[t], mat.astype(np.float64).ravel())
        if t > 0 and np.dot(root_quat[t], root_quat[t - 1]) < 0.:
            root_quat[t] = -root_quat[t]

    root_pos = 0.5 * (pelvis + chest)
    go2_hips = root_pos[:, None] + np.einsum('tij,kj->tki', R, GO2_HIPS_LOCAL)
    target_paws = go2_hips + scaled[:, list(DOG_PAW_IDS)] - scaled[:, list(DOG_HIP_IDS)]
    target_paws[..., 2] = np.maximum(FOOT_RADIUS, scaled[:, list(DOG_PAW_IDS), 2] + FOOT_RADIUS)

    # differentiable FK leg retargeting

    fk = ForwardKinematics.from_mujoco(model, tracked_body_names = FOOT_BODIES)
    low, high = as_tensor(model.jnt_range[1:, 0], dtype = torch.float32), as_tensor(model.jnt_range[1:, 1], dtype = torch.float32)
    rp_t, rq_t, tp_t = (as_tensor(x, dtype = torch.float32) for x in (root_pos, root_quat, target_paws))
    q_def = as_tensor(Q_DEFAULT, dtype = torch.float32)

    qj = q_def[None].repeat(len(dog_pts), 1).clone().requires_grad_(True)
    opt = torch.optim.Adam([qj], lr = lr)

    for _ in range(steps):
        pred = fk(qpos = torch.cat((rp_t, rq_t, qj), dim = -1))['body_pos']
        loss = (pred - tp_t).square().mean() + 0.002 * (qj - q_def).square().mean() + 0.005 * (qj[1:] - qj[:-1]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            qj.clamp_(low, high)

    qpos = torch.cat((rp_t, rq_t, qj.detach()), dim = -1).numpy().astype(np.float32)

    dt = 1. / FPS
    w_body = np.zeros((len(qpos), 3), dtype = np.float32)
    for t in range(len(qpos)):
        t0, t1 = max(0, t - 1), min(len(qpos) - 1, t + 1)
        res = np.zeros(3)
        mujoco.mju_subQuat(res, root_quat[t1], root_quat[t0])
        w_body[t] = res / (max(1, t1 - t0) * dt)

    qvel = np.concatenate((np.gradient(qpos[:, :3], dt, axis = 0), w_body, np.gradient(qpos[:, 7:], dt, axis = 0)), axis = -1).astype(np.float32)
    states = qpos_qvel_to_heading_state(qpos, qvel)
    feet_h = fk(qpos = torch.cat((torch.zeros((len(states), 2)), as_tensor(states[:, :17])), dim = -1))['body_pos'].numpy().astype(np.float32)
    paw_err_mm = float((fk(qpos = as_tensor(qpos))['body_pos'] - tp_t).norm(dim = -1).mean().item() * 1000.)

    return dict(dog_pts = scaled.astype(np.float32), qpos = qpos, qvel = qvel, states = states, feet = feet_h, paw_err_mm = paw_err_mm)

def load_or_retarget_clips(model: mujoco.MjModel, cache_dir: Path, npz_path: Path) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    if npz_path.exists():
        raw = dict(np.load(npz_path))
        return [{k: raw[f'train_{i}_{k}'] for k in CLIP_KEYS} for i in range(int(raw['num_train']))], {k: raw[f'eval_{k}'] for k in CLIP_KEYS}

    dog_dir = cache_dir / 'dog_mocap'
    specs = [(f'train_{i}', *s) for i, s in enumerate(TRAIN_CLIPS)] + [('eval', *EVAL_CLIP)]
    clips, save_dict = [], dict(num_train = np.array(len(TRAIN_CLIPS)))

    for tag, fname, start, end in specs:
        clip = retarget_dog_clip(model, load_dog_clip(dog_dir, fname, start, end))
        print(f'[retarget {tag}] {fname} [{start}:{end}] -> {len(clip["qpos"])} frames @ {FPS}Hz | paw {clip["paw_err_mm"]:.1f} mm', flush = True)
        clips.append(clip)
        save_dict.update({f'{tag}_{k}': clip[k] for k in CLIP_KEYS})

    npz_path.parent.mkdir(parents = True, exist_ok = True)
    np.savez(npz_path, **save_dict)
    return clips[:-1], clips[-1]

# parallel MuJoCo Unitree Go2 environments with RSI & fall termination

class Go2Envs:
    def __init__(
        self,
        model: mujoco.MjModel,
        clips: list[dict[str, np.ndarray]],
        mean: np.ndarray,
        std: np.ndarray,
        device: torch.device,
        num_envs = 16,
        horizon = 128,
        goal_window = 8,
        rsi_pos_noise = 0.04,
        rsi_vel_noise = 0.15,
        is_eval = False
    ):
        self.model = model
        self.clips = clips
        self.mean = mean
        self.std = std
        self.device = device
        self.rsi_pos_noise = rsi_pos_noise
        self.rsi_vel_noise = rsi_vel_noise
        self.is_eval = is_eval
        self.num_envs = 1 if is_eval else num_envs

        self.clip_lengths = np.array([len(c['qpos']) for c in clips], dtype = np.int64)
        self.clip_offsets = np.concatenate(([0], np.cumsum(self.clip_lengths[:-1])))
        self.clip_probs = self.clip_lengths / self.clip_lengths.sum()

        all_states = np.concatenate([c['states'] for c in clips], axis = 0)
        self.all_states = as_tensor(all_states, device = device)
        self.all_refs_norm = as_tensor((all_states[:, :DIM_REF] - mean[:DIM_REF]) / std[:DIM_REF], device = device)
        self.all_feet = as_tensor(np.concatenate([c['feet'] for c in clips], axis = 0), device = device)
        self.clip_ends = as_tensor(
            np.concatenate([np.full(n, off + n - 1, dtype = np.int64) for n, off in zip(self.clip_lengths, self.clip_offsets)]),
            device = device
        )

        gw = HistoryWindow(goal_window, DIM_REF, future = True, pad_mode = 'replicate')
        self.goal_windows = [gw(as_tensor((c['states'][1:, :DIM_REF] - mean[:DIM_REF]) / std[:DIM_REF], device = device)[None])[0] for c in clips]

        self.horizon = int(self.clip_lengths[0] - 1) if is_eval else horizon
        self.datas = [mujoco.MjData(model) for _ in range(self.num_envs)]
        self.clip_ids = np.zeros(self.num_envs, dtype = np.int64)
        self.frames = np.zeros(self.num_envs, dtype = np.int64)
        self.steps = np.zeros(self.num_envs, dtype = np.int64)
        self.states = np.zeros((self.num_envs, DIM_STATE), dtype = np.float32)

        for i in range(self.num_envs):
            self.reset(i)

    def global_frames(self) -> np.ndarray:
        return self.clip_offsets[self.clip_ids] + self.frames

    def reset(self, index: int):
        cid = 0 if self.is_eval else int(np.random.choice(len(self.clips), p = self.clip_probs))
        clip = self.clips[cid]
        frame = 0 if self.is_eval else int(np.random.randint(max(1, len(clip['qpos']) - 24)))

        data = self.datas[index]
        mujoco.mj_resetData(self.model, data)
        qpos, qvel = clip['qpos'][frame].copy(), clip['qvel'][frame].copy()

        if not self.is_eval:
            qpos[2] += np.random.uniform(-0.02, 0.02)
            qpos[7:] = np.clip(qpos[7:] + np.random.normal(0., self.rsi_pos_noise, NUM_JOINTS), self.model.jnt_range[1:, 0], self.model.jnt_range[1:, 1])
            qvel[:6] += np.random.normal(0., self.rsi_vel_noise, 6)
            qvel[6:] += np.random.normal(0., self.rsi_vel_noise * 2., NUM_JOINTS)

        data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, qpos[7:]
        mujoco.mj_forward(self.model, data)

        self.clip_ids[index], self.frames[index], self.steps[index] = cid, frame, 0
        self.states[index] = qpos_qvel_to_heading_state(data.qpos, data.qvel)

    def normalize(self, states: np.ndarray | None = None) -> np.ndarray:
        return (default(states, self.states) - self.mean) / self.std

    def goals(self) -> Tensor:
        return torch.stack([self.goal_windows[c][min(f, len(self.goal_windows[c]) - 1)] for c, f in zip(self.clip_ids, self.frames)])

    def is_fallen(self, state: np.ndarray) -> bool:
        up_z = 1. - 2. * float(state[2] ** 2 + state[3] ** 2)
        return (state[0] < MIN_HEIGHT) or (up_z < 1. - MAX_TILT) or (not np.isfinite(state).all())

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        next_states = np.empty_like(self.states)
        dones, falls = np.zeros(self.num_envs, dtype = bool), np.zeros(self.num_envs, dtype = bool)

        for i, data in enumerate(self.datas):
            q_target = Q_DEFAULT + ACTION_SCALE * np.clip(actions[i], -1., 1.)
            data.ctrl[:] = np.clip(q_target, self.model.jnt_range[1:, 0], self.model.jnt_range[1:, 1])
            mujoco.mj_step(self.model, data, nstep = SUBSTEPS)

            ns = qpos_qvel_to_heading_state(data.qpos, data.qvel)
            next_states[i] = ns
            self.frames[i] += 1
            self.steps[i] += 1
            falls[i] = self.is_fallen(ns)

            if falls[i] or self.steps[i] >= self.horizon or self.frames[i] >= self.clip_lengths[self.clip_ids[i]] - 1:
                dones[i] = True
                if not self.is_eval:
                    self.reset(i)
            else:
                self.states[i] = ns

        return next_states, dones, falls

# actor, analytical FK reward, termination & evaluation

class TrackingActor(Module):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        dim_ref: int,
        obs_window = 2,
        action_window = 1,
        goal_window = 8,
        dim_hidden = 256,
        depth = 3,
        dim_latent = 4
    ):
        super().__init__()
        self.dim_action = dim_action
        self.obs_history = HistoryWindow(obs_window, dim_state, pad_mode = 'replicate')
        self.action_history = HistoryWindow(action_window, dim_action, lag = 1, pad_mode = 'zero')
        self.goal_window = HistoryWindow(goal_window, dim_ref, future = True, pad_mode = 'replicate')

        self.dim_in = obs_window * dim_state + action_window * dim_action + goal_window * dim_ref + dim_latent
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

    def step_policy(self, state: Tensor, goals: Tensor, caches: tuple, noise = 0., random_std = 0.):
        obs_cache, act_cache = caches
        obs, obs_cache = self.obs_history.step(state, obs_cache)
        past_act = self.action_history.read(act_cache)

        if random_std > 0.:
            act = (torch.randn((len(state), self.dim_action), device = state.device) * random_std).clamp(-1., 1.)
        else:
            act = self(safe_cat([rearrange(t, 'b w d -> b (w d)') for t in (obs, past_act, goals)], dim = -1))
            if noise > 0.:
                act = (act + torch.randn_like(act) * noise).clamp(-1., 1.)

        return act, (obs_cache, self.action_history.push(act, act_cache))

def unpack_heading_state(raw: Tensor, fk_feet: ForwardKinematics | None = None) -> dict[str, Tensor]:
    root_z = raw[..., 0:1]
    root_quat = F.normalize(raw[..., 1:5], dim = -1, eps = 1e-8)
    joint_pos = raw[..., 5:17]
    zeros_xy = torch.zeros((*raw.shape[:-1], 2), device = raw.device, dtype = raw.dtype)
    root_pos = torch.cat((zeros_xy, root_z), dim = -1)

    out = dict(
        root_pos = root_pos,
        root_z = root_z,
        root_quat = root_quat,
        joint_pos = joint_pos,
        root_lin_vel = raw[..., 17:20],
        root_ang_vel = raw[..., 20:23]
    )
    if exists(fk_feet):
        out['body_pos'] = fk_feet(qpos = torch.cat((root_pos, root_quat, joint_pos), dim = -1))['body_pos']
    return out

def make_reward(model: mujoco.MjModel, mean: Tensor | None = None, std: Tensor | None = None, joint_weight = 1.5, foot_weight = 2.0) -> AnalyticalReward:
    fk_feet = ForwardKinematics.from_mujoco(model, tracked_body_names = FOOT_BODIES)

    def state_transform(state: Tensor) -> dict[str, Tensor]:
        raw = state * std.to(state) + mean.to(state) if exists(mean) and exists(std) else state
        return unpack_heading_state(raw, fk_feet)

    terms = dict(
        joint_pos = TrackingTerm('joint_pos', std = 0.35, weight = joint_weight, dim_weights = torch.full((NUM_JOINTS,), 1. / NUM_JOINTS)),
        root_height = TrackingTerm('root_z', std = 0.05, weight = 1.0),
        root_orientation = TrackingTerm('root_quat', metric = 'rotation', std = 0.25, weight = 1.0),
        root_lin_vel = TrackingTerm('root_lin_vel', std = 0.35, weight = 1.5),
        root_ang_vel = TrackingTerm('root_ang_vel', std = 0.50, weight = 1.0),
        joint_limit = JointLimitPenalty(model.jnt_range[1:, 0], model.jnt_range[1:, 1], key = 'joint_pos', margin = 0.05, weight = -0.5),
        action_smoothness = ActionSmoothnessTerm(weight = -0.02)
    )
    if foot_weight > 0.:
        terms['feet'] = TrackingTerm('body_pos', std = 0.055, weight = foot_weight, reduce_items = True)

    return AnalyticalReward(terms, state_transform = state_transform)

def make_termination(mean: Tensor, std: Tensor, temperature = 0.015):
    return lambda states, references = None: fall_termination(
        unpack_heading_state(states * std.to(states) + mean.to(states)),
        min_height = MIN_HEIGHT, tilt_threshold = MAX_TILT, temperature = temperature
    )

@torch.no_grad()
def evaluate(env: Go2Envs, raw_reward_fn: AnalyticalReward, device: torch.device, actor: TrackingActor | None = None) -> dict:
    env.reset(0)
    caches = (None, actor.action_history.init_cache(1, device = device)) if exists(actor) else None
    ref_q = env.clips[0]['qpos'][:, 7:]
    states, actions, qposes, survived = [], [], [env.datas[0].qpos.copy()], 0

    for t in range(env.horizon):
        if exists(actor):
            act_t, caches = actor.step_policy(as_tensor(env.normalize(), device = device), env.goals(), caches)
            act = act_t.cpu().numpy()
        else:
            act = np.clip((ref_q[t + 1] - Q_DEFAULT) / ACTION_SCALE, -1., 1.)[None]

        next_state, dones, falls = env.step(act)
        states.append(next_state[0])
        actions.append(act[0])
        qposes.append(env.datas[0].qpos.copy())
        survived += int(not falls[0])
        if dones[0]:
            break

    n = len(states)
    states_t, actions_t = as_tensor(np.asarray(states), device = device)[None], as_tensor(np.asarray(actions), device = device)[None]
    ref_d = dict(**unpack_heading_state(env.all_states[1 : n + 1][None]), body_pos = env.all_feet[1 : n + 1][None])
    rewards = raw_reward_fn(states_t, ref_d, actions = actions_t)
    pred = raw_reward_fn.state_transforms[0](states_t)

    ref_qpos, pred_qpos = env.clips[0]['qpos'][:n + 1], np.asarray(qposes)
    yaw_err = np.abs(np.unwrap(quat_yaw_np(pred_qpos[:, 3:7])) - np.unwrap(quat_yaw_np(ref_qpos[:, 3:7])))

    return dict(
        track = float(rewards.sum().item() / (env.horizon * MAX_EVAL_REWARD)),
        surv = float(survived / env.horizon),
        foot_mm = float((pred['body_pos'] - ref_d['body_pos']).norm(dim = -1).mean().item() * 1000.),
        q_rmse = float((pred['joint_pos'] - ref_d['joint_pos']).square().mean().sqrt().item()),
        vel_rmse = float((pred['root_lin_vel'] - ref_d['root_lin_vel']).square().mean().sqrt().item()),
        z_cm = float((pred['root_z'] - ref_d['root_z']).abs().mean().item() * 100.),
        yaw_mae = float(np.degrees(yaw_err).mean()),
        ref_dist = float(np.linalg.norm(ref_qpos[-1, :2] - ref_qpos[0, :2])),
        pred_dist = float(np.linalg.norm(pred_qpos[-1, :2] - pred_qpos[0, :2])),
        qpos = pred_qpos
    )

# plotting & 3-panel comparison video

def add_banner(frames: list[np.ndarray], title: str, subtitle: str, accent: tuple[int, int, int]) -> list[np.ndarray]:
    bar = Image.new('RGB', (frames[0].shape[1], 48), (15, 23, 42))
    draw = ImageDraw.Draw(bar)
    draw.text((10, 6), title, fill = (248, 250, 252), font = ImageFont.load_default(size = 14))
    draw.text((10, 25), subtitle, fill = (203, 213, 225), font = ImageFont.load_default(size = 11))
    draw.rectangle((0, 45, bar.width, 48), fill = accent)
    bar_arr = np.asarray(bar)
    return [np.concatenate((bar_arr, f), axis = 0) for f in frames]

def add_geom(scene: mujoco.MjvScene, geom_type, radius: float, pos: np.ndarray, rgba: tuple[float, ...], p2 = None):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, geom_type, np.array([radius, 0., 0.]), pos.astype(np.float64), np.eye(3).ravel(), np.array(rgba, dtype = np.float32))
    if exists(p2):
        mujoco.mjv_connector(g, geom_type, radius, pos.astype(np.float64), p2.astype(np.float64))
    scene.ngeom += 1

def draw_dog_skeleton(scene: mujoco.MjvScene, pts: np.ndarray):
    pts = pts + np.array([0., 0., FOOT_RADIUS], dtype = np.float32)
    for chain, rgba, rad in DOG_CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            add_geom(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, rad, pts[a], rgba, p2 = pts[b])

    add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, 0.042, pts[5], DOG_CHAINS[0][1])
    for pid, (_, rgba, _) in zip(DOG_PAW_IDS, DOG_CHAINS[2:]):
        add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, 0.024, pts[pid], rgba)

def save_curves(path: Path, title: str, series: list[tuple], hlines = (), foot_ylim: float | None = None):
    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize = (12.0, 4.8), dpi = 150)
    fig.suptitle(title, fontsize = 10.5, fontweight = 'bold')
    single = len(series) == 1

    for name, color, ls, hist, final in series:
        xs, ys_t, ys_f = [r[0] / 1e3 for r in hist], [r[1] for r in hist], [r[2] for r in hist]
        ax_t.plot(xs, ys_t, color = color, linestyle = ls, linewidth = 2., marker = 'o', markersize = 3.2, label = f'{name} ({final["track"]:.3f})')
        ax_f.plot(xs, ys_f, color = color, linestyle = ls, linewidth = 2., marker = 'o', markersize = 3.2, label = f'{name} ({final["foot_mm"]:.1f} mm)')
        if single:
            ax_t.annotate(f'{ys_t[-1]:.3f}', xy = (xs[-1], ys_t[-1]), xytext = (-6, -14), textcoords = 'offset points', ha = 'right', color = color, fontsize = 9, fontweight = 'bold')
            ax_f.annotate(f'{ys_f[-1]:.1f} mm', xy = (xs[-1], ys_f[-1]), xytext = (-6, 8), textcoords = 'offset points', ha = 'right', color = color, fontsize = 9, fontweight = 'bold')

    for tr_h, mm_h, label, color in hlines:
        ax_t.axhline(tr_h, color = color, linestyle = ':', linewidth = 1.4, label = f'{label} ({tr_h:.3f})')
        ax_f.axhline(mm_h, color = color, linestyle = ':', linewidth = 1.4, label = f'{label} ({mm_h:.1f} mm)')

    max_foot = default(foot_ylim, max(r[2] for _, _, _, h, _ in series for r in h) * 1.15)
    ax_t.set(xlabel = 'Environment Steps (×1,000)', ylabel = 'Normalized Tracking Return (0–1)', title = 'Held-Out Dog Locomotion Return', ylim = (0. if single else -0.05, 1.02))
    ax_f.set(xlabel = 'Environment Steps (×1,000)', ylabel = 'Mean 3D Paw Error (mm)', title = 'Held-Out 3D Paw Tracking Error', ylim = (0., max_foot))

    for ax in (ax_t, ax_f):
        ax.grid(True, alpha = 0.25)
        if not single or len(hlines) > 0:
            ax.legend(fontsize = 7.6, loc = 'best')

    fig.tight_layout(rect = (0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)

def save_comparison_video(model: mujoco.MjModel, eval_clip: dict[str, np.ndarray], q_mimic: np.ndarray, final: dict, out_path: Path, render_size = 360, webp_quality = 80):
    n = min(len(q_mimic), len(eval_clip['qpos']))
    dog_pts, q_ref, q_mimic = eval_clip['dog_pts'][:n], eval_clip['qpos'][:n], q_mimic[:n]
    yaws_ref, yaws_mim = np.unwrap(quat_yaw_np(q_ref[:, 3:7])), np.unwrap(quat_yaw_np(q_mimic[:, 3:7]))

    def make_cam(xy: np.ndarray, yaw: float) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.elevation, cam.azimuth = [float(xy[0]), float(xy[1]), 0.25], 1.35, -18.0, float(np.degrees(yaw)) - 125.0
        return cam

    dog_model = mujoco.MjModel.from_xml_string(DOG_SCENE_XML)
    dog_data, dog_ren = mujoco.MjData(dog_model), mujoco.Renderer(dog_model, height = render_size, width = render_size)
    p1_raw = []
    for t in range(n):
        mujoco.mj_forward(dog_model, dog_data)
        dog_ren.update_scene(dog_data, camera = make_cam(0.5 * (dog_pts[t, 0, :2] + dog_pts[t, 3, :2]), yaws_ref[t]))
        draw_dog_skeleton(dog_ren.scene, dog_pts[t])
        p1_raw.append(dog_ren.render())
    dog_ren.close()

    go2_data, go2_ren = mujoco.MjData(model), mujoco.Renderer(model, height = render_size, width = render_size)
    def render_go2(qs: np.ndarray, yaws: np.ndarray) -> list[np.ndarray]:
        out = []
        for q, yaw in zip(qs, yaws):
            go2_data.qpos[:] = q
            mujoco.mj_forward(model, go2_data)
            go2_ren.update_scene(go2_data, camera = make_cam(q[:2], yaw))
            out.append(go2_ren.render())
        return out

    p1 = add_banner(p1_raw, 'Real German Shepherd Mocap', 'Held-out 30 fps 3D dog recording', (22, 163, 74))
    p2 = add_banner(render_go2(q_ref, yaws_ref), 'Kinematic Reference (Unitree Go2)', 'Retargeted via ForwardKinematics', (100, 116, 139))
    p3 = add_banner(render_go2(q_mimic, yaws_mim), 'SuperMimic Policy (Unitree Go2)', f'Track: {final["track"]:.3f} | Surv: {final["surv"]*100:.0f}% | Feet: {final["foot_mm"]:.1f} mm', (37, 99, 235))
    go2_ren.close()

    div = np.full((p1[0].shape[0], 4, 3), (15, 23, 42), dtype = np.uint8)
    combined = [np.concatenate((a, div, b, div, c), axis = 1) for a, b, c in zip(p1, p2, p3)]

    imageio.mimsave(out_path / 'dog_vs_go2_supermimic.mp4', combined, fps = FPS, codec = 'libx264', pixelformat = 'yuv420p', macro_block_size = 1, output_params = ['-movflags', '+faststart'])
    Image.fromarray(combined[0]).save(out_path / 'dog_vs_go2_supermimic.webp', save_all = True, append_images = [Image.fromarray(f) for f in combined[2::2]], duration = int(2000 / FPS), loop = 0, quality = webp_quality)

# training & ablations

def train(
    out_dir = './results_unitree_dog',
    seed = 42,
    num_iterations = 64,
    num_envs = 16,
    env_horizon = 128,
    warmup_steps = 160,
    warmup_action_std = 0.35,
    warmup_wm_updates = 400,
    collect_steps = 24,
    buffer_capacity = 100_000,
    action_noise_start = 0.12,
    action_noise_end = 0.02,
    obs_window = 2,
    action_window = 1,
    goal_window = 8,
    actor_dim_hidden = 256,
    actor_depth = 3,
    wm_dim_hiddens = (512, 512, 512),
    critic_dim_hiddens = (256, 256),
    num_explorative_rollouts = 2,
    dim_explorative_latent = 4,
    explorative_latent_scale_start = 0.25,
    explorative_latent_scale_end = 0.03,
    joint_reward_weight = 1.5,
    foot_reward_weight = 2.0,
    batch_size = 256,
    wm_horizon = 8,
    imag_horizon = 16,
    wm_updates = 24,
    policy_updates = 8,
    critic_updates = 2,
    wm_lr = 6e-4,
    wm_lr_end = 1e-4,
    actor_lr = 4e-4,
    actor_lr_end = 3e-5,
    critic_lr = 3e-4,
    ema_actor_decay = 0.96,
    critic_ema_decay = 0.99,
    discount_factor = 0.99,
    gae_lam = 0.95,
    eval_every = 4,
    use_critic = True,
    train_wm = True,
    save_video = True,
    cache_root: str | None = None
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = select_device()

    out_path = Path(out_dir)
    out_path.mkdir(parents = True, exist_ok = True)
    shared = Path(default(cache_root, out_dir))

    model = build_go2_model(shared / 'cache')
    train_clips, eval_clip = load_or_retarget_clips(model, shared / 'cache', shared / 'go2_dog_mocap.npz')
    mean, std = norm_stats([c['states'] for c in train_clips])
    mean_t, std_t = as_tensor(mean, device = device), as_tensor(std, device = device)

    train_envs = Go2Envs(model, train_clips, mean, std, device, num_envs = num_envs, horizon = env_horizon, goal_window = goal_window)
    eval_env = Go2Envs(model, [eval_clip], mean, std, device, goal_window = goal_window, is_eval = True)

    # models

    active_latent_dim = dim_explorative_latent if num_explorative_rollouts > 1 else 0

    actor = TrackingActor(
        DIM_STATE, NUM_JOINTS, DIM_REF,
        obs_window = obs_window, action_window = action_window, goal_window = goal_window,
        dim_hidden = actor_dim_hidden, depth = actor_depth, dim_latent = active_latent_dim
    ).to(device)
    ema_actor = EMA(actor, beta = ema_actor_decay, include_online_model = False, update_after_step = 0, update_every = 1).to(device)

    world_model = WorldModel(
        dim_state = DIM_STATE, dim_action = NUM_JOINTS, dim_sim_context = 0,
        dim_hiddens = wm_dim_hiddens, activation = SiLU(), normed_mlp = True, pred_terminal = False
    ).to(device)

    critic = Critic(
        dim_state = DIM_STATE, dim_goal = DIM_REF, dim_sim_context = 0,
        dim_hiddens = critic_dim_hiddens, activation = SiLU(), normed_mlp = True
    ).to(device) if use_critic else None

    reward_fn = make_reward(model, mean_t, std_t, joint_weight = joint_reward_weight, foot_weight = foot_reward_weight).to(device)
    raw_reward_fn = make_reward(model).to(device)

    supermimic = SuperMimic(
        actor = actor, world_model = world_model, critic = critic,
        use_ema_critic = use_critic, critic_ema_decay = critic_ema_decay,
        obs_history = actor.obs_history, action_history = actor.action_history, goal_window = actor.goal_window,
        reward_fn = reward_fn, termination_fn = make_termination(mean_t, std_t), discount_factor = discount_factor,
        num_explorative_rollouts = num_explorative_rollouts,
        dim_explorative_latent = active_latent_dim,
        explorative_latent_scale = explorative_latent_scale_start if active_latent_dim > 0 else 0.
    ).to(device)

    # optimizers & replay buffer

    wm_opt = AdamW(world_model.parameters(), lr = wm_lr, weight_decay = 1e-5)
    wm_sched = CosineAnnealingLR(wm_opt, T_max = max(1, warmup_wm_updates + num_iterations * wm_updates), eta_min = wm_lr_end)
    actor_opt = AdamW(actor.parameters(), lr = actor_lr, betas = (0.9, 0.95), weight_decay = 1e-5)
    actor_sched = CosineAnnealingLR(actor_opt, T_max = max(1, num_iterations * policy_updates), eta_min = actor_lr_end)
    critic_opt = AdamW(critic.parameters(), lr = critic_lr, weight_decay = 1e-5) if use_critic else None

    buffer = StreamReplayBuffer(
        num_streams = train_envs.num_envs, capacity = buffer_capacity,
        fields = dict(state = ('float', DIM_STATE), action = ('float', NUM_JOINTS), next_state = ('float', DIM_STATE), frame = 'int')
    )

    seed_horizon = max(obs_window, action_window + 1)
    caches = (None, actor.action_history.init_cache(train_envs.num_envs, device = device))
    total_steps, history, best_track, start_time = 0, [], -float('inf'), time.time()

    def collect(steps: int, noise = 0., random_std = 0.):
        nonlocal caches, total_steps
        with temp_eval(actor), torch.no_grad():
            for _ in range(steps):
                norm_s, gframes = train_envs.normalize(), train_envs.global_frames()
                act, caches = actor.step_policy(as_tensor(norm_s, device = device), train_envs.goals(), caches, noise = noise, random_std = random_std)
                act_np = act.cpu().numpy()
                next_s, dones, _ = train_envs.step(act_np)
                norm_ns = train_envs.normalize(next_s)

                for i in range(train_envs.num_envs):
                    buffer.store_step(i, state = norm_s[i], action = act_np[i], next_state = norm_ns[i], frame = int(gframes[i]))
                    if dones[i]:
                        buffer.end_episode(i)

                total_steps += train_envs.num_envs
                if dones.any():
                    done_t = as_tensor(dones, device = device)
                    caches = (actor.obs_history.reset(caches[0], done_t), actor.action_history.reset(caches[1], done_t))

    def fit_world_model(updates: int) -> float:
        if not train_wm:
            return float('nan')
        world_model.train()
        for _ in range(updates):
            batch = buffer.sample(batch_size, horizon = wm_horizon, to_named_tuple = ('state', 'action', 'next_state'), device = device)
            loss = world_model.rollout(batch.state[:, 0], batch.action, batch.next_state)[0] / wm_horizon
            step_optimizer(wm_opt, loss, world_model.parameters())
            wm_sched.step()
        return float(loss.item())

    def log_eval(iteration: int, wm_loss: float) -> dict:
        with temp_eval(ema_actor.ema_model):
            m = evaluate(eval_env, raw_reward_fn, device, actor = ema_actor.ema_model)
        history.append((total_steps, m['track'], m['foot_mm'], m['q_rmse'], m['vel_rmse'], m['surv']))
        print(
            f'[Go2] iter {iteration:2d} | steps {total_steps:5d} | wm {wm_loss:.5f} | '
            f'track {m["track"]:.3f} | surv {m["surv"]*100:4.0f}% | feet {m["foot_mm"]:5.1f} mm | '
            f'vel {m["vel_rmse"]:.2f} m/s | yaw {m["yaw_mae"]:4.1f}° | dist {m["pred_dist"]:.2f}/{m["ref_dist"]:.2f} m | {time.time() - start_time:.1f}s',
            flush = True
        )
        return m

    # warmup & training loop

    ol = {k: v for k, v in evaluate(eval_env, raw_reward_fn, device).items() if k != 'qpos'}
    print(f'device {device} | train {len(train_envs.all_states)}f ({len(train_clips)} clips) | eval {len(eval_clip["qpos"])}f ({EVAL_CLIP[0]})', flush = True)
    print(f'[open-loop PD] track {ol["track"]:.3f} | surv {ol["surv"]*100:.0f}% | feet {ol["foot_mm"]:.1f} mm | dist {ol["pred_dist"]:.2f}/{ol["ref_dist"]:.2f} m', flush = True)

    collect(warmup_steps, random_std = warmup_action_std)
    log_eval(0, fit_world_model(warmup_wm_updates))

    ckpt_path = out_path / 'supermimic_go2_actor.pt'

    for iteration in range(1, num_iterations + 1):
        frac = (iteration - 1) / max(1, num_iterations - 1)
        collect(collect_steps, noise = action_noise_start + (action_noise_end - action_noise_start) * frac)
        wm_loss = fit_world_model(wm_updates)

        actor.train()
        latent_scale = cosine_anneal(explorative_latent_scale_start, explorative_latent_scale_end, frac) if active_latent_dim > 0 else 0.

        for _ in range(policy_updates):
            seed_batch = buffer.sample(batch_size, horizon = seed_horizon, to_named_tuple = ('state', 'action', 'frame'), device = device)
            last_frames = seed_batch.frame[:, -1].long()
            starts = torch.minimum(last_frames + 1, train_envs.clip_ends[last_frames])

            goals = gather_clip_seq(train_envs.all_refs_norm, train_envs.clip_ends, starts, imag_horizon + goal_window)
            refs = unpack_heading_state(gather_clip_seq(train_envs.all_states, train_envs.clip_ends, starts, imag_horizon))
            refs['body_pos'] = gather_clip_seq(train_envs.all_feet, train_envs.clip_ends, starts, imag_horizon)
            critic_goals = goals[:, :imag_horizon + 1]

            rollout = supermimic(
                seed_batch.state[:, -1], actor_goals = goals, references = refs,
                past_obs = seed_batch.state[:, -obs_window:-1] if obs_window > 1 else None,
                past_actions = seed_batch.action[:, -(action_window + 1):-1] if action_window > 0 else None,
                num_rollout_timesteps = imag_horizon, explorative_latent_scale = latent_scale
            )

            loss_kwargs = dict(final_state = rollout.states[:, -1], final_goal = critic_goals[:, -1]) if use_critic else dict()
            actor_loss = supermimic.actor_loss(rollout.rewards, pred_terminal = rollout.terminals, **loss_kwargs)
            step_optimizer(actor_opt, actor_loss, actor.parameters(), clip_value = 0.1)
            actor_sched.step()
            ema_actor.update()

            if use_critic:
                for _ in range(critic_updates):
                    critic_loss = supermimic.critic_loss(rollout.states, rollout.rewards, pred_terminal = rollout.terminals, goals = critic_goals, lam = gae_lam)
                    step_optimizer(critic_opt, critic_loss, critic.parameters())
                    supermimic.update_ema()

        if iteration % eval_every == 0 or iteration == num_iterations:
            m = log_eval(iteration, wm_loss)
            if m['track'] > best_track:
                best_track = m['track']
                torch.save(ema_actor.ema_model.state_dict(), ckpt_path)

    if ckpt_path.exists():
        ema_actor.ema_model.load_state_dict(torch.load(ckpt_path, map_location = device, weights_only = True))

    with temp_eval(ema_actor.ema_model):
        final = evaluate(eval_env, raw_reward_fn, device, actor = ema_actor.ema_model)

    if save_video:
        save_curves(
            out_path / 'go2_convergence.png',
            'Unitree Go2 (MuJoCo Contact Physics): SuperMimic Tracking Held-Out Real Dog Motion',
            [('SuperMimic', '#1d4ed8', '-', history, final)],
            hlines = [(ol['track'], ol['foot_mm'], 'Open-Loop PD Replay', '#64748b')]
        )
        save_comparison_video(model, eval_clip, final['qpos'], final, out_path)

    print(f'[final] track={final["track"]:.3f} | surv={final["surv"]*100:.0f}% | feet={final["foot_mm"]:.1f} mm | q_rmse={final["q_rmse"]:.3f} rad | vel={final["vel_rmse"]:.2f} m/s -> {out_path}', flush = True)
    return dict(history = history, open_loop = ol, final = {k: v for k, v in final.items() if k != 'qpos'})

def run_ablations(out_path: Path, seed = 42, num_iterations = 48):
    variants = (
        ('Full SuperMimic',         '#1d4ed8', '-',  dict()),
        ('1-Step BPTT (H=1, no V)', '#7c3aed', '--', dict(imag_horizon = 1, use_critic = False)),
        ('Myopic Goal (W=1)',       '#d97706', '-',  dict(goal_window = 1)),
        ('Neg. Ctrl: Untrained WM', '#dc2626', ':',  dict(train_wm = False)),
    )

    results = {}
    for name, _, _, kwargs in variants:
        print(f'\n=== Ablation: {name} ===', flush = True)
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        results[name] = train(out_dir = str(out_path / 'ablation' / slug), seed = seed, num_iterations = num_iterations, save_video = False, cache_root = str(out_path), **kwargs)

    device = select_device()
    model = build_go2_model(out_path / 'cache')
    train_clips, eval_clip = load_or_retarget_clips(model, out_path / 'cache', out_path / 'go2_dog_mocap.npz')
    mean, std = norm_stats([c['states'] for c in train_clips])

    rev_env = Go2Envs(model, [eval_clip], mean, std, device, is_eval = True)
    rev_env.goal_windows[0] = rev_env.goal_windows[0].flip(0)
    actor = TrackingActor(DIM_STATE, NUM_JOINTS, DIM_REF).to(device)
    actor.load_state_dict(torch.load(out_path / 'ablation' / 'full_supermimic' / 'supermimic_go2_actor.pt', map_location = device, weights_only = True))
    with temp_eval(actor):
        rev = {k: v for k, v in evaluate(rev_env, make_reward(model).to(device), device, actor = actor).items() if k != 'qpos'}

    ol = results['Full SuperMimic']['open_loop']
    results['Neg. Ctrl: Reversed Eval Goals'] = dict(final = rev, history = [])
    results['Baseline: Open-Loop PD Replay'] = dict(final = ol, history = [])
    (out_path / 'ablation_results.json').write_text(json.dumps(results, indent = 2))

    save_curves(
        out_path / 'go2_ablation.png',
        'Unitree Go2 SuperMimic Ablations & Negative Controls (Held-Out Real Dog Motion)',
        [(name, color, ls, results[name]['history'], results[name]['final']) for name, color, ls, _ in variants],
        hlines = [(ol['track'], ol['foot_mm'], 'Baseline: Open-Loop PD', '#475569'), (rev['track'], rev['foot_mm'], 'Neg. Ctrl: Reversed Goals', '#111827')],
        foot_ylim = 160.
    )

def main(out_dir = './results_unitree_dog', seed = 42, num_iterations: int | None = None, ablate = False, **kwargs):
    if ablate:
        return run_ablations(Path(out_dir), seed = seed, num_iterations = default(num_iterations, 48))
    train(out_dir = out_dir, seed = seed, num_iterations = default(num_iterations, 64), **kwargs)

if __name__ == '__main__':
    fire.Fire(main)
