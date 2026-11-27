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
import pickle
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
    HistoryWindow, JointLimitPenalty, SuperMimic, TrackingTerm, WorldModel
)

# constants

RUKA_RAW = 'https://raw.githubusercontent.com/ruka-hand-v2/RUKA-v2/main/assets/'
HUMAN_RAW = 'https://raw.githubusercontent.com/dexsuite/dex-retargeting/main/example/'

LOCKED_JOINTS = ('base_pitch', 'wrist_yaw', 'base_yaw')
TIP_BODIES = ('thumb_actual_tip', 'index_actual_tip', 'middle_actual_tip', 'ring_actual_tip', 'pinky_actual_tip')

FINGER_SPANS = (
    ('mcp',   'index_actual_tip',   5,  8),
    ('mcp_2', 'middle_actual_tip',  9, 12),
    ('mcp_3', 'ring_actual_tip',   13, 16),
    ('mcp_4', 'pinky_actual_tip',  17, 20)
)

RETARGET_BONES = (
    ('thumb___joint_1', 'thumb___joint_2',    2,  3),
    ('thumb___joint_2', 'thumb___joint_3',    3,  4),
    ('thumb___joint_1', 'thumb_actual_tip',   2,  4),
    ('mcp',             'pip',                5,  6),
    ('pip',             'finger___joint_3',   6,  7),
    ('mcp_2',           'pip_2',              9, 10),
    ('pip_2',           'finger___joint_3_2', 10, 11),
    ('mcp_3',           'pip_3',             13, 14),
    ('pip_3',           'finger___joint_3_3', 14, 15),
    ('mcp_4',           'pinky___joint_2',   17, 18),
    ('pinky___joint_2', 'pinky___joint_3',   18, 19),
    *FINGER_SPANS
)

FPS = 30
SUBSTEPS = 8
MAX_EVAL_REWARD = 5.0

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
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def fetch(url: str, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents = True, exist_ok = True)
        urllib.request.urlretrieve(url, dst)
    return dst

def palm_basis(up: np.ndarray, across: np.ndarray) -> np.ndarray:
    norm = lambda v: v / (np.linalg.norm(v, axis = -1, keepdims = True) + 1e-8)
    up = norm(up)
    across = norm(across - (across * up).sum(axis = -1, keepdims = True) * up)
    return np.stack((across, np.cross(across, up), up), axis = -1)

def norm_stats(qpos: np.ndarray, qvel: np.ndarray, min_qpos_std = 0.15, min_qvel_std = 1.0) -> tuple[np.ndarray, np.ndarray]:
    mean = np.concatenate((qpos, qvel), axis = -1).mean(axis = 0).astype(np.float32)
    std = np.concatenate((
        np.maximum(qpos.std(0), min_qpos_std),
        np.maximum(qvel.std(0), min_qvel_std)
    )).astype(np.float32)
    return mean, std

def step_optimizer(opt, loss: Tensor, params, max_norm = 1., clip_value = None):
    params = list(params)
    opt.zero_grad()
    loss.backward()
    clip_grad_norm_(params, max_norm)
    if exists(clip_value):
        clip_grad_value_(params, clip_value)
    opt.step()

def gather_seq(seq: Tensor, starts: Tensor, length: int) -> Tensor:
    frames = einx.add('b, l -> b l', starts, torch.arange(length, device = seq.device)).clamp(max = len(seq) - 1)
    return seq[frames]

# RUKA-v2 URDF/STL download, MuJoCo compilation & differentiable FK retargeting

def build_ruka_model(
    cache_dir: Path,
    actuator_gear = 0.32,
    joint_damping = 0.015,
    joint_armature = 0.002
) -> tuple[mujoco.MjModel, Path, Path]:
    ruka_dir = cache_dir / 'ruka_v2'
    human_dir = cache_dir / 'human'

    urdf_txt = fetch(RUKA_RAW + 'robot.urdf', ruka_dir / 'robot.urdf').read_text()
    for mesh in sorted({el.attrib['filename'].split('/')[-1] for el in ET.fromstring(urdf_txt).findall('.//mesh')}):
        fetch(RUKA_RAW + mesh, ruka_dir / mesh)

    pkl_path = fetch(HUMAN_RAW + 'profiling/human_joint_right.pkl', human_dir / 'human_joint_right.pkl')
    vid_path = fetch(HUMAN_RAW + 'vector_retargeting/data/human_hand_video.mp4', human_dir / 'human_hand_video.mp4')

    mj_compiler = '<mujoco><compiler meshdir="." discardvisual="false" fusestatic="false"/></mujoco></robot>'
    mj_urdf = ruka_dir / 'robot_mj.urdf'
    mj_urdf.write_text(urdf_txt.replace('package://assets/', '').replace('</robot>', mj_compiler))

    spec = mujoco.MjSpec.from_file(str(mj_urdf))

    for joint in list(spec.joints):
        if joint.name in LOCKED_JOINTS:
            spec.delete(joint)

    for geom in list(spec.geoms):
        if geom.group == 0:
            spec.delete(geom)
        else:
            geom.contype = 0
            geom.conaffinity = 0
            geom.group = 0

    spec.option.timestep = 1.0 / (FPS * SUBSTEPS)
    spec.option.gravity = [0.0, 0.0, -9.81]

    for joint in spec.joints:
        spec.add_actuator(
            name = f'act_{joint.name}', target = joint.name,
            trntype = mujoco.mjtTrn.mjTRN_JOINT, gear = [actuator_gear, 0, 0, 0, 0, 0],
            ctrllimited = True, ctrlrange = [-1.0, 1.0]
        )

    spec.worldbody.add_light(pos = [0.2, -0.6, 0.5], dir = [-0.2, 0.8, -0.5], diffuse = [0.85, 0.85, 0.85], specular = [0.25, 0.25, 0.25], castshadow = False)
    spec.worldbody.add_light(pos = [-0.3, -0.2, 0.6], dir = [0.4, 0.3, -0.7], diffuse = [0.4, 0.42, 0.46], castshadow = False)
    spec.add_texture(
        name = 'sky', type = mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1 = [0.93, 0.95, 0.98], rgb2 = [0.80, 0.84, 0.90], width = 64, height = 64
    )

    model = spec.compile()
    model.dof_damping[:] = joint_damping
    model.dof_armature[:] = joint_armature
    return model, pkl_path, vid_path

def retarget_human_hand(
    model: mujoco.MjModel,
    pkl_path: Path,
    cache_path: Path,
    retarget_steps = 300,
    retarget_lr = 0.05,
    retarget_smooth_weight = 0.12
) -> dict[str, np.ndarray]:
    if cache_path.exists():
        return dict(np.load(cache_path))

    pts = np.asarray(pickle.loads(pkl_path.read_bytes()), dtype = np.float32)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_pos = lambda name: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]

    # align real human open-hand palm frame to RUKA-v2 rest frame

    open_idx = int(np.argmax(sum(np.linalg.norm(pts[:, ht] - pts[:, hm], axis = -1) for _, _, hm, ht in FINGER_SPANS)))
    R_ruka = palm_basis(np.mean([body_pos(rt) - body_pos(rm) for rm, rt, _, _ in FINGER_SPANS], axis = 0), body_pos('mcp') - body_pos('mcp_4'))
    R_hum = palm_basis(np.mean([pts[open_idx, ht] - pts[open_idx, hm] for _, _, hm, ht in FINGER_SPANS], axis = 0), pts[open_idx, 5] - pts[open_idx, 17])
    pts_ruka = pts @ (R_ruka @ R_hum.T).T

    # optimize 18 joint trajectories via differentiable ForwardKinematics

    tracked = sorted({body for a, b, _, _ in RETARGET_BONES for body in (a, b)})
    b2i = {name: i for i, name in enumerate(tracked)}

    orig_idx = [b2i[a] for a, _, _, _ in RETARGET_BONES]
    task_idx = [b2i[b] for _, b, _, _ in RETARGET_BONES]
    hum_orig = [a for _, _, a, _ in RETARGET_BONES]
    hum_task = [b for _, _, _, b in RETARGET_BONES]

    fk = ForwardKinematics.from_mujoco(model, tracked_body_names = tracked)
    target_dirs = F.normalize(as_tensor(pts_ruka[:, hum_task] - pts_ruka[:, hum_orig], dtype = torch.float32), dim = -1)
    low = as_tensor(model.jnt_range[:, 0], dtype = torch.float32)
    high = as_tensor(model.jnt_range[:, 1], dtype = torch.float32)

    q = torch.zeros((len(pts), model.nq), requires_grad = True)
    opt = torch.optim.Adam([q], lr = retarget_lr)

    for _ in range(retarget_steps):
        pos = fk(q)['body_pos']
        pred_dirs = F.normalize(pos[:, task_idx] - pos[:, orig_idx], dim = -1)
        loss = (1. - (pred_dirs * target_dirs).sum(dim = -1)).mean() + (q[1:] - q[:-1]).square().mean() * retarget_smooth_weight
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            q.clamp_(low, high)

    q_ref = q.detach().numpy().astype(np.float32)
    qvel_ref = np.gradient(q_ref, 1.0 / FPS, axis = 0).astype(np.float32)
    tips_ref = ForwardKinematics.from_mujoco(model, tracked_body_names = TIP_BODIES)(as_tensor(q_ref))['body_pos'].numpy().astype(np.float32)

    np.savez(cache_path, qpos = q_ref, qvel = qvel_ref, tips = tips_ref)
    print(f'[retarget] {len(q_ref)} real human hand frames -> RUKA-v2 {model.nq}-DOF', flush = True)
    return dict(qpos = q_ref, qvel = qvel_ref, tips = tips_ref)

# parallel MuJoCo RUKA-v2 environments with reference state initialization

class RukaEnvs:
    def __init__(
        self,
        model: mujoco.MjModel,
        qpos: np.ndarray,
        qvel: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        device: torch.device,
        num_envs = 16,
        horizon = 96,
        goal_window = 8,
        rsi_pos_noise = 0.03,
        rsi_vel_scale_min = 0.5,
        is_eval = False
    ):
        self.model = model
        self.qpos = qpos
        self.qvel = qvel
        self.mean = mean
        self.std = std
        self.device = device
        self.rsi_pos_noise = rsi_pos_noise
        self.rsi_vel_scale_min = rsi_vel_scale_min
        self.is_eval = is_eval

        self.num_envs = 1 if is_eval else num_envs
        self.total_frames = len(qpos)
        self.horizon = self.total_frames - 1 if is_eval else min(horizon, self.total_frames - 2)
        self.max_start = max(1, self.total_frames - self.horizon - 1)

        self.refs = as_tensor(qpos, device = device)
        self.goal_windows = HistoryWindow(goal_window, model.nq, future = True, pad_mode = 'replicate')(self.refs[None, 1:])[0]
        self.datas = [mujoco.MjData(model) for _ in range(self.num_envs)]
        self.frames = np.zeros(self.num_envs, dtype = np.int64)
        self.steps = np.zeros(self.num_envs, dtype = np.int64)
        self.states = np.zeros((self.num_envs, 2 * model.nq), dtype = np.float32)

        for i in range(self.num_envs):
            self.reset(i)

    def reset(self, index: int):
        frame = 0 if self.is_eval else int(np.random.randint(self.max_start))
        data = self.datas[index]
        mujoco.mj_resetData(self.model, data)

        noise = 0. if self.is_eval else np.random.normal(0., self.rsi_pos_noise, self.model.nq)
        vel_scale = 1. if self.is_eval else np.random.uniform(self.rsi_vel_scale_min, 1.0)
        data.qpos[:] = np.clip(self.qpos[frame] + noise, self.model.jnt_range[:, 0], self.model.jnt_range[:, 1])
        data.qvel[:] = self.qvel[frame] * vel_scale
        mujoco.mj_forward(self.model, data)

        self.frames[index] = frame
        self.steps[index] = 0
        self.states[index] = np.concatenate((data.qpos, data.qvel))

    def normalize(self, states: np.ndarray | None = None) -> np.ndarray:
        return (default(states, self.states) - self.mean) / self.std

    def goals(self) -> Tensor:
        return self.goal_windows[as_tensor(self.frames, device = self.device)]

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        next_states = np.empty_like(self.states)
        dones = np.zeros(self.num_envs, dtype = bool)

        for i, data in enumerate(self.datas):
            data.ctrl[:] = np.clip(actions[i], -1., 1.)
            mujoco.mj_step(self.model, data, nstep = SUBSTEPS)
            next_states[i] = np.concatenate((data.qpos, data.qvel))
            self.frames[i] += 1
            self.steps[i] += 1

            if self.steps[i] >= self.horizon or self.frames[i] >= self.total_frames - 1:
                dones[i] = True
                if not self.is_eval:
                    self.reset(i)
            else:
                self.states[i] = next_states[i]

        return next_states, dones

# actor, analytical FK reward & evaluation

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

    def step_policy(self, state: Tensor, goals: Tensor, caches: tuple, noise = 0., random = False):
        obs_cache, act_cache = caches
        obs, obs_cache = self.obs_history.step(state, obs_cache)
        past_act = self.action_history.read(act_cache)

        if random:
            act = torch.empty((len(state), self.dim_action), device = state.device).uniform_(-1., 1.)
        else:
            features = [rearrange(t, 'b w d -> b (w d)') for t in (obs, past_act, goals)]
            act = self(safe_cat(features, dim = -1))
            if noise > 0.:
                act = (act + torch.randn_like(act) * noise).clamp(-1., 1.)

        return act, (obs_cache, self.action_history.push(act, act_cache))

def make_reward(
    model: mujoco.MjModel,
    mean: Tensor | None = None,
    std: Tensor | None = None,
    joint_std = 0.25,
    joint_weight = 2.0,
    tip_std = 0.018,
    tip_weight = 3.0,
    joint_limit_margin = 0.02,
    joint_limit_weight = -0.5,
    action_smoothness_weight = -0.01
) -> AnalyticalReward:
    nq = model.nq
    fk_tips = ForwardKinematics.from_mujoco(model, tracked_body_names = TIP_BODIES)

    def state_transform(state: Tensor) -> dict[str, Tensor]:
        raw = state * std.to(state) + mean.to(state) if exists(mean) and exists(std) else state
        q = raw[..., :nq]
        return dict(joint_pos = q, body_pos = fk_tips(q)['body_pos'])

    terms = dict(
        joint_pos = TrackingTerm('joint_pos', std = joint_std, weight = joint_weight, dim_weights = torch.full((nq,), 1. / nq)),
        joint_limit = JointLimitPenalty(
            model.jnt_range[:, 0] - joint_limit_margin,
            model.jnt_range[:, 1] + joint_limit_margin,
            key = 'joint_pos',
            weight = joint_limit_weight
        ),
        action_smoothness = ActionSmoothnessTerm(weight = action_smoothness_weight)
    )

    if tip_weight > 0.:
        terms['fingertips'] = TrackingTerm('body_pos', std = tip_std, weight = tip_weight, reduce_items = True)

    return AnalyticalReward(terms, state_transform = state_transform)

@torch.no_grad()
def evaluate(env: RukaEnvs, actor: TrackingActor, raw_reward_fn: AnalyticalReward, tips_ref: Tensor, device: torch.device) -> dict:
    env.reset(0)
    caches = (None, actor.action_history.init_cache(1, device = device))
    states, actions = [], []

    for _ in range(env.horizon):
        act, caches = actor.step_policy(as_tensor(env.normalize(), device = device), env.goals(), caches)
        next_state, _ = env.step(act.cpu().numpy())
        states.append(next_state[0])
        actions.append(act[0].cpu().numpy())

    states_arr = np.asarray(states)
    states_t = as_tensor(states_arr, device = device)[None]
    actions_t = as_tensor(np.asarray(actions), device = device)[None]
    ref_q = env.refs[1 : env.horizon + 1][None]
    ref_tips = tips_ref[1 : env.horizon + 1][None]

    rewards = raw_reward_fn(states_t, dict(joint_pos = ref_q, body_pos = ref_tips), actions = actions_t)
    pred_tips = raw_reward_fn.state_transforms[0](states_t)['body_pos']

    return dict(
        track = float(rewards.mean().item() / MAX_EVAL_REWARD),
        q_rmse = float((states_t[..., :env.model.nq] - ref_q).square().mean().sqrt().item()),
        tip_mm = float((pred_tips - ref_tips).norm(dim = -1).mean().item() * 1000.),
        qpos = states_arr[:, :env.model.nq]
    )

# plotting & 3-panel comparison video

def add_banner(frames: list[np.ndarray], title: str, subtitle: str, accent: tuple[int, int, int]) -> list[np.ndarray]:
    bar = Image.new('RGB', (frames[0].shape[1], 48), (15, 23, 42))
    draw = ImageDraw.Draw(bar)
    draw.text((10, 6), title, fill = (248, 250, 252), font = ImageFont.load_default(size = 14))
    draw.text((10, 25), subtitle, fill = (203, 213, 225), font = ImageFont.load_default(size = 11))
    draw.rectangle((0, 45, bar.width, 48), fill = accent)
    bar_arr = np.asarray(bar)
    return [np.concatenate((bar_arr, frame), axis = 0) for frame in frames]

def save_curves(
    path: Path,
    title: str,
    series: list[tuple],
    hlines: tuple[float, float, str] | None = None,
    tip_ylim: float | None = None
):
    fig, (ax_track, ax_tips) = plt.subplots(1, 2, figsize = (12.0, 4.8), dpi = 150)
    fig.suptitle(title, fontsize = 10.5, fontweight = 'bold')
    annotate = len(series) == 1

    for name, color, ls, history, final in series:
        steps_k = [row[0] / 1e3 for row in history]
        tracks = [row[1] for row in history]
        tip_mms = [row[2] for row in history]

        ax_track.plot(steps_k, tracks, color = color, linestyle = ls, linewidth = 2.0, marker = 'o', markersize = 3.2, label = f'{name} ({final["track"]:.3f})')
        ax_tips.plot(steps_k, tip_mms, color = color, linestyle = ls, linewidth = 2.0, marker = 'o', markersize = 3.2, label = f'{name} ({final["tip_mm"]:.1f} mm)')

        if annotate:
            ax_track.annotate(f'{tracks[-1]:.3f}', xy = (steps_k[-1], tracks[-1]), xytext = (-6, -14), textcoords = 'offset points', ha = 'right', color = color, fontsize = 9, fontweight = 'bold')
            ax_tips.annotate(f'{tip_mms[-1]:.1f} mm', xy = (steps_k[-1], tip_mms[-1]), xytext = (-6, 8), textcoords = 'offset points', ha = 'right', color = color, fontsize = 9, fontweight = 'bold')

    if exists(hlines):
        tr_h, mm_h, label = hlines
        ax_track.axhline(tr_h, color = '#111827', linestyle = ':', linewidth = 1.4, label = f'{label} ({tr_h:.3f})')
        ax_tips.axhline(mm_h, color = '#111827', linestyle = ':', linewidth = 1.4, label = f'{label} ({mm_h:.1f} mm)')

    max_tip = default(tip_ylim, max(row[2] for _, _, _, h, _ in series for row in h) * 1.12)
    ax_track.set(xlabel = 'Environment Steps (×1,000)', ylabel = 'Normalized Tracking Return (0–1)', title = 'Held-Out Pose + 3D Fingertip Return', ylim = (-0.05 if not annotate else 0., 1.06))
    ax_tips.set(xlabel = 'Environment Steps (×1,000)', ylabel = 'Mean 3D Fingertip Error (mm)', title = 'Held-Out 3D Fingertip Error', ylim = (0., max_tip))

    for ax in (ax_track, ax_tips):
        ax.grid(True, alpha = 0.25)
        if not annotate:
            ax.legend(fontsize = 7.6, loc = 'best')

    fig.tight_layout(rect = (0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)

def save_comparison_video(
    model: mujoco.MjModel,
    vid_path: Path,
    eval_start: int,
    q_ref: np.ndarray,
    q_mimic: np.ndarray,
    final: dict,
    out_path: Path,
    render_size = 360,
    webp_quality = 80
):
    reader = imageio.get_reader(vid_path)
    vid_frames = [reader.get_data(i) for i in range(eval_start + 1, eval_start + 1 + len(q_mimic))]
    reader.close()

    renderer = mujoco.Renderer(model, height = render_size, width = render_size)
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.015, -0.005, 0.125]
    cam.distance = 0.34
    cam.elevation = -8
    cam.azimuth = 90

    def render_seq(qs: np.ndarray) -> list[np.ndarray]:
        frames = []
        for q in qs:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera = cam)
            frames.append(renderer.render())
        return frames

    human_crops = [np.asarray(Image.fromarray(f[:, 260:980]).resize((render_size, render_size))) for f in vid_frames]
    p1 = add_banner(human_crops, 'Actual Human Hand', 'Held-out 30 fps real video', (22, 163, 74))
    p2 = add_banner(render_seq(q_ref[1 : 1 + len(q_mimic)]), 'Kinematic Reference (RUKA-v2)', 'Retargeted via ForwardKinematics', (100, 116, 139))
    p3 = add_banner(render_seq(q_mimic), 'SuperMimic Policy (MuJoCo Torque)', f'Track: {final["track"]:.3f} | Tips: {final["tip_mm"]:.1f} mm | RMSE: {final["q_rmse"]:.3f} rad', (37, 99, 235))
    renderer.close()

    divider = np.full((p1[0].shape[0], 4, 3), (15, 23, 42), dtype = np.uint8)
    combined = [np.concatenate((a, divider, b, divider, c), axis = 1) for a, b, c in zip(p1, p2, p3)]

    imageio.mimsave(
        out_path / 'human_vs_ruka_supermimic.mp4', combined, fps = FPS,
        codec = 'libx264', pixelformat = 'yuv420p', macro_block_size = 1, output_params = ['-movflags', '+faststart']
    )
    Image.fromarray(combined[0]).save(
        out_path / 'human_vs_ruka_supermimic.webp', save_all = True,
        append_images = [Image.fromarray(f) for f in combined[2::2]],
        duration = int(2000 / FPS), loop = 0, quality = webp_quality
    )

# training & ablations

def train(
    out_dir = './results_ruka_hand',
    seed = 42,
    num_iterations = 60,
    split_frac = 0.75,
    # physics & retargeting
    actuator_gear = 0.32,
    joint_damping = 0.015,
    joint_armature = 0.002,
    retarget_steps = 300,
    retarget_lr = 0.05,
    retarget_smooth_weight = 0.12,
    # env & collection
    num_envs = 16,
    env_horizon = 96,
    rsi_pos_noise = 0.03,
    rsi_vel_scale_min = 0.5,
    warmup_steps = 128,
    warmup_wm_updates = 300,
    collect_steps = 16,
    buffer_capacity = 64_000,
    action_noise_start = 0.08,
    action_noise_end = 0.02,
    # model architecture
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
    # reward terms
    joint_reward_std = 0.25,
    joint_reward_weight = 2.0,
    tip_reward_std = 0.018,
    tip_reward_weight = 3.0,
    joint_limit_margin = 0.02,
    joint_limit_weight = -0.5,
    action_smoothness_weight = -0.01,
    # optimization & RL
    batch_size = 256,
    wm_horizon = 8,
    wm_rollout_loss_decay = 0.8,
    wm_state_noise_scale = 0.005,
    imag_horizon = 16,
    wm_updates = 20,
    policy_updates = 6,
    critic_updates = 4,
    wm_lr = 6e-4,
    wm_lr_end = 1e-4,
    actor_lr = 4e-4,
    actor_lr_end = 3e-5,
    critic_lr = 3e-4,
    weight_decay = 1e-5,
    grad_clip_norm = 1.0,
    actor_grad_clip_value = 0.1,
    ema_actor_decay = 0.96,
    critic_ema_decay = 0.99,
    discount_factor = 0.99,
    gae_lam = 0.95,
    eval_every = 4,
    # switches
    use_critic = True,
    train_wm = True,
    shuffle_refs = False,
    save_video = True,
    cache_root: str | None = None
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = select_device()

    out_path = Path(out_dir)
    out_path.mkdir(parents = True, exist_ok = True)
    shared = Path(default(cache_root, out_dir))

    model, pkl_path, vid_path = build_ruka_model(
        shared / 'cache',
        actuator_gear = actuator_gear,
        joint_damping = joint_damping,
        joint_armature = joint_armature
    )
    mocap = retarget_human_hand(
        model,
        pkl_path,
        shared / 'ruka_human_mocap.npz',
        retarget_steps = retarget_steps,
        retarget_lr = retarget_lr,
        retarget_smooth_weight = retarget_smooth_weight
    )

    nq = model.nq
    dim_state = 2 * nq
    split = int(len(mocap['qpos']) * split_frac)
    mean, std = norm_stats(mocap['qpos'][:split], mocap['qvel'][:split])

    train_envs = RukaEnvs(
        model, mocap['qpos'][:split], mocap['qvel'][:split], mean, std, device,
        num_envs = num_envs, horizon = env_horizon, goal_window = goal_window,
        rsi_pos_noise = rsi_pos_noise, rsi_vel_scale_min = rsi_vel_scale_min
    )
    eval_env = RukaEnvs(
        model, mocap['qpos'][split:], mocap['qvel'][split:], mean, std, device,
        goal_window = goal_window, is_eval = True
    )
    train_tips = as_tensor(mocap['tips'][:split], device = device)
    eval_tips = as_tensor(mocap['tips'][split:], device = device)

    # models

    active_latent_dim = dim_explorative_latent if num_explorative_rollouts > 1 else 0

    actor = TrackingActor(
        dim_state, nq, nq,
        obs_window = obs_window, action_window = action_window, goal_window = goal_window,
        dim_hidden = actor_dim_hidden, depth = actor_depth, dim_latent = active_latent_dim
    ).to(device)
    ema_actor = EMA(actor, beta = ema_actor_decay, include_online_model = False, update_after_step = 0, update_every = 1).to(device)

    world_model = WorldModel(
        dim_state = dim_state, dim_action = nq, dim_sim_context = 0,
        dim_hiddens = wm_dim_hiddens, activation = SiLU(), normed_mlp = True, pred_terminal = False,
        rollout_loss_decay = wm_rollout_loss_decay,
        state_noise_scale = wm_state_noise_scale
    ).to(device)

    critic = Critic(
        dim_state = dim_state, dim_goal = nq, dim_sim_context = 0,
        dim_hiddens = critic_dim_hiddens, activation = SiLU(), normed_mlp = True
    ).to(device) if use_critic else None

    reward_fn = make_reward(
        model, as_tensor(mean, device = device), as_tensor(std, device = device),
        joint_std = joint_reward_std, joint_weight = joint_reward_weight,
        tip_std = tip_reward_std, tip_weight = tip_reward_weight,
        joint_limit_margin = joint_limit_margin, joint_limit_weight = joint_limit_weight,
        action_smoothness_weight = action_smoothness_weight
    ).to(device)
    raw_reward_fn = make_reward(model).to(device)

    supermimic = SuperMimic(
        actor = actor, world_model = world_model, critic = critic,
        use_ema_critic = use_critic, critic_ema_decay = critic_ema_decay,
        obs_history = actor.obs_history, action_history = actor.action_history, goal_window = actor.goal_window,
        reward_fn = reward_fn, discount_factor = discount_factor,
        num_explorative_rollouts = num_explorative_rollouts,
        dim_explorative_latent = active_latent_dim,
        explorative_latent_scale = explorative_latent_scale_start if active_latent_dim > 0 else 0.
    ).to(device)

    # optimizers & replay buffer

    wm_opt = AdamW(world_model.parameters(), lr = wm_lr, weight_decay = weight_decay)
    wm_sched = CosineAnnealingLR(wm_opt, T_max = max(1, warmup_wm_updates + num_iterations * wm_updates), eta_min = wm_lr_end)
    actor_opt = AdamW(actor.parameters(), lr = actor_lr, betas = (0.9, 0.95), weight_decay = weight_decay)
    actor_sched = CosineAnnealingLR(actor_opt, T_max = max(1, num_iterations * policy_updates), eta_min = actor_lr_end)
    critic_opt = AdamW(critic.parameters(), lr = critic_lr, weight_decay = weight_decay) if use_critic else None

    buffer = StreamReplayBuffer(
        num_streams = train_envs.num_envs,
        capacity = buffer_capacity,
        fields = dict(state = ('float', dim_state), action = ('float', nq), next_state = ('float', dim_state), frame = 'int')
    )

    caches = (None, actor.action_history.init_cache(train_envs.num_envs, device = device))
    total_steps = 0
    history = []
    best_track = -float('inf')
    start_time = time.time()

    def collect(steps: int, noise = 0., random = False):
        nonlocal caches, total_steps

        with temp_eval(actor), torch.no_grad():
            for _ in range(steps):
                norm_states = train_envs.normalize()
                frames = train_envs.frames.copy()

                act, caches = actor.step_policy(
                    as_tensor(norm_states, device = device), train_envs.goals(),
                    caches, noise = noise, random = random
                )

                act_np = act.cpu().numpy()
                next_states, dones = train_envs.step(act_np)
                norm_next = train_envs.normalize(next_states)

                for i in range(train_envs.num_envs):
                    buffer.store_step(i, state = norm_states[i], action = act_np[i], next_state = norm_next[i], frame = int(frames[i]))
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
            step_optimizer(wm_opt, loss, world_model.parameters(), max_norm = grad_clip_norm)
            wm_sched.step()

        return float(loss.item())

    def log_eval(iteration: int, wm_loss: float) -> dict:
        with temp_eval(ema_actor.ema_model):
            metrics = evaluate(eval_env, ema_actor.ema_model, raw_reward_fn, eval_tips, device)

        history.append((total_steps, metrics['track'], metrics['tip_mm'], metrics['q_rmse']))
        print(
            f'[SuperMimic RUKA] iter {iteration:2d} | steps {total_steps:5d} | wm {wm_loss:.5f} | '
            f'track {metrics["track"]:.3f} | q_rmse {metrics["q_rmse"]:.3f} rad | '
            f'tips {metrics["tip_mm"]:5.1f} mm | {time.time() - start_time:.1f}s',
            flush = True
        )
        return metrics

    # warmup & main training loop

    print(f'using device: {device} | train frames: {split} | eval frames: {len(mocap["qpos"]) - split}', flush = True)
    collect(warmup_steps, random = True)
    log_eval(0, fit_world_model(warmup_wm_updates))

    ckpt_path = out_path / 'supermimic_ruka_actor.pt'

    for iteration in range(1, num_iterations + 1):
        frac = (iteration - 1) / max(1, num_iterations - 1)
        cur_noise = action_noise_start + (action_noise_end - action_noise_start) * frac

        collect(collect_steps, noise = cur_noise)
        wm_loss = fit_world_model(wm_updates)

        actor.train()
        latent_scale = cosine_anneal(explorative_latent_scale_start, explorative_latent_scale_end, frac) if active_latent_dim > 0 else 0.

        for _ in range(policy_updates):
            seed_batch = buffer.sample(batch_size, horizon = 2, to_named_tuple = ('state', 'action', 'frame'), device = device)
            starts = seed_batch.frame[:, -1].long() + 1

            goals = gather_seq(train_envs.refs, starts, imag_horizon + goal_window)
            ref_starts = torch.randint(0, split - 1, starts.shape, device = device) if shuffle_refs else starts
            refs = dict(
                joint_pos = gather_seq(train_envs.refs, ref_starts, imag_horizon),
                body_pos = gather_seq(train_tips, ref_starts, imag_horizon)
            )
            critic_goals = goals[:, :imag_horizon + 1]

            rollout = supermimic(
                seed_batch.state[:, -1], actor_goals = goals, references = refs,
                past_obs = seed_batch.state[:, :-1], past_actions = seed_batch.action[:, :-1],
                num_rollout_timesteps = imag_horizon, explorative_latent_scale = latent_scale
            )

            loss_kwargs = dict(final_state = rollout.states[:, -1], final_goal = critic_goals[:, -1]) if use_critic else dict()
            step_optimizer(actor_opt, supermimic.actor_loss(rollout.rewards, **loss_kwargs), actor.parameters(), max_norm = grad_clip_norm, clip_value = actor_grad_clip_value)
            actor_sched.step()
            ema_actor.update()

            if use_critic:
                for _ in range(critic_updates):
                    critic_loss = supermimic.critic_loss(rollout.states, rollout.rewards, goals = critic_goals, lam = gae_lam)
                    step_optimizer(critic_opt, critic_loss, critic.parameters(), max_norm = grad_clip_norm)
                    supermimic.update_ema()

        if iteration % eval_every == 0 or iteration == num_iterations:
            metrics = log_eval(iteration, wm_loss)
            if metrics['track'] > best_track:
                best_track = metrics['track']
                torch.save(ema_actor.ema_model.state_dict(), ckpt_path)

    if ckpt_path.exists():
        ema_actor.ema_model.load_state_dict(torch.load(ckpt_path, map_location = device, weights_only = True))

    with temp_eval(ema_actor.ema_model):
        final = evaluate(eval_env, ema_actor.ema_model, raw_reward_fn, eval_tips, device)

    if save_video:
        save_curves(
            out_path / 'ruka_convergence.png',
            'RUKA-v2 Hand (STL + MuJoCo Torque Control): SuperMimic Tracking Real Human Hand Motion (Held-Out)',
            [('SuperMimic', '#1d4ed8', '-', history, final)]
        )
        save_comparison_video(model, vid_path, split, mocap['qpos'][split:], final['qpos'], final, out_path)

    print(f'[final held-out] track={final["track"]:.3f} | q_rmse={final["q_rmse"]:.3f} rad | tips={final["tip_mm"]:.1f} mm -> {out_path}', flush = True)
    return dict(history = history, final = {k: v for k, v in final.items() if k != 'qpos'})

def run_ablations(out_path: Path, seed = 42, num_iterations = 40, split_frac = 0.75):
    variants = (
        ('Full SuperMimic',          '#1d4ed8', '-',  dict()),
        ('No FK Fingertip Term',     '#0f766e', '-',  dict(tip_reward_weight = 0., joint_reward_weight = 5.0)),
        ('Myopic Goal (W=1)',        '#d97706', '-',  dict(goal_window = 1)),
        ('1-Step BPTT (H=1, no V)',  '#7c3aed', '--', dict(imag_horizon = 1, use_critic = False)),
        ('No Explorative Latent',    '#64748b', '--', dict(num_explorative_rollouts = 1, dim_explorative_latent = 0)),
        ('Neg. Ctrl: Untrained WM',  '#dc2626', ':',  dict(train_wm = False)),
        ('Neg. Ctrl: Shuffled Refs', '#991b1b', '-.', dict(shuffle_refs = True))
    )

    results = {}

    for name, _, _, kwargs in variants:
        print(f'\n=== Ablation: {name} ===', flush = True)
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        results[name] = train(
            out_dir = str(out_path / 'ablation' / slug),
            seed = seed,
            num_iterations = num_iterations,
            split_frac = split_frac,
            save_video = False,
            cache_root = str(out_path),
            **kwargs
        )

    # eval-time negative control: feed time-reversed goals to the trained policy

    device = select_device()
    model, pkl_path, _ = build_ruka_model(out_path / 'cache')
    mocap = retarget_human_hand(model, pkl_path, out_path / 'ruka_human_mocap.npz')

    nq = model.nq
    split = int(len(mocap['qpos']) * split_frac)
    mean, std = norm_stats(mocap['qpos'][:split], mocap['qvel'][:split])

    rev_env = RukaEnvs(model, mocap['qpos'][split:], mocap['qvel'][split:], mean, std, device, is_eval = True)
    rev_env.goal_windows = rev_env.goal_windows.flip(0)

    actor = TrackingActor(2 * nq, nq, nq).to(device)
    ckpt = out_path / 'ablation' / 'full_supermimic' / 'supermimic_ruka_actor.pt'
    actor.load_state_dict(torch.load(ckpt, map_location = device, weights_only = True))

    with temp_eval(actor):
        rev_metrics = evaluate(rev_env, actor, make_reward(model).to(device), as_tensor(mocap['tips'][split:], device = device), device)

    results['Neg. Ctrl: Reversed Eval Goals'] = dict(
        final = {k: rev_metrics[k] for k in ('track', 'q_rmse', 'tip_mm')},
        history = []
    )

    (out_path / 'ablation_results.json').write_text(json.dumps(results, indent = 2))

    series = [(name, color, ls, results[name]['history'], results[name]['final']) for name, color, ls, _ in variants]
    save_curves(
        out_path / 'ruka_ablation.png',
        'RUKA-v2 SuperMimic Ablations & Negative Controls (Held-Out Real Human Hand Motion)',
        series,
        hlines = (rev_metrics['track'], rev_metrics['tip_mm'], 'Neg. Ctrl: Reversed Goals'),
        tip_ylim = 55.
    )

    print('\n=== Summary (Held-Out Real Human Hand) ===')
    for name, res in results.items():
        f = res['final']
        print(f'  {name:32s} | track {f["track"]:6.3f} | q_rmse {f["q_rmse"]:.3f} rad | tips {f["tip_mm"]:5.1f} mm')

def main(
    out_dir = './results_ruka_hand',
    seed = 42,
    num_iterations: int | None = None,
    split_frac = 0.75,
    ablate = False,
    **kwargs
):
    if ablate:
        run_ablations(Path(out_dir), seed = seed, num_iterations = default(num_iterations, 40), split_frac = split_frac)
        return

    train(out_dir = out_dir, seed = seed, num_iterations = default(num_iterations, 60), split_frac = split_frac, **kwargs)

if __name__ == '__main__':
    fire.Fire(main)
