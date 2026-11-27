from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect as py_inspect
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
from ema_pytorch import EMA
from torch import Tensor, as_tensor, cat
from torch.nn import SiLU
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_einops_utils import temp_eval

from supermimic_pytorch import (
    ActionSmoothnessTerm, AnalyticalReward, Critic, HistoryWindow,
    JointLimitPenalty, LatentMotionActor, LatentPolicy, SuperMimic,
    TrackingTerm, WelfordNormalizer, WorldModel, fall_termination, load_policy
)
from supermimic_pytorch.mujoco_mocap import (
    MocapCollectEnvs, MotionBank, Robot, build_motion_bank,
    build_robot as _build_robot, clip_tracking_metrics, compute_action_scaling,
    evaluate_policy, load_mocap_clips, parse_csv, render_qpos_video,
    require, rollout_policy, unpack_state
)
from supermimic_pytorch.supermimic import default, divisible_by, exists
from supermimic_pytorch.welford_normalizer import identity

# helpers

def cosine_anneal(start: float, end: float, frac: float) -> float:
    return end + (start - end) * 0.5 * (1. + math.cos(math.pi * frac))

def parse_ints(val, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not exists(val):
        return fallback
    if isinstance(val, int):
        return (val,)
    items = parse_csv(val) if isinstance(val, str) else val
    return tuple(int(x) for x in items)

def pick_kwonly(fn, source: dict[str, Any]) -> dict[str, Any]:
    return {
        k: source[k]
        for k, p in py_inspect.signature(fn).parameters.items()
        if p.kind == py_inspect.Parameter.KEYWORD_ONLY and k in source
    }

def is_callable_spec(val) -> bool:
    if callable(val):
        return True
    if not isinstance(val, str) or ':' not in val:
        return False
    mod_part, attr_part = val.rsplit(':', 1)
    return len(mod_part) > 1 and attr_part.isidentifier()

def load_callable(ref):
    if callable(ref):
        return ref

    assert isinstance(ref, str) and ':' in ref, f'expected callable or `file_or_module:attr`, got {ref!r}'
    mod_part, attr_name = ref.rsplit(':', 1)
    mod_path = Path(mod_part).expanduser()

    if mod_path.suffix == '.py' or mod_path.exists():
        resolved = mod_path.resolve()
        assert resolved.exists(), f'hook file not found: {resolved}'
        spec = importlib.util.spec_from_file_location(f'_supermimic_hook_{resolved.stem}', resolved)
        assert exists(spec) and exists(spec.loader), f'failed to load hook file: {resolved}'
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_part)

    assert hasattr(mod, attr_name), f'{attr_name!r} not found in {mod_part!r}'
    return getattr(mod, attr_name)

def call_factory(ref, ctx: dict[str, Any]):
    fn = load_callable(ref)
    params = py_inspect.signature(fn).parameters
    if any(p.kind == py_inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**ctx)
    return fn(**{k: v for k, v in ctx.items() if k in params})

def resolve_hook(ref, ctx: dict[str, Any]):
    fn = load_callable(ref)
    if 'states' in py_inspect.signature(fn).parameters:
        return fn
    return call_factory(fn, ctx)

def build_robot(robot_path, **kwargs) -> Robot:
    factory = call_factory if is_callable_spec(robot_path) else None
    return _build_robot(robot_path, factory_fn = factory, **kwargs)

def with_config(fn):
    sig = py_inspect.signature(fn)
    valid = set(sig.parameters) - {'config'}

    @functools.wraps(fn)
    def wrapped(*args, config: str | None = None, **kwargs):
        if not exists(config):
            return fn(*args, **kwargs)

        path = Path(config).expanduser().resolve()
        assert path.exists(), f'config file not found: {path}'

        suf = path.suffix.lower()
        if suf == '.json':
            raw = json.loads(path.read_text())
        elif suf == '.toml':
            import tomllib
            raw = tomllib.loads(path.read_text())
        else:
            raise ValueError(f'unsupported config format {path.suffix!r}; use .json or .toml')

        cfg = {k.replace('-', '_'): v for k, v in raw.items()}
        unknown = sorted(set(cfg) - valid)
        assert not unknown, f'unknown config keys in {config!r}: {unknown}'

        # Fire passes positional defaults as *args; keep config values unless explicitly overridden
        bound_pos = sig.bind_partial(*args).arguments
        overrides = {k: v for k, v in bound_pos.items() if k not in cfg or v != sig.parameters[k].default}
        overrides.update(kwargs)
        return fn(**{**cfg, **overrides})

    return wrapped

# accelerate

def make_accelerator(device = 'auto', mixed_precision = 'no', log_with = None, project_dir = None):
    accelerate = require('accelerate')

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        from accelerate.state import AcceleratorState, PartialState
        PartialState._reset_state()
        AcceleratorState._reset_state()
        if device.startswith('cuda:') and torch.cuda.is_available():
            torch.cuda.set_device(device)

    kwargs = dict(cpu = (device == 'cpu'), mixed_precision = mixed_precision)
    if exists(log_with):
        kwargs['log_with'] = log_with
        if exists(project_dir):
            kwargs['project_dir'] = project_dir
    return accelerate.Accelerator(**kwargs)

def sync_models(accelerator, *models):
    if accelerator.num_processes <= 1:
        return

    for model in models:
        for tensor in (*model.parameters(), *model.buffers()):
            torch.distributed.broadcast(tensor.data, 0)

def step_optimizer(accelerator, opt, loss: Tensor, params, *, max_norm = 1.0, clip_value = None, sched = None):
    params = list(params)
    opt.zero_grad()
    accelerator.backward(loss)

    if accelerator.num_processes > 1:
        with torch.no_grad():
            for p in params:
                if exists(p.grad):
                    p.grad.copy_(accelerator.reduce(p.grad, reduction = 'mean'))

    accelerator.clip_grad_norm_(params, max_norm)
    if exists(clip_value):
        accelerator.clip_grad_value_(params, clip_value)

    opt.step()
    if exists(sched) and not accelerator.optimizer_step_was_skipped:
        sched.step()

# normalizers, masked rewards & mask dropout

def build_normalizers(
    norm: str,
    dim_state: int,
    dim_obs: int,
    robot: Robot,
    has_obs_fn: bool,
    max_count: float | None
):
    assert norm in ('velocities', 'all', 'none'), f'unknown norm mode {norm!r}; use velocities, all, or none'
    if norm == 'none':
        return None, None

    if norm == 'velocities':
        state_norm = WelfordNormalizer(
            dim_state,
            subset = slice(robot.vel_start, None),
            min_std = 1.0,
            max_count = max_count
        )
    else:
        min_std = torch.ones(dim_state)
        min_std[:robot.vel_start] = 0.15
        if robot.has_free_root:
            min_std[0] = 0.05

        excl = slice(1, 5) if robot.has_free_root else None
        state_norm = WelfordNormalizer(
            dim_state,
            exclude = excl,
            min_std = min_std,
            max_count = max_count
        )

    obs_norm = WelfordNormalizer(dim_obs, min_std = 1.0, max_count = max_count) if has_obs_fn else state_norm
    return state_norm, obs_norm

def sample_dropped_masks(
    jm: Tensor,
    rm: Tensor,
    bm: Tensor | None,
    prob: float,
    has_free_root: bool
) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
    batch, _, nj = jm.shape
    dev = jm.device
    drop = torch.rand(batch, device = dev) < prob
    if not drop.any():
        return jm, rm, bm, drop

    jm_out = jm.clone()
    rm_out = rm.clone()
    bm_out = bm.clone() if exists(bm) else None
    modes = torch.randint(0, 4 if exists(bm) else 2, (batch,), device = dev)
    half = max(1, nj // 2)

    for i in drop.nonzero(as_tuple = False).squeeze(-1).tolist():
        mode = int(modes[i])
        if mode == 0 and nj >= 2:
            drop_slice = slice(None, half) if torch.rand((), device = dev) < 0.5 else slice(half, None)
            jm_out[i, :, drop_slice] = 0.
            if has_free_root and torch.rand((), device = dev) < 0.5:
                rm_out[i] = 0.
            if exists(bm_out) and torch.rand((), device = dev) < 0.5:
                bm_out[i] = 0.
        elif mode == 1:
            keep_p = float(torch.empty(()).uniform_(0.25, 0.75).item())
            jm_out[i] = jm_out[i] * (torch.rand((1, nj), device = dev) < keep_p).to(jm_out)
            if has_free_root and torch.rand((), device = dev) < 0.5:
                rm_out[i] = 0.
            if exists(bm_out):
                bm_out[i] = bm_out[i] * (torch.rand((1, bm_out.shape[-1]), device = dev) < keep_p).to(bm_out)
        elif mode == 2 and exists(bm_out) and (bm[i] > 0.5).any():
            jm_out[i] = 0.
            if has_free_root:
                rm_out[i] = 0.
        elif exists(bm_out) and (jm[i] > 0.5).any():
            bm_out[i] = 0.

        has_any = (
            (jm_out[i] > 0.5).any()
            or (has_free_root and (rm_out[i] > 0.5).any())
            or (exists(bm_out) and (bm_out[i] > 0.5).any())
        )
        changed = (
            not torch.equal(jm_out[i], jm[i])
            or (has_free_root and not torch.equal(rm_out[i], rm[i]))
            or (exists(bm_out) and not torch.equal(bm_out[i], bm[i]))
        )
        if not has_any or not changed:
            jm_out[i] = jm[i]
            rm_out[i] = rm[i]
            if exists(bm_out):
                bm_out[i] = bm[i]
            drop[i] = False

    return jm_out, rm_out, bm_out, drop

def make_reward_and_termination(
    robot: Robot,
    bank: MotionBank,
    *,
    joint_weight: float = 2.0,
    body_weight: float = 2.5,
    body_std: float | None = None,
    joint_limit_weight: float = -0.5,
    action_smoothness_weight: float = -0.015,
    unobserved_joint_weight: float = 0.15,
    fall_termination_enabled: bool = True,
    min_height: float | None = None,
    max_tilt: float = 0.55
):
    j_weights = torch.full((robot.nj,), 1.0 / max(1, robot.nj))
    terms: dict[str, Any] = dict(
        joint_pos = TrackingTerm('joint_pos', mask_key = 'joint_mask', std = 0.30, weight = joint_weight, dim_weights = j_weights),
        action_smoothness = ActionSmoothnessTerm(weight = action_smoothness_weight)
    )

    if robot.jnt_limited.any() and joint_limit_weight != 0.:
        terms['joint_limit'] = JointLimitPenalty(
            robot.jnt_low,
            robot.jnt_high,
            key = 'joint_pos',
            margin = 0.03,
            weight = joint_limit_weight
        )

    if bank.use_masks and unobserved_joint_weight > 0.:
        nom_j = default(bank.nominal_qpos, robot.qpos0)[robot.q_start:].clone()

        def unobserved_posture(states, references = None, **kwargs):
            if not exists(references) or 'joint_mask' not in references:
                return None
            unobs = (1. - references['joint_mask'].to(states['joint_pos'])).clamp(min = 0., max = 1.)
            cnt = unobs.sum(dim = -1)
            err = ((states['joint_pos'] - nom_j.to(states['joint_pos'])).square() * unobs).sum(dim = -1) / cnt.clamp(min = 1.)
            return unobserved_joint_weight * torch.exp(-err / (0.45 ** 2)) * (cnt > 0.).to(err)

        terms['unobserved_posture'] = unobserved_posture

    med_z = 0.3
    if robot.has_free_root:
        valid_r = bank.all_root_mask[:, 0] > 0.5
        med_z = float(bank.all_states[valid_r, 0].median().item()) if valid_r.any() else float(default(bank.nominal_qpos, robot.qpos0)[2])
        terms.update(
            root_height = TrackingTerm('root_z', mask_key = 'root_mask', std = max(0.03, 0.15 * max(med_z, 0.1)), weight = 1.0),
            root_orientation = TrackingTerm('root_quat', mask_key = 'root_mask', metric = 'rotation', std = 0.30, weight = 1.0),
            root_lin_vel = TrackingTerm('root_lin_vel', mask_key = 'root_mask', std = 0.35, weight = 1.5),
            root_ang_vel = TrackingTerm('root_ang_vel', mask_key = 'root_mask', std = 0.50, weight = 1.0)
        )

    if exists(robot.fk) and body_weight > 0.:
        b_std = float(default(body_std, max(0.015, 0.14 * bank.body_span)))
        terms['body_pos'] = TrackingTerm('body_pos', mask_key = 'body_mask', std = b_std, weight = body_weight, reduce_items = True)

    reward_fn = AnalyticalReward(
        terms,
        state_transform = lambda s: unpack_state(s, robot, with_fk = True),
        normalize_masked = True
    )

    term_fn = fall_cfg = None
    if robot.has_free_root and fall_termination_enabled:
        min_h = float(default(min_height, 0.45 * max(med_z, 0.1)))
        fall_cfg = dict(min_height = min_h, max_tilt = float(max_tilt))
        term_fn = lambda s, _r = None: fall_termination(
            unpack_state(s, robot),
            min_height = min_h,
            tilt_threshold = max_tilt,
            temperature = 0.015
        )

    return reward_fn, term_fn, fall_cfg

# CLI commands

@with_config
def inspect(
    robot: str | None = None, mocap: str | None = None, *, config: str | None = None,
    fps: float = 30.0, substeps: int = 8, control: str = 'auto', pd_kp: float | None = None, pd_kd: float | None = None,
    lock_joints = None, tracked_bodies = 'auto', quat_order: str = 'wxyz',
    mode: str = 'auto', imag_horizon: int = 16, clip_len: int | None = None, clip_stride: int | None = None
) -> dict[str, Any]:
    """Inspect a MuJoCo robot model and verify reference mocap compatibility."""
    assert exists(robot), 'pass `--robot` (or set `robot` in `--config`)'
    args = locals()

    rob = build_robot(robot, **pick_kwonly(_build_robot, args))
    summary: dict[str, Any] = dict(
        robot = rob.path,
        floating_base = rob.has_free_root,
        nq = rob.nq,
        nv = rob.nv,
        nu = rob.nu,
        dim_state = rob.dim_state,
        state_layout = rob.state_layout,
        control = rob.control,
        tracked_bodies = rob.tracked_bodies
    )
    print(
        f'Robot: {rob.path}\n'
        f'  floating_base={rob.has_free_root} | nq={rob.nq} nv={rob.nv} nu={rob.nu} | state_dim={rob.dim_state} ({rob.state_layout})\n'
        f'  control={rob.control} | tracked_bodies={list(rob.tracked_bodies)}'
    )

    if exists(mocap):
        clips = load_mocap_clips(mocap, rob, quat_order = quat_order, target_fps = fps)
        bank = build_motion_bank(clips, rob, **pick_kwonly(build_motion_bank, args))
        summary.update(
            num_clips = len(clips),
            total_frames = int(sum(len(c['qpos']) for c in clips)),
            mode = bank.mode,
            num_primitives = len(bank.names),
            dim_motion = bank.dim_motion,
            partial_masks = bank.has_partial
        )
        print(
            f'Mocap: {len(clips)} clip(s), {summary["total_frames"]} frames @ {fps:g} Hz\n'
            f'  mode={bank.mode} | primitives={len(bank.names)} (window={bank.window_len}) | dim_motion={bank.dim_motion} | partial_masks={bank.has_partial}'
        )
        for clip in clips[:8]:
            j_cov = int((clip['joint_mask'][0] > 0.5).sum().item())
            b_cov = int((clip['body_mask'][0] > 0.5).sum().item()) if exists(clip['body_mask']) else 0
            r_s = f' root={int(clip["root_mask"][0, 0] > 0.5)}' if rob.has_free_root else ''
            print(f'    - {clip["name"]}: {len(clip["qpos"])} frames (joints={j_cov}/{rob.nj}, bodies={b_cov}/{rob.nb}{r_s})')
        if len(clips) > 8:
            print(f'    ... (+{len(clips) - 8} more)')

    return summary

@with_config
def train(
    robot: str | Any | None = None, mocap: str | None = None, out: str = './supermimic_policy.pt', *, config: str | None = None,
    # latent space & normalization
    mode: str = 'auto', dim_latent: int = 4, latent_space: str = 'box', clip_len: int | None = None, clip_stride: int | None = None,
    enc_frames: int = 5, uniformity_weight: float = 4.0, latent_noise: float = 0.10, norm: str = 'all', norm_max_count: float | None = 100_000,
    # masked references & optional policy mask conditioning
    mask_dropout: float = 0.0, mask_align_weight: float = 1.0, policy_mask: bool = False, policy_mask_dropout: float = 0.5, unobserved_joint_weight: float = 0.15,
    # MuJoCo robot & reward
    fps: float = 30.0, substeps: int = 8, control: str = 'auto', pd_kp: float | None = None, pd_kd: float | None = None,
    torque_gear: float = 0.32, max_torque: float | None = None, lock_joints = None, tracked_bodies = 'auto',
    joint_damping: float | None = None, joint_armature: float | None = None, disable_collisions: bool = False, quat_order: str = 'wxyz',
    fall_termination_enabled: bool = True, min_height: float | None = None, max_tilt: float = 0.55, obs_noise: float = 0.0,
    joint_weight: float = 2.0, body_weight: float = 2.5, body_std: float | None = None, joint_limit_weight: float = -0.5, action_smoothness_weight: float = -0.015,
    # custom Python hooks
    reward_fn = None, termination_fn = None, extra_state_fn = None, obs_fn = None, actor_fn = None, world_model_fn = None,
    # training loop & architecture
    iterations: int = 80, num_envs: int = 16, env_horizon: int = 96, warmup_steps: int = 128, warmup_wm_updates: int = 300,
    collect_steps: int = 16, buffer_capacity: int = 64_000, batch_size: int = 256, wm_horizon: int = 8, imag_horizon: int = 16,
    wm_rollout_loss_decay: float = 1., wm_state_noise_scale: float = 0.,
    wm_updates: int = 20, policy_updates: int = 8, critic_updates: int = 4, command_every: int = 8, switch_every: int = 24,
    obs_window: int = 2, action_window: int = 1, encoder_hiddens = (128, 128), policy_hiddens = (256, 256, 256),
    wm_hiddens = (512, 512, 512), critic_hiddens = (256, 256), dim_explorative_latent: int = 4, num_explorative_rollouts: int = 2,
    wm_lr: float = 6e-4, actor_lr: float = 5e-4, critic_lr: float = 3e-4, ema_actor_decay: float = 0.96, critic_ema_decay: float = 0.99,
    discount_factor: float = 0.99, gae_lam: float = 0.95, use_median_grad_clip: bool | None = None, median_grad_clip_mult = 10.,
    # accelerate & logging
    mixed_precision: str = 'no', log_with: str | None = None, eval_every: int = 10, seed: int = 42, device: str = 'auto'
) -> dict[str, Any]:
    """Train a SuperMimic latent-controllable motor policy from a MuJoCo robot and reference mocap."""
    assert exists(robot) and exists(mocap), 'pass `--robot` and `--mocap` (or set them in `--config`)'
    args = locals()

    StreamReplayBuffer = require('memmap-replay-buffer', 'memmap_replay_buffer').StreamReplayBuffer
    from accelerate.utils import set_seed

    out_path = Path(out).expanduser()
    if out_path.suffix != '.pt':
        out_path = out_path / 'supermimic_policy.pt'
    out_path.parent.mkdir(parents = True, exist_ok = True)

    # accelerator, robot & motion bank

    accelerator = make_accelerator(device, mixed_precision, log_with, str(out_path.parent))
    set_seed(seed + accelerator.process_index)
    dev = accelerator.device

    if exists(log_with):
        accelerator.init_trackers('supermimic')

    rob = build_robot(robot, **pick_kwonly(_build_robot, args))
    clips = load_mocap_clips(mocap, rob, quat_order = quat_order, target_fps = fps)
    action_offset, action_scale, ctrl_low, ctrl_high = compute_action_scaling(clips, rob)
    rob.ctrl_low = ctrl_low
    rob.ctrl_high = ctrl_high

    bank = build_motion_bank(clips, rob, **pick_kwonly(build_motion_bank, args))
    extra_fn = load_callable(extra_state_fn) if exists(extra_state_fn) else None
    to_obs = load_callable(obs_fn) if exists(obs_fn) else identity

    default_reward, default_term, fall_cfg = make_reward_and_termination(
        rob, bank, **pick_kwonly(make_reward_and_termination, args)
    )
    envs = MocapCollectEnvs(
        rob,
        clips,
        bank,
        action_offset,
        action_scale,
        ctrl_low,
        ctrl_high,
        fall_cfg,
        extra_state_fn = extra_fn,
        num_envs = num_envs,
        horizon = env_horizon,
        switch_every = switch_every
    )
    bank.to(dev)

    dim_state = envs.dim_state
    dim_action = rob.nu
    dim_motion = bank.dim_motion
    with torch.no_grad():
        dim_obs = int(to_obs(envs.states[:1].to(dev)).shape[-1])

    state_norm, obs_norm = build_normalizers(norm, dim_state, dim_obs, rob, exists(obs_fn), norm_max_count)

    # models, optimizers & supermimic

    active_exp = dim_explorative_latent if num_explorative_rollouts > 1 else 0
    dim_pmask = rob.dim_policy_mask if policy_mask else 0

    actor_kw = dict(
        dim_obs = dim_obs,
        dim_action = dim_action,
        dim_motion = dim_motion,
        dim_latent = dim_latent,
        dim_mask = dim_pmask,
        num_obs_frames = obs_window,
        num_action_frames = action_window,
        dim_explorative_latent = active_exp,
        latent_space = latent_space,
        latent_noise_scale = latent_noise,
        encoder_hiddens = parse_ints(encoder_hiddens, (128, 128)),
        policy_hiddens = parse_ints(policy_hiddens, (256, 256, 256)),
        activation = SiLU(),
        normed_mlp = True,
        squash_action = True
    )
    wm_kw = dict(
        dim_state = dim_state,
        dim_action = dim_action,
        dim_sim_context = 0,
        dim_hiddens = parse_ints(wm_hiddens, (512, 512, 512)),
        activation = SiLU(),
        normed_mlp = True,
        pred_terminal = False,
        rollout_loss_decay = wm_rollout_loss_decay,
        state_noise_scale = wm_state_noise_scale,
        state_normalizer = state_norm
    )

    actor = (call_factory(actor_fn, dict(**actor_kw, robot = rob, bank = bank)) if exists(actor_fn) else LatentMotionActor(**actor_kw)).to(dev)
    world_model = (call_factory(world_model_fn, dict(**wm_kw, robot = rob, bank = bank)) if exists(world_model_fn) else WorldModel(**wm_kw)).to(dev)
    critic = Critic(
        dim_state = dim_state,
        dim_goal = dim_latent + dim_pmask,
        dim_sim_context = 0,
        dim_hiddens = parse_ints(critic_hiddens, (256, 256)),
        activation = SiLU(),
        normed_mlp = True,
        state_normalizer = state_norm
    ).to(dev)
    sync_models(accelerator, world_model, actor, critic)

    wm_opt = AdamW(world_model.parameters(), lr = wm_lr, weight_decay = 1e-5)
    actor_opt = AdamW(actor.parameters(), lr = actor_lr, betas = (0.9, 0.95), weight_decay = 1e-5)
    critic_opt = AdamW(critic.parameters(), lr = critic_lr, weight_decay = 1e-5)

    wm_sched = CosineAnnealingLR(wm_opt, T_max = max(1, warmup_wm_updates + iterations * wm_updates), eta_min = wm_lr * 0.15)
    actor_sched = CosineAnnealingLR(actor_opt, T_max = max(1, iterations * policy_updates), eta_min = actor_lr * 0.10)
    wm_opt, actor_opt, critic_opt = accelerator.prepare(wm_opt, actor_opt, critic_opt)

    ema_actor = EMA(actor, beta = ema_actor_decay, include_online_model = False, update_after_step = 0, update_every = 1).to(dev)

    hook_ctx = dict(robot = rob, bank = bank, default_reward = default_reward, default_termination = default_term)
    resolved_reward_fn = resolve_hook(reward_fn, hook_ctx) if exists(reward_fn) else default_reward
    if isinstance(resolved_reward_fn, torch.nn.Module):
        resolved_reward_fn = resolved_reward_fn.to(dev)
    resolved_term_fn = resolve_hook(termination_fn, hook_ctx) if exists(termination_fn) else default_term

    obs_history = HistoryWindow(obs_window, dim_obs, pad_mode = 'replicate')
    action_history = HistoryWindow(action_window, dim_action, lag = 1, pad_mode = 'zero') if action_window > 0 else None

    supermimic = SuperMimic(
        actor = actor,
        world_model = world_model,
        critic = critic,
        use_ema_critic = True,
        critic_ema_decay = critic_ema_decay,
        observation_noise_scale = obs_noise if obs_noise > 0. else None,
        state_to_obs = (lambda s, _c = None: to_obs(s)) if exists(obs_fn) else None,
        obs_normalizer = obs_norm,
        obs_history = obs_history,
        action_history = action_history,
        reward_fn = resolved_reward_fn,
        termination_fn = resolved_term_fn,
        discount_factor = discount_factor,
        num_explorative_rollouts = num_explorative_rollouts,
        dim_explorative_latent = active_exp,
        explorative_latent_scale = 0.20 if active_exp > 0 else 0.,
        use_median_grad_clip = use_median_grad_clip,
        median_grad_clip_mult = median_grad_clip_mult
    ).to(dev)

    # replay buffer & rollouts

    buffer = StreamReplayBuffer(
        num_streams = num_envs,
        capacity = buffer_capacity,
        fields = dict(
            state = ('float', dim_state),
            action = ('float', dim_action),
            next_state = ('float', dim_state),
            frame = 'int'
        )
    )
    num_prims = len(bank.motion_feats)
    seed_horizon = max(obs_window, action_window + 1, 2)
    obs_cache = None
    act_cache = action_history.init_cache(num_envs, device = dev) if exists(action_history) else None
    smooth_rand = torch.zeros((num_envs, dim_action), device = dev)
    env_z_cmd = torch.zeros((num_envs, dim_latent), device = dev)
    env_m_cmd = torch.zeros((num_envs, dim_pmask), device = dev) if dim_pmask > 0 else None
    total_steps = 0

    def encode_window(gframes: Tensor, noise_scale = 0., drop_prob = 0.):
        s_w, b_w, jm_w, rm_w, bm_w = bank.gather(gframes, bank.window_len)

        z_full = align_mask = None
        if drop_prob > 0.:
            with torch.no_grad():
                z_full = actor.encode(bank.encode(s_w, b_w, jm_w, rm_w, bm_w), add_noise = False)
            jm_e, rm_e, bm_e, align_mask = sample_dropped_masks(jm_w, rm_w, bm_w, drop_prob, rob.has_free_root)
        else:
            jm_e, rm_e, bm_e = jm_w, rm_w, bm_w

        z_clean = actor.encode(bank.encode(s_w, b_w, jm_e, rm_e, bm_e), add_noise = False)
        z = actor.project_latent(z_clean + torch.randn_like(z_clean) * noise_scale) if noise_scale > 0. else z_clean
        pm = bank.policy_mask(jm_e[:, 0], rm_e[:, 0], bm_e[:, 0] if exists(bm_e) else None) if dim_pmask > 0 else None
        return z, z_clean, z_full, align_mask, pm

    def collect(steps: int, action_noise = 0., lat_noise = 0., random = False):
        nonlocal obs_cache, act_cache, smooth_rand, env_z_cmd, env_m_cmd, total_steps

        with temp_eval(actor), torch.no_grad():
            z_bank = actor.encode(bank.motion_feats, add_noise = False) if (not random and bank.mode == 'skills') else None

            for _ in range(steps):
                raw_st = envs.states.clone()
                gframes = envs.global_frames().clone()
                raw_obs_t = to_obs(raw_st.to(dev))

                if exists(obs_norm) and obs_norm is not state_norm:
                    obs_norm.update(raw_obs_t)

                obs_in = obs_norm.normalize(raw_obs_t) if exists(obs_norm) else raw_obs_t
                obs_w, obs_cache = obs_history.step(obs_in, obs_cache)
                past_a = action_history.read(act_cache) if exists(action_history) else None

                if random:
                    smooth_rand = 0.65 * smooth_rand + 0.35 * torch.empty_like(smooth_rand).uniform_(-1., 1.)
                    act_t = (smooth_rand * 1.4).clamp(-1., 1.)
                else:
                    m_cmd = None
                    if bank.mode == 'skills':
                        pids = envs.prim_ids.to(dev)
                        z_cmd = z_bank[pids]
                        if lat_noise > 0.:
                            z_cmd = actor.project_latent(z_cmd + torch.randn_like(z_cmd) * lat_noise)
                        if dim_pmask > 0:
                            bm0 = bank.skill_ref_bmask[pids, 0] if exists(bank.skill_ref_bmask) else None
                            m_cmd = bank.policy_mask(bank.skill_ref_jmask[pids, 0], bank.skill_ref_rmask[pids, 0], bm0)
                    else:
                        refresh = (envs.steps % command_every) == 0
                        if refresh.any():
                            idx = refresh.nonzero(as_tuple = False).squeeze(-1)
                            new_z, _, _, _, new_m = encode_window(gframes[idx].to(dev), lat_noise)

                            if num_prims > 1:
                                swap = torch.rand(len(idx), device = dev) < 0.12
                                if swap.any():
                                    rand_ids = torch.randint(0, num_prims, (int(swap.sum().item()),), device = dev)
                                    new_z[swap] = actor.encode(bank.motion_feats[rand_ids], add_noise = False)
                                    if exists(new_m):
                                        new_m[swap] = 0.

                            idx_dev = idx.to(dev)
                            env_z_cmd[idx_dev] = new_z
                            if exists(env_m_cmd):
                                env_m_cmd[idx_dev] = new_m

                        z_cmd = env_z_cmd
                        m_cmd = env_m_cmd

                    if exists(m_cmd) and policy_mask_dropout > 0.:
                        keep = (torch.rand((num_envs, 1), device = dev) >= policy_mask_dropout).to(m_cmd)
                        m_cmd = m_cmd * keep

                    act_t = actor.act(obs_w, z_cmd, mask = m_cmd, past_action = past_a)
                    if action_noise > 0.:
                        act_t = (act_t + torch.randn_like(act_t) * action_noise).clamp(-1., 1.)

                if exists(action_history):
                    act_cache = action_history.push(act_t, act_cache)

                act_cpu = act_t.cpu()
                next_states, dones = envs.step(act_cpu)

                for i in range(num_envs):
                    buffer.store_step(i, state = raw_st[i], action = act_cpu[i], next_state = next_states[i], frame = int(gframes[i]))
                    if dones[i]:
                        buffer.end_episode(i)

                total_steps += num_envs * accelerator.num_processes
                if dones.any():
                    done_t = dones.to(dev)
                    obs_cache = obs_history.reset(obs_cache, done_t)
                    if exists(action_history):
                        act_cache = action_history.reset(act_cache, done_t)
                    smooth_rand[done_t] = 0.

    def fit_world_model(updates: int) -> float:
        world_model.train()
        last_loss = 0.0

        for _ in range(updates):
            b = buffer.sample(batch_size, horizon = wm_horizon, to_named_tuple = ('state', 'action', 'next_state'), device = dev)
            with accelerator.autocast():
                loss = world_model.rollout(b.state[:, 0], b.action, b.next_state)[0] / wm_horizon
            step_optimizer(accelerator, wm_opt, loss, world_model.parameters(), max_norm = 1.0, sched = wm_sched)
            last_loss = float(loss.item())

        return last_loss

    # training loop

    unif_temp = 5.0 / max(1, dim_latent)
    t0 = time.time()
    proc_s = f' x{accelerator.num_processes}' if accelerator.num_processes > 1 else ''
    mask_s = f', masked(pmask={policy_mask}, drop={mask_dropout:g})' if bank.use_masks else ''
    accelerator.print(
        f'[supermimic] robot={Path(rob.path).name} (floating={rob.has_free_root}, nq={rob.nq}, nu={rob.nu}, '
        f'control={rob.control}, norm={norm}{mask_s}) | state={dim_state}d obs={dim_obs}d | '
        f'mocap={len(clips)} clips -> {num_prims} primitives ({bank.mode}) -> {dim_latent}d {latent_space} latent | '
        f'device={dev}{proc_s}'
    )

    collect(warmup_steps, random = True)
    wm_loss = fit_world_model(warmup_wm_updates)
    best_score = float('inf')
    history = []

    for it in range(1, iterations + 1):
        frac = (it - 1) / max(1, iterations - 1)
        cur_lat_noise = cosine_anneal(latent_noise, latent_noise * 0.65, frac)

        collect(collect_steps, action_noise = 0.08 - 0.06 * frac, lat_noise = cur_lat_noise * 0.75)
        wm_loss = fit_world_model(wm_updates)

        actor.train()
        for _ in range(policy_updates):
            seed_b = buffer.sample(batch_size, horizon = seed_horizon, to_named_tuple = ('state', 'action', 'frame'), device = dev)
            past_obs = to_obs(seed_b.state[:, :-1]) if obs_window > 1 else None
            past_act = seed_b.action[:, :-1] if action_window > 0 else None

            with accelerator.autocast():
                z_clean = z_full = align_mask = pmask = None

                if bank.mode == 'skills':
                    targets = torch.randint(0, num_prims, (batch_size,), device = dev)
                    z_all = actor.encode(bank.motion_feats, add_noise = False)
                    z_cmd = actor.project_latent(z_all[targets] + torch.randn((batch_size, dim_latent), device = dev) * cur_lat_noise)
                    ref_states = bank.skill_ref_states[targets]
                    ref_bodies = bank.skill_ref_bodies[targets] if exists(bank.skill_ref_bodies) else None
                    ref_jm = bank.skill_ref_jmask[targets]
                    ref_rm = bank.skill_ref_rmask[targets]
                    ref_bm = bank.skill_ref_bmask[targets] if exists(bank.skill_ref_bmask) else None
                    z_unif = z_all
                    if dim_pmask > 0:
                        pmask = bank.policy_mask(ref_jm[:, 0], ref_rm[:, 0], ref_bm[:, 0] if exists(ref_bm) else None)
                else:
                    next_starts = seed_b.frame[:, -1] + 1
                    ref_states, ref_bodies, ref_jm, ref_rm, ref_bm = bank.gather(next_starts, imag_horizon)
                    z_cmd, z_clean, z_full, align_mask, pmask = encode_window(next_starts, cur_lat_noise, drop_prob = mask_dropout)
                    sub = bank.motion_feats if num_prims <= 32 else bank.motion_feats[torch.randperm(num_prims, device = dev)[:32]]
                    z_unif = actor.encode(sub, add_noise = False)

                refs = unpack_state(ref_states, rob)
                refs.update(joint_mask = ref_jm, root_mask = ref_rm)
                if exists(ref_bodies):
                    refs.update(body_pos = ref_bodies, body_mask = ref_bm)

                if dim_pmask > 0:
                    keep = (torch.rand((batch_size, 1), device = dev) >= policy_mask_dropout).to(pmask)
                    actor_cond = cat((z_cmd, pmask * keep), dim = -1)
                    critic_goal = cat((z_cmd.detach(), pmask), dim = -1)
                else:
                    actor_cond = z_cmd
                    critic_goal = z_cmd.detach()

                rollout = supermimic(
                    seed_b.state[:, -1],
                    actor_cond = actor_cond,
                    references = refs,
                    past_obs = past_obs,
                    past_actions = past_act,
                    num_rollout_timesteps = imag_horizon,
                    explorative_latent_scale = cosine_anneal(0.20, 0.03, frac) if active_exp > 0 else 0.
                )

                loss = supermimic.actor_loss(
                    rollout.rewards,
                    rollout.terminals,
                    final_state = rollout.states[:, -1],
                    final_goal = critic_goal
                )
                if uniformity_weight > 0. and len(z_unif) >= 2:
                    loss = loss + uniformity_weight * actor.uniformity_loss(
                        z_unif,
                        is_encoded = True,
                        temperature = unif_temp,
                        bound_weight = 15.0 / max(1.0, len(z_unif) / 6.0)
                    )
                if exists(align_mask) and align_mask.any() and mask_align_weight > 0.:
                    loss = loss + mask_align_weight * (z_clean[align_mask] - z_full[align_mask]).square().mean()

            step_optimizer(accelerator, actor_opt, loss, actor.parameters(), max_norm = 1.0, clip_value = 0.1, sched = actor_sched)
            ema_actor.update()

            for _ in range(critic_updates):
                with accelerator.autocast():
                    c_loss = supermimic.critic_loss(
                        rollout.states,
                        rollout.rewards,
                        rollout.terminals,
                        goals = critic_goal,
                        lam = gae_lam
                    )
                step_optimizer(accelerator, critic_opt, c_loss, critic.parameters(), max_norm = 1.0)
                supermimic.update_ema()

        # periodic evaluation & checkpointing

        if (divisible_by(it, eval_every) or it == iterations) and accelerator.is_main_process:
            with temp_eval(ema_actor.ema_model), torch.no_grad():
                z_all = ema_actor.ema_model.encode(bank.motion_feats, add_noise = False).cpu()
                obs_mean, obs_std = tuple(t.detach().cpu() for t in obs_norm.stats()) if exists(obs_norm) else (None, None)

            meta = dict(
                robot_path = rob.path,
                robot_cfg = rob.cfg,
                floating_base = rob.has_free_root,
                state_layout = rob.state_layout,
                extra_state_fn = extra_state_fn if isinstance(extra_state_fn, str) else None,
                obs_fn = obs_fn if isinstance(obs_fn, str) else None,
                nominal_qpos = bank.nominal_qpos.cpu().tolist() if exists(bank.nominal_qpos) else None,
                ctrl_low = ctrl_low.tolist(),
                ctrl_high = ctrl_high.tolist(),
                **bank.to_metadata()
            )
            bundle = LatentPolicy(
                ema_actor.ema_model,
                obs_mean = obs_mean,
                obs_std = obs_std,
                action_offset = action_offset,
                action_scale = action_scale,
                clip_latents = z_all,
                clip_names = bank.names,
                metadata = meta
            ).to(dev)

            ev = evaluate_policy(
                rob,
                clips,
                bank,
                bundle,
                dev,
                fall_cfg,
                extra_state_fn = extra_fn,
                obs_fn = to_obs,
                command_every = command_every
            )
            history.append(dict(iter = it, steps = total_steps, wm_loss = wm_loss, **ev))
            if exists(log_with):
                accelerator.log(dict(wm_loss = wm_loss, **ev), step = total_steps)

            ee_s = f' | ee {ev["ee_mm"]:5.1f}mm' if bank.has_bodies else ''
            surv_s = f' | surv {ev["surv"] * 100:5.1f}%' if rob.has_free_root else ''
            accelerator.print(
                f'[supermimic] {it:3d}/{iterations:3d} | steps {total_steps:6d} | wm {wm_loss:.5f} '
                f'| q_rmse {ev["q_rmse"]:.3f}rad{ee_s}{surv_s} | min|dz| {ev["min_z_dist"]:.2f} | {time.time() - t0:.1f}s'
            )

            score = ev['q_rmse'] + (0.01 * ev['ee_mm'] if bank.has_bodies else 0.) + (2.0 * (1.0 - ev['surv']) if rob.has_free_root else 0.)
            if score < best_score or not out_path.exists():
                best_score = score
                bundle.save(out_path)

    accelerator.wait_for_everyone()
    if exists(log_with):
        accelerator.end_training()
    accelerator.print(f'[supermimic] saved portable latent policy -> {out_path}')
    return dict(checkpoint = str(out_path), history = history)

@with_config
@torch.no_grad()
def eval(
    checkpoint: str | None = None, *, config: str | None = None, robot: str | Any | None = None, mocap: str | None = None,
    extra_state_fn = None, obs_fn = None, clip: int | str = 0, random_latent: bool = False, steps: int | None = None,
    switch_every: int = 24, command_every: int = 8, video: str | None = None, video_size: int = 480,
    quat_order: str = 'wxyz', seed: int = 0, device: str = 'auto'
) -> dict[str, Any]:
    """Evaluate a saved SuperMimic LatentPolicy checkpoint and optionally render a video."""
    assert exists(checkpoint), 'pass `--checkpoint` (or set `checkpoint` in `--config`)'
    from accelerate.utils import set_seed

    accelerator = make_accelerator(device = device)
    set_seed(seed)
    dev = accelerator.device

    policy = load_policy(checkpoint, map_location = dev).to(dev)
    meta = policy.metadata

    robot_path = default(robot, meta.get('robot_path'))
    assert exists(robot_path), 'pass `--robot` (checkpoint metadata has no robot_path)'
    rob = build_robot(robot_path, **meta.get('robot_cfg', {}))
    if exists(meta.get('ctrl_low')):
        rob.ctrl_low = as_tensor(meta['ctrl_low'], dtype = torch.float32)
        rob.ctrl_high = as_tensor(meta['ctrl_high'], dtype = torch.float32)

    extra_ref = default(extra_state_fn, meta.get('extra_state_fn'))
    obs_ref = default(obs_fn, meta.get('obs_fn'))
    extra_fn = load_callable(extra_ref) if exists(extra_ref) else None
    to_obs = load_callable(obs_ref) if exists(obs_ref) else identity

    bank = MotionBank.from_metadata(meta, rob)
    if exists(meta.get('nominal_qpos')):
        bank.nominal_qpos = as_tensor(meta['nominal_qpos'], dtype = torch.float32)

    ref_clip = z_fixed = None
    if exists(mocap):
        loaded = load_mocap_clips(mocap, rob, quat_order = quat_order, target_fps = rob.fps)
        if isinstance(clip, int) or str(clip).isdigit():
            idx = int(clip)
        else:
            idx = next((i for i, c in enumerate(loaded) if c['name'] == str(clip)), 0)
        ref_clip = loaded[idx % len(loaded)]
        num_steps = min(int(steps), len(ref_clip['qpos']) - 1) if exists(steps) else (len(ref_clip['qpos']) - 1)
    else:
        num_steps = int(default(steps, 150))
        if not random_latent:
            assert exists(policy.clip_latents) and len(policy.clip_latents) > 0, (
                'checkpoint has no clip_latents; pass --random-latent or --mocap'
            )
            if isinstance(clip, str) and not clip.isdigit() and clip in policy.clip_names:
                cidx = policy.clip_names.index(clip)
            else:
                cidx = int(clip) % len(policy.clip_latents)
            z_fixed = policy.clip_latents[cidx : cidx + 1].to(dev)

    states_run, qposes, _ = rollout_policy(
        rob,
        policy,
        dev,
        num_steps,
        ref_clip = ref_clip,
        z_fixed = z_fixed,
        random_latent = random_latent,
        bank = bank,
        extra_state_fn = extra_fn,
        obs_fn = to_obs,
        switch_every = switch_every,
        command_every = command_every
    )

    result: dict[str, Any] = dict(
        steps = len(states_run),
        dim_latent = policy.dim_latent,
        latent_space = policy.latent_space
    )
    if exists(ref_clip) and states_run:
        q_rmse, ee_mm = clip_tracking_metrics(states_run, ref_clip, rob, dev)
        parts = [f'steps={len(states_run)}']
        if exists(q_rmse):
            result['q_rmse'] = q_rmse
            parts.append(f'q_rmse={q_rmse:.3f} rad')
        if exists(ee_mm):
            result['ee_mm'] = ee_mm
            parts.append(f'ee={ee_mm:.1f} mm')
        accelerator.print('[supermimic eval] ' + ' | '.join(parts))
    else:
        accelerator.print(f'[supermimic eval] completed {len(states_run)} steps (latent_dim={policy.dim_latent}, space={policy.latent_space})')

    if exists(video):
        vid_path = Path(video).expanduser()
        render_qpos_video(rob.model, qposes, vid_path, rob.fps, rob.has_free_root, size = video_size)
        result['video'] = str(vid_path)
        accelerator.print(f'[supermimic eval] saved video -> {vid_path}')

    return result

def ignore_return(fn):
    @functools.wraps(fn)
    def runner(*args, **kwargs):
        fn(*args, **kwargs)
    return runner

def main():
    require('fire').Fire(dict(
        inspect = ignore_return(inspect),
        train = ignore_return(train),
        eval = ignore_return(eval)
    ))

if __name__ == '__main__':
    main()
