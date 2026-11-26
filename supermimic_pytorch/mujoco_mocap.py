from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import einx
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, as_tensor, cat, stack

from supermimic_pytorch import ForwardKinematics, LatentPolicy
from supermimic_pytorch.rewards import l2norm, quat_mul, quat_rotate, world_to_heading_frame
from supermimic_pytorch.supermimic import default, divisible_by, exists
from supermimic_pytorch.welford_normalizer import identity

MOCAP_EXTS = {'.npz', '.npy', '.pt', '.pth', '.csv'}

# helpers

def require(pkg: str, import_name: str | None = None):
    try:
        return __import__(default(import_name, pkg))
    except ImportError as err:
        raise ImportError(
            f'SuperMimic CLI requires `{pkg}` (`pip install "supermimic-pytorch[cli]"`).'
        ) from err

def parse_csv(val) -> tuple[str, ...]:
    if not exists(val):
        return ()
    if isinstance(val, (tuple, list)):
        return tuple(s for item in val for s in parse_csv(str(item)))

    text = str(val).strip()
    if text.lower() in ('', 'none', 'false'):
        return ()

    return tuple(s.strip() for s in text.split(',') if s.strip())

def first_key(mapping: dict, *keys: str) -> str | None:
    return next((k for k in keys if k in mapping), None)

def to_float_tensor(x) -> Tensor:
    if torch.is_tensor(x):
        return x.detach().cpu().float()
    return as_tensor(x, dtype = torch.float32)

def to_str_list(val) -> list[str]:
    if isinstance(val, str):
        return list(parse_csv(val))
    items = val.tolist() if hasattr(val, 'tolist') else list(val)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in items]

def fix_quat_sign(quats: Tensor) -> Tensor:
    q = quats.clone()
    dots = (q[1:] * q[:-1]).sum(dim = -1)
    flips = (dots < 0.).long().cumsum(dim = 0) % 2
    q[1:] = torch.where(rearrange(flips == 1, 't -> t 1'), -q[1:], q[1:])
    return q

def quat_sub(qa: Tensor, qb: Tensor) -> Tensor:
    qb_inv = cat((qb[..., :1], -qb[..., 1:]), dim = -1)
    dq = quat_mul(qb_inv, qa)
    dq = torch.where(dq[..., :1] < 0., -dq, dq)

    u = dq[..., 1:]
    norm = u.norm(dim = -1, keepdim = True)
    angle = 2. * torch.atan2(norm, dq[..., :1].clamp(min = 1e-8))
    return u * (angle / norm.clamp(min = 1e-8))

def pad_or_slice(t: Tensor, target_len: int) -> Tensor:
    if len(t) >= target_len:
        return t[:target_len]
    pad = t[-1:].expand(target_len - len(t), *t.shape[1:])
    return cat((t, pad), dim = 0)

def gather_clip_seq(seq: Tensor, clip_ends: Tensor, starts: Tensor, length: int) -> Tensor:
    offsets = torch.arange(length, device = seq.device)
    ends = clip_ends[starts.clamp(max = len(clip_ends) - 1)]
    indices = torch.minimum(einx.add('b, l -> b l', starts, offsets), rearrange(ends, 'b -> b 1'))
    return seq[indices]

def slice_clip_window(clip: dict, start: int, length: int, *, include_bodies: bool = True):
    end = start + length
    states = pad_or_slice(clip['states'][start:end], length)
    jmask = pad_or_slice(clip['joint_mask'][start:end], length)
    rmask = pad_or_slice(clip['root_mask'][start:end], length)

    has_bodies = include_bodies and exists(clip.get('body_pos'))
    bodies = pad_or_slice(clip['body_pos'][start:end], length) if has_bodies else None
    bmask = pad_or_slice(clip['body_mask'][start:end], length) if has_bodies else None
    return states, bodies, jmask, rmask, bmask

def expand_mask(mask, num_frames: int, dim: int) -> Tensor:
    t = to_float_tensor(mask)
    if t.ndim == 0:
        return t.expand(num_frames, dim).clone()
    if t.ndim == 1:
        if len(t) == num_frames and dim == 1:
            return rearrange(t, 't -> t 1').clone()
        return t.reshape(1, dim).expand(num_frames, dim).clone()

    assert t.shape == (num_frames, dim), f'expected mask shape {(num_frames, dim)}, got {tuple(t.shape)}'
    return t

def build_kin_mask(joint_mask: Tensor, root_mask: Tensor | None, has_free_root: bool) -> Tensor:
    if not has_free_root:
        return joint_mask

    rm = default(root_mask, torch.ones_like(joint_mask[..., :1]))
    root_pose = rm.expand(*rm.shape[:-1], 5)
    root_vel = rm.expand(*rm.shape[:-1], 6)
    return cat((root_pose, joint_mask, root_vel), dim = -1)

def resample_tensor(t: Tensor, num_dst: int, *, mode: str = 'linear') -> Tensor:
    if len(t) == num_dst or len(t) < 2:
        return t

    orig_shape = t.shape[1:]
    flat = rearrange(t, 't ... -> 1 (...) t')
    kwargs = dict(align_corners = True) if mode == 'linear' else dict()
    out = F.interpolate(flat, size = num_dst, mode = mode, **kwargs)
    return rearrange(out, '1 d t -> t d').reshape(num_dst, *orig_shape)

# heading-frame geometry & state layout

def to_heading_state(qpos: Tensor, qvel: Tensor) -> Tensor:
    q_world = l2norm(qpos[..., 3:7])
    w, x, y, z = q_world.unbind(dim = -1)
    half_yaw = -0.5 * torch.atan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))

    c = half_yaw.cos()
    s = half_yaw.sin()
    zeros = torch.zeros_like(c)

    q_yaw_inv = stack((c, zeros, zeros, s), dim = -1)
    q_tilt = quat_mul(q_yaw_inv, q_world)
    q_tilt = torch.where(q_tilt[..., :1] < 0., -q_tilt, q_tilt)

    lin_vel = world_to_heading_frame(q_world, qvel[..., :3])
    ang_vel = quat_rotate(q_tilt, qvel[..., 3:6])

    return cat((qpos[..., 2:3], q_tilt, qpos[..., 7:], lin_vel, ang_vel, qvel[..., 6:]), dim = -1).float()

def extract_sim_state(model, data, has_free_root: bool, extra_state_fn = None) -> Tensor:
    qpos = torch.from_numpy(data.qpos.copy()).float()
    qvel = torch.from_numpy(data.qvel.copy()).float()
    kin = to_heading_state(qpos, qvel) if has_free_root else cat((qpos, qvel), dim = -1)

    if not exists(extra_state_fn):
        return kin

    extra = to_float_tensor(extra_state_fn(model, data)).reshape(-1)
    return cat((kin, extra), dim = -1)

def unpack_state(raw: Tensor, robot: Robot, *, with_fk: bool = False) -> dict[str, Tensor]:
    if robot.has_free_root:
        nj = robot.nj
        kin_end = 11 + nj + (robot.nv - 6)
        root_z = raw[..., 0:1]
        root_quat = l2norm(raw[..., 1:5])
        joint_pos = raw[..., 5 : 5 + nj]
        root_pos = cat((raw.new_zeros((*raw.shape[:-1], 2)), root_z), dim = -1)

        out = dict(
            root_pos = root_pos,
            root_z = root_z,
            root_quat = root_quat,
            joint_pos = joint_pos,
            root_lin_vel = raw[..., 5 + nj : 8 + nj],
            root_ang_vel = raw[..., 8 + nj : 11 + nj],
            joint_vel = raw[..., 11 + nj : kin_end]
        )
        fk_q = cat((root_pos, root_quat, joint_pos), dim = -1)
    else:
        kin_end = robot.nq + robot.nv
        out = dict(
            joint_pos = raw[..., :robot.nq],
            joint_vel = raw[..., robot.nq : kin_end]
        )
        fk_q = out['joint_pos']

    if with_fk and exists(robot.fk):
        out['body_pos'] = robot.fk(qpos = fk_q)['body_pos']

    if raw.shape[-1] > kin_end:
        out['extra'] = raw[..., kin_end:]

    return out

def compose_clip_pose(clip: dict, frame: int, robot: Robot, nominal_qpos: Tensor) -> tuple[Tensor, Tensor]:
    qpos = nominal_qpos.clone()
    qvel = torch.zeros(robot.nv)

    jm = clip['joint_mask'][frame]
    rm = clip['root_mask'][frame]

    if robot.has_free_root and rm[0] > 0.5:
        qpos[:7] = clip['qpos'][frame, :7]
        qvel[:6] = clip['qvel'][frame, :6]

    obs_j = jm > 0.5
    if obs_j.any():
        v_start = 6 if robot.has_free_root else 0
        qpos[robot.q_start:][obs_j] = clip['qpos'][frame, robot.q_start:][obs_j]
        qvel[v_start:][obs_j] = clip['qvel'][frame, v_start:][obs_j]

    return qpos, qvel

# MuJoCo robot loading

@dataclass
class Robot:
    path: str
    model: Any
    has_free_root: bool
    fps: float
    substeps: int
    control: str
    use_pd: bool
    jnt_low: Tensor
    jnt_high: Tensor
    jnt_limited: Tensor
    joint_names: tuple[str, ...]
    qpos0: Tensor
    tracked_bodies: tuple[str, ...]
    body_joint_deps: tuple[tuple[int, ...], ...]
    fk: ForwardKinematics | None
    cfg: dict[str, Any]
    ctrl_low: Tensor | None = None
    ctrl_high: Tensor | None = None

    def __post_init__(self):
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu
        self.q_start = 7 if self.has_free_root else 0
        self.nj = self.nq - self.q_start
        self.nb = len(self.tracked_bodies)
        self.kin_dim = (11 + self.nj) if self.has_free_root else self.nq
        self.vel_start = (5 + self.nj) if self.has_free_root else self.nq
        self.dim_state = (11 + self.nj + (self.nv - 6)) if self.has_free_root else (self.nq + self.nv)
        self.dim_policy_mask = int(self.has_free_root) + self.nj + (self.nb if exists(self.fk) else 0)
        self.state_layout = 'heading' if self.has_free_root else 'qpos_qvel'

    def __getitem__(self, key: str):
        return getattr(self, key)

def auto_detect_leaf_bodies(model) -> tuple[str, ...]:
    mujoco = require('mujoco')
    parents = {int(model.body_parentid[b]) for b in range(1, model.nbody)}
    mobile = [False] * model.nbody

    for b in range(1, model.nbody):
        mobile[b] = bool(model.body_jntnum[b] > 0) or mobile[int(model.body_parentid[b])]

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        for b in range(1, model.nbody)
        if b not in parents and mobile[b]
    ]
    return tuple(n for n in names if exists(n))

def compute_body_joint_deps(
    model, tracked_names: tuple[str, ...], q_start: int, nj: int
) -> tuple[tuple[int, ...], ...]:
    mujoco = require('mujoco')
    body_joints: dict[int, list[int]] = {}

    for j in range(model.njnt):
        idx = int(model.jnt_qposadr[j]) - q_start
        if 0 <= idx < nj:
            body_joints.setdefault(int(model.jnt_bodyid[j]), []).append(idx)

    deps = []
    for name in tracked_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        chain = []
        while bid > 0:
            chain.extend(body_joints.get(bid, ()))
            bid = int(model.body_parentid[bid])
        deps.append(tuple(sorted(set(chain))))

    return tuple(deps)

def build_robot(
    robot_path: str | Path | Any,
    *,
    fps: float = 30.0,
    substeps: int = 8,
    control: str = 'auto',
    pd_kp: float | None = None,
    pd_kd: float | None = None,
    torque_gear: float = 0.32,
    max_torque: float | None = None,
    lock_joints = None,
    tracked_bodies = 'auto',
    joint_damping: float | None = None,
    joint_armature: float | None = None,
    disable_collisions: bool = False,
    factory_fn = None
) -> Robot:
    mujoco = require('mujoco')
    assert control in ('auto', 'native', 'pd', 'torque'), f'unknown control mode {control!r}'

    locked = set(parse_csv(lock_joints))
    kp = float(default(pd_kp, 40.0))
    kd = float(default(pd_kd, 0.5))
    tau = float(default(max_torque, 40.0))
    precompiled = spec = None

    if isinstance(robot_path, mujoco.MjModel):
        precompiled = robot_path
        label = '<MjModel>'
    elif isinstance(robot_path, mujoco.MjSpec):
        spec = robot_path
        label = '<MjSpec>'
    elif exists(factory_fn):
        built = factory_fn(robot_path, dict(fps = float(fps), substeps = int(substeps)))
        built = built[0] if isinstance(built, tuple) else built
        assert isinstance(built, (mujoco.MjModel, mujoco.MjSpec)), f'expected MjModel or MjSpec, got {type(built).__name__}'
        if isinstance(built, mujoco.MjModel):
            precompiled = built
        else:
            spec = built
        label = str(robot_path) if isinstance(robot_path, str) else getattr(robot_path, '__name__', '<callable>')
    else:
        path = Path(robot_path).expanduser().resolve()
        assert path.exists(), f'robot model not found: {path}'
        spec = mujoco.MjSpec.from_file(str(path))
        label = str(path)

    def configure_pd_actuator(act, lo: float, hi: float):
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = kp
        act.biasprm[:3] = [0., -kp, -kd]
        act.forcelimited = True
        act.forcerange = [lo, hi]
        act.ctrllimited = False

    if exists(precompiled):
        model = precompiled
        has_free_root = model.njnt > 0 and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
        model.opt.timestep = 1.0 / (float(fps) * int(substeps))
        use_pd = control == 'pd'
        resolved_control = 'pd' if use_pd else 'native'
    else:
        for joint in list(spec.joints):
            if joint.name in locked:
                spec.delete(joint)

        if disable_collisions:
            for geom in list(spec.geoms):
                geom.contype = 0
                geom.conaffinity = 0

        has_free_root = len(spec.joints) > 0 and spec.joints[0].type == mujoco.mjtJoint.mjJNT_FREE

        if has_free_root and not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms):
            spec.worldbody.add_geom(
                name = 'supermimic_floor',
                type = mujoco.mjtGeom.mjGEOM_PLANE,
                size = [0., 0., 0.05],
                rgba = [0.22, 0.28, 0.36, 1.0]
            )

        if len(spec.lights) == 0:
            spec.worldbody.add_light(
                pos = [0.5, -1.0, 2.0],
                dir = [-0.2, 0.4, -1.0],
                diffuse = [0.8, 0.8, 0.8],
                castshadow = False
            )

        had_actuators = len(spec.actuators) > 0
        if control == 'pd' or (exists(pd_kp) and control != 'native'):
            use_pd, resolved_control = True, 'pd'
        elif control == 'torque':
            use_pd, resolved_control = False, 'torque'
        elif control == 'native':
            assert had_actuators, 'robot has no actuators; use --control pd or --control torque'
            use_pd, resolved_control = False, 'native'
        else:
            use_pd = not had_actuators
            resolved_control = 'pd' if use_pd else 'native'

        if not had_actuators:
            for joint in spec.joints:
                if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                if use_pd:
                    act = spec.add_actuator(name = f'pd_{joint.name}', target = joint.name, trntype = mujoco.mjtTrn.mjTRN_JOINT)
                    configure_pd_actuator(act, -tau, tau)
                else:
                    spec.add_actuator(
                        name = f'motor_{joint.name}',
                        target = joint.name,
                        trntype = mujoco.mjtTrn.mjTRN_JOINT,
                        gear = [float(torque_gear), 0., 0., 0., 0., 0.],
                        ctrllimited = True,
                        ctrlrange = [-1., 1.]
                    )
        elif use_pd:
            for act in spec.actuators:
                rng = act.forcerange if float(act.forcerange[1]) > float(act.forcerange[0]) else act.ctrlrange
                lo, hi = (float(rng[0]), float(rng[1])) if float(rng[1]) > float(rng[0]) else (-tau, tau)
                configure_pd_actuator(act, lo, hi)

        spec.option.timestep = 1.0 / (float(fps) * int(substeps))
        model = spec.compile()

    dof_start = 6 if has_free_root else 0
    if exists(joint_damping):
        model.dof_damping[dof_start:] = float(joint_damping)
    if exists(joint_armature):
        model.dof_armature[dof_start:] = float(joint_armature)
    assert model.nu > 0, 'compiled MuJoCo model has 0 actuators'

    q_start = 7 if has_free_root else 0
    nj = model.nq - q_start
    jnt_low = torch.full((nj,), -1e6)
    jnt_high = torch.full((nj,), 1e6)
    jnt_limited = torch.zeros(nj, dtype = torch.bool)
    joint_names = [f'joint_{i}' for i in range(nj)]
    one_dof_types = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))

    for j in range(model.njnt):
        idx = int(model.jnt_qposadr[j]) - q_start
        if not (0 <= idx < nj):
            continue

        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if exists(jname):
            joint_names[idx] = jname

        if int(model.jnt_type[j]) in one_dof_types and bool(model.jnt_limited[j]):
            jnt_low[idx] = float(model.jnt_range[j, 0])
            jnt_high[idx] = float(model.jnt_range[j, 1])
            jnt_limited[idx] = True

    is_auto = isinstance(tracked_bodies, str) and tracked_bodies.strip().lower() == 'auto'
    tracked_names = auto_detect_leaf_bodies(model) if is_auto else parse_csv(tracked_bodies)
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f'body_{b}' for b in range(1, model.nbody)]

    for name in tracked_names:
        assert name in body_names, f'tracked body {name!r} not in model bodies: {body_names}'

    fk = None
    if tracked_names:
        try:
            fk = ForwardKinematics.from_mujoco(model, tracked_body_names = tracked_names)
        except AssertionError as err:
            print(f'[supermimic] warning: disabling FK body tracking ({err})', flush = True)
            tracked_names = ()

    body_deps = compute_body_joint_deps(model, tracked_names, q_start, nj) if tracked_names else ()
    qpos0 = torch.from_numpy(model.qpos0.copy()).float()
    if has_free_root:
        qpos0[3:7] = l2norm(qpos0[3:7])

    robot_cfg = dict(
        fps = float(fps),
        substeps = int(substeps),
        control = resolved_control,
        pd_kp = kp if use_pd else None,
        pd_kd = kd if use_pd else None,
        torque_gear = float(torque_gear),
        max_torque = tau if use_pd else None,
        lock_joints = tuple(sorted(locked)),
        tracked_bodies = tracked_names or 'none',
        joint_damping = joint_damping,
        joint_armature = joint_armature,
        disable_collisions = bool(disable_collisions)
    )

    return Robot(
        path = label,
        model = model,
        has_free_root = has_free_root,
        fps = float(fps),
        substeps = int(substeps),
        control = resolved_control,
        use_pd = use_pd,
        jnt_low = jnt_low,
        jnt_high = jnt_high,
        jnt_limited = jnt_limited,
        joint_names = tuple(joint_names),
        qpos0 = qpos0,
        tracked_bodies = tracked_names,
        body_joint_deps = body_deps,
        fk = fk,
        cfg = robot_cfg
    )

# reference mocap loader (full, partial-joint, and keypoint-only clips)

def resolve_mocap_files(mocap_spec) -> list[Path]:
    files = []

    for item in parse_csv(mocap_spec):
        if any(ch in item for ch in '*?['):
            matches = sorted(Path(x) for x in glob.glob(item, recursive = True))
            files.extend(p for p in matches if p.is_file() and p.suffix.lower() in MOCAP_EXTS)
            continue

        path = Path(item).expanduser()
        assert path.exists(), f'mocap path not found: {path}'

        if path.is_dir():
            children = sorted(path.rglob('*'))
            files.extend(f for f in children if f.is_file() and not f.name.startswith('.') and f.suffix.lower() in MOCAP_EXTS)
        else:
            files.append(path)

    unique = list(dict.fromkeys(f.resolve() for f in files))
    assert unique, f'no supported mocap files (.npz, .npy, .pt, .csv) found for {mocap_spec!r}'
    return unique

def extract_clips(obj, stem: str, quat_order: str) -> list[dict[str, Any]]:
    if isinstance(obj, (list, tuple)):
        return [row for i, item in enumerate(obj) for row in extract_clips(item, f'{stem}_{i}', quat_order)]

    if not isinstance(obj, dict):
        t = to_float_tensor(obj)
        if t.ndim == 3:
            return [dict(name = f'{stem}_{i}', qpos = t[i]) for i in range(len(t))]
        return [dict(name = stem, qpos = t)]

    d = obj
    clip_fps = float(to_float_tensor(d['fps']).item()) if 'fps' in d else None
    meta_keys = ('joint_names', 'joint_mask', 'root_mask', 'body_pos', 'body_names', 'body_mask')
    meta = {k: d[k] for k in meta_keys if k in d}

    qpos = qvel = root_pos = root_quat = joint_pos = None

    if 'qpos' in d:
        qpos = to_float_tensor(d['qpos'])
        qvel = to_float_tensor(d['qvel']) if 'qvel' in d else None
    else:
        jp_k = first_key(d, 'joint_pos', 'dof_pos', 'dof_positions', 'joint_positions')
        rp_k = first_key(d, 'root_pos', 'base_pos', 'trans', 'body_pos_root')
        rq_k = first_key(d, 'root_quat', 'base_quat', 'root_rot', 'base_rot')

        if exists(rp_k) and exists(rq_k):
            root_pos = to_float_tensor(d[rp_k])
            rq = to_float_tensor(d[rq_k])
            root_quat = rq[..., [3, 0, 1, 2]] if quat_order == 'xyzw' else rq

        if exists(jp_k):
            joint_pos = to_float_tensor(d[jp_k])
            jv_k = first_key(d, 'joint_vel', 'dof_vel', 'dof_velocities')
            rv_k = first_key(d, 'root_lin_vel', 'base_lin_vel')
            rw_k = first_key(d, 'root_ang_vel', 'base_ang_vel')

            if exists(jv_k) and not exists(root_pos):
                qvel = to_float_tensor(d[jv_k])
            elif exists(jv_k) and exists(rv_k) and exists(rw_k):
                qvel = cat((to_float_tensor(d[rv_k]), to_float_tensor(d[rw_k]), to_float_tensor(d[jv_k])), dim = -1)

    body_t = to_float_tensor(meta['body_pos']) if 'body_pos' in meta else None
    anchor = next((x for x in (qpos, joint_pos, root_pos, body_t) if exists(x)), None)

    if exists(anchor):
        only_bodies = not exists(qpos) and not exists(joint_pos) and not exists(root_pos)
        is_batched = (anchor.ndim == 4) if only_bodies else (anchor.ndim == 3)

        if not is_batched:
            return [dict(
                name = stem, fps = clip_fps, qpos = qpos, qvel = qvel,
                root_pos = root_pos, root_quat = root_quat, joint_pos = joint_pos, **meta
            )]

        names_key = first_key(d, 'names', 'clip_names')
        names = to_str_list(d[names_key]) if exists(names_key) else []

        rows = []
        for i in range(len(anchor)):
            sliced_meta = {k: (v[i] if torch.is_tensor(v) and v.ndim >= 3 else v) for k, v in meta.items()}
            rows.append(dict(
                name = names[i] if i < len(names) else f'{stem}_{i}',
                fps = clip_fps,
                qpos = qpos[i] if exists(qpos) else None,
                qvel = qvel[i] if exists(qvel) else None,
                root_pos = root_pos[i] if exists(root_pos) else None,
                root_quat = root_quat[i] if exists(root_quat) else None,
                joint_pos = joint_pos[i] if exists(joint_pos) else None,
                **sliced_meta
            ))
        return rows

    prefixed = sorted(k[:-5] for k in d if k.endswith('_qpos'))
    if prefixed:
        return [
            dict(
                name = f'{stem}:{p}',
                fps = clip_fps,
                qpos = to_float_tensor(d[f'{p}_qpos']),
                qvel = to_float_tensor(d[f'{p}_qvel']) if f'{p}_qvel' in d else None
            )
            for p in prefixed
        ]

    if 'clips' in d and isinstance(d['clips'], (list, tuple)):
        rows = extract_clips(d['clips'], stem, quat_order)
        for row in rows:
            row['fps'] = default(row.get('fps'), clip_fps)
        return rows

    raise ValueError(f'unrecognized mocap dict keys in {stem!r}: {sorted(d)}')

def resample_qpos(qpos: Tensor, src_fps: float, dst_fps: float, has_free_root: bool) -> Tensor:
    if abs(src_fps - dst_fps) < 1e-3 or len(qpos) < 2:
        return qpos

    q = qpos.clone()
    if has_free_root:
        q[:, 3:7] = fix_quat_sign(q[:, 3:7])

    num_dst = max(2, int(round((len(q) - 1) * dst_fps / src_fps)) + 1)
    out = resample_tensor(q, num_dst, mode = 'linear')
    if has_free_root:
        out[:, 3:7] = l2norm(out[:, 3:7])

    return out

def compute_qvel(qpos: Tensor, dt: float, has_free_root: bool) -> Tensor:
    if len(qpos) < 2:
        return torch.zeros((len(qpos), qpos.shape[-1] - int(has_free_root)))
    if not has_free_root:
        return torch.gradient(qpos, spacing = dt, dim = 0)[0]

    quats = fix_quat_sign(l2norm(qpos[:, 3:7]))
    q_prev = cat((quats[:1], quats[:-2], quats[-2:-1]), dim = 0)
    q_next = cat((quats[1:2], quats[2:], quats[-1:]), dim = 0)
    dt_span = torch.full((len(quats), 1), 2. * dt)
    dt_span[0] = dt_span[-1] = dt

    w_body = quat_sub(q_next, q_prev) / dt_span
    lin_v = torch.gradient(qpos[:, :3], spacing = dt, dim = 0)[0]
    jnt_v = torch.gradient(qpos[:, 7:], spacing = dt, dim = 0)[0]
    return cat((lin_v, w_body, jnt_v), dim = -1)

def load_mocap_clips(mocap_spec, robot: Robot, *, quat_order: str = 'wxyz', target_fps = None) -> list[dict[str, Any]]:
    assert quat_order in ('wxyz', 'xyzw'), f'unknown quat_order {quat_order!r}'
    raw_entries: list[dict[str, Any]] = []

    for path in resolve_mocap_files(mocap_spec):
        suf = path.suffix.lower()
        if suf in ('.pt', '.pth'):
            obj = torch.load(path, map_location = 'cpu', weights_only = False)
        else:
            np = require('numpy')
            if suf == '.npz':
                with np.load(path, allow_pickle = True) as data:
                    obj = {k: data[k] for k in data.files}
            elif suf == '.npy':
                obj = np.load(path)
            else:
                obj = np.loadtxt(path, delimiter = ',', dtype = np.float32)

        raw_entries.extend(extract_clips(obj, path.stem, quat_order))

    dst_fps = float(default(target_fps, robot.fps))
    jname_to_idx = {name: i for i, name in enumerate(robot.joint_names)}
    bname_to_idx = {name: i for i, name in enumerate(robot.tracked_bodies)}
    clips = []

    for entry in raw_entries:
        name = entry['name']
        qpos_in = entry.get('qpos')
        qvel = entry.get('qvel')
        rp_in = entry.get('root_pos')
        rq_in = entry.get('root_quat')
        jp_in = entry.get('joint_pos')
        bp_in = to_float_tensor(entry['body_pos']) if exists(entry.get('body_pos')) else None
        clip_fps = entry.get('fps')

        # resolve qpos, joint_mask, and root_mask

        if exists(qpos_in):
            assert qpos_in.ndim == 2 and qpos_in.shape[-1] == robot.nq and len(qpos_in) >= 2, (
                f'clip {name!r}: expected shape (T>=2, {robot.nq}), got {tuple(qpos_in.shape)} (qpos dim mismatch)'
            )
            num_frames = len(qpos_in)
            qpos = qpos_in.clone()
            jmask = expand_mask(entry['joint_mask'], num_frames, robot.nj) if exists(entry.get('joint_mask')) else torch.ones((num_frames, robot.nj))
            rmask = expand_mask(entry['root_mask'], num_frames, 1) if exists(entry.get('root_mask')) else torch.ones((num_frames, 1))
        else:
            anchor = next(x for x in (jp_in, rp_in, bp_in) if exists(x))
            num_frames = len(anchor)
            assert num_frames >= 2, f'clip {name!r}: expected at least 2 frames, got {num_frames}'

            qpos = robot.qpos0.unsqueeze(0).expand(num_frames, -1).clone()
            jmask = torch.zeros((num_frames, robot.nj))
            rmask = torch.zeros((num_frames, 1))

            if robot.has_free_root and exists(rp_in) and exists(rq_in):
                qpos[:, :3] = rp_in
                qpos[:, 3:7] = l2norm(rq_in)
                rmask.fill_(1.)

            if exists(jp_in):
                assert jp_in.ndim == 2, f'clip {name!r}: expected 2D joint_pos, got {tuple(jp_in.shape)}'
                if exists(entry.get('joint_names')):
                    jnames = to_str_list(entry['joint_names'])
                    assert len(jnames) == jp_in.shape[-1], f'clip {name!r}: joint_names length mismatch'
                    for col, jn in enumerate(jnames):
                        assert jn in jname_to_idx, f'clip {name!r}: unknown joint {jn!r}; robot has {robot.joint_names}'
                        idx = jname_to_idx[jn]
                        qpos[:, robot.q_start + idx] = jp_in[:, col]
                        jmask[:, idx] = 1.
                else:
                    assert jp_in.shape[-1] == robot.nj, (
                        f'clip {name!r}: expected {robot.nj} joints (or pass `joint_names`), got {jp_in.shape[-1]} (qpos dim mismatch)'
                    )
                    qpos[:, robot.q_start:] = jp_in
                    jmask.fill_(1.)

            if exists(entry.get('joint_mask')):
                jmask = expand_mask(entry['joint_mask'], num_frames, robot.nj)
            if exists(entry.get('root_mask')):
                rmask = expand_mask(entry['root_mask'], num_frames, 1)

        if not robot.has_free_root:
            rmask = torch.ones((num_frames, 1))
        else:
            qpos[:, 3:7] = l2norm(qpos[:, 3:7])

        # resolve body_pos and body_mask

        body_pos = bmask = None
        if exists(bp_in):
            assert exists(robot.fk) and robot.nb > 0, f'clip {name!r} provides body_pos but robot has no tracked_bodies'
            assert bp_in.ndim == 3 and bp_in.shape[-1] == 3, f'clip {name!r}: expected body_pos (T, K, 3), got {tuple(bp_in.shape)}'

            if exists(entry.get('body_names')):
                bnames = to_str_list(entry['body_names'])
                assert len(bnames) == bp_in.shape[1], f'clip {name!r}: body_names length mismatch'
                body_pos = torch.zeros((num_frames, robot.nb, 3))
                bmask = torch.zeros((num_frames, robot.nb))
                for col, bn in enumerate(bnames):
                    assert bn in bname_to_idx, f'clip {name!r}: unknown body {bn!r}; tracked_bodies={robot.tracked_bodies}'
                    bi = bname_to_idx[bn]
                    body_pos[:, bi] = bp_in[:, col]
                    bmask[:, bi] = 1.
            else:
                assert bp_in.shape[1] == robot.nb, f'clip {name!r}: expected {robot.nb} tracked bodies (or pass `body_names`), got {bp_in.shape[1]}'
                body_pos = bp_in.clone()
                bmask = torch.ones((num_frames, robot.nb))

            if exists(entry.get('body_mask')):
                bmask = expand_mask(entry['body_mask'], num_frames, robot.nb)

        # optional FPS resampling

        if exists(clip_fps) and abs(clip_fps - dst_fps) >= 1e-3:
            qpos = resample_qpos(qpos, clip_fps, dst_fps, robot.has_free_root)
            num_frames = len(qpos)
            qvel = None
            jmask = resample_tensor(jmask, num_frames, mode = 'nearest')
            rmask = resample_tensor(rmask, num_frames, mode = 'nearest')
            if exists(body_pos):
                body_pos = resample_tensor(body_pos, num_frames, mode = 'linear')
                bmask = resample_tensor(bmask, num_frames, mode = 'nearest')

        if not exists(qvel) or qvel.shape != (len(qpos), robot.nv):
            qvel = compute_qvel(qpos, 1.0 / dst_fps, robot.has_free_root)

        states = to_heading_state(qpos, qvel) if robot.has_free_root else cat((qpos, qvel), dim = -1)

        if not exists(body_pos) and exists(robot.fk):
            with torch.no_grad():
                fk_q = cat((torch.zeros((len(states), 2)), states[:, :5 + robot.nj]), dim = -1) if robot.has_free_root else qpos
                body_pos = robot.fk(qpos = fk_q)['body_pos'].float()

            if exists(entry.get('body_mask')):
                bmask = expand_mask(entry['body_mask'], len(states), robot.nb)
            else:
                bmask = torch.ones((len(states), robot.nb))
                for bi, dep_joints in enumerate(robot.body_joint_deps):
                    valid = rmask[:, 0] if robot.has_free_root else torch.ones(len(states))
                    if dep_joints:
                        valid = valid * jmask[:, list(dep_joints)].amin(dim = -1)
                    bmask[:, bi] = valid

        clips.append(dict(
            name = name,
            qpos = qpos,
            qvel = qvel,
            states = states,
            body_pos = body_pos,
            joint_mask = jmask,
            root_mask = rmask,
            body_mask = bmask,
            fps = dst_fps
        ))

    return clips

# motion bank & action scaling

@dataclass
class MotionBank:
    mode: str
    window_len: int
    enc_idx: Tensor
    kin_dim: int
    has_free_root: bool
    motion_mean: Tensor
    motion_std: Tensor
    body_mean: Tensor | None = None
    body_span: float = 0.0
    body_enc_scale: float = 1.0
    use_masks: bool = False
    has_partial: bool = False
    nominal_qpos: Tensor | None = None
    names: tuple[str, ...] = ()
    motion_feats: Tensor | None = None
    skill_ref_states: Tensor | None = None
    skill_ref_bodies: Tensor | None = None
    skill_ref_jmask: Tensor | None = None
    skill_ref_rmask: Tensor | None = None
    skill_ref_bmask: Tensor | None = None
    clip_lengths: Tensor | None = None
    clip_offsets: Tensor | None = None
    clip_ends: Tensor | None = None
    all_states: Tensor | None = None
    all_body_pos: Tensor | None = None
    all_joint_mask: Tensor | None = None
    all_root_mask: Tensor | None = None
    all_body_mask: Tensor | None = None

    @property
    def has_bodies(self) -> bool:
        return exists(self.body_mean)

    @property
    def dim_motion(self) -> int:
        return int(self.motion_feats.shape[-1]) if exists(self.motion_feats) else 0

    def __getitem__(self, key: str):
        return getattr(self, key)

    def to(self, device) -> MotionBank:
        tensor_fields = (
            'enc_idx', 'motion_mean', 'motion_std', 'body_mean', 'nominal_qpos', 'motion_feats',
            'skill_ref_states', 'skill_ref_bodies', 'skill_ref_jmask', 'skill_ref_rmask', 'skill_ref_bmask',
            'clip_ends', 'all_states', 'all_body_pos', 'all_joint_mask', 'all_root_mask', 'all_body_mask'
        )
        for field in tensor_fields:
            val = getattr(self, field)
            if torch.is_tensor(val):
                setattr(self, field, val.to(device))
        return self

    def gather(self, starts: Tensor, length: int):
        states = gather_clip_seq(self.all_states, self.clip_ends, starts, length)
        jmask = gather_clip_seq(self.all_joint_mask, self.clip_ends, starts, length)
        rmask = gather_clip_seq(self.all_root_mask, self.clip_ends, starts, length)
        bodies = gather_clip_seq(self.all_body_pos, self.clip_ends, starts, length) if self.has_bodies else None
        bmask = gather_clip_seq(self.all_body_mask, self.clip_ends, starts, length) if self.has_bodies else None
        return states, bodies, jmask, rmask, bmask

    def encode(
        self,
        states_w: Tensor,
        body_w: Tensor | None = None,
        joint_mask_w: Tensor | None = None,
        root_mask_w: Tensor | None = None,
        body_mask_w: Tensor | None = None
    ) -> Tensor:
        unbatched = states_w.ndim == 2
        s = rearrange(states_w, 't d -> 1 t d') if unbatched else states_w
        idx = self.enc_idx.to(s.device)
        kin_norm = (s[:, idx, :self.kin_dim] - self.motion_mean.to(s)) / self.motion_std.to(s)

        if self.use_masks:
            if exists(joint_mask_w):
                jm = rearrange(joint_mask_w, 't d -> 1 t d') if unbatched else joint_mask_w
                rm = rearrange(root_mask_w, 't d -> 1 t d') if (exists(root_mask_w) and unbatched) else root_mask_w
                km = build_kin_mask(jm.to(s), rm.to(s) if exists(rm) else None, self.has_free_root)[:, idx]
            else:
                km = torch.ones_like(kin_norm)

            parts = [
                rearrange(kin_norm * km, 'b t d -> b (t d)'),
                rearrange(km, 'b t d -> b (t d)')
            ]
        else:
            parts = [rearrange(kin_norm, 'b t d -> b (t d)')]

        if exists(body_w) and exists(self.body_mean):
            b = rearrange(body_w, 't k d -> 1 t k d') if unbatched else body_w
            b_norm = (b[:, idx] - self.body_mean[0].to(b)) / self.body_enc_scale

            if self.use_masks:
                if exists(body_mask_w):
                    bm = (rearrange(body_mask_w, 't k -> 1 t k') if unbatched else body_mask_w).to(b)[:, idx]
                else:
                    bm = torch.ones(b_norm.shape[:-1], device = b.device, dtype = b.dtype)

                masked_b = einx.multiply('b t k d, b t k -> b t k d', b_norm, bm)
                parts.extend((
                    rearrange(masked_b, 'b t k d -> b (t k d)'),
                    rearrange(bm, 'b t k -> b (t k)')
                ))
            else:
                parts.append(rearrange(b_norm, 'b t k d -> b (t k d)'))

        out = cat(parts, dim = -1)
        return out[0] if unbatched else out

    def policy_mask(
        self,
        joint_mask: Tensor,
        root_mask: Tensor | None = None,
        body_mask: Tensor | None = None
    ) -> Tensor:
        parts = []
        if self.has_free_root:
            parts.append(default(root_mask, torch.ones_like(joint_mask[..., :1])).to(joint_mask))
        parts.append(joint_mask)
        if self.has_bodies and exists(body_mask):
            parts.append(body_mask.to(joint_mask))
        return cat(parts, dim = -1)

    def to_metadata(self) -> dict[str, Any]:
        return dict(
            mode = self.mode,
            window_len = self.window_len,
            enc_idx = self.enc_idx.cpu().tolist(),
            kin_dim = self.kin_dim,
            use_masks = self.use_masks,
            motion_mean = self.motion_mean.cpu().tolist(),
            motion_std = self.motion_std.cpu().tolist(),
            body_mean = self.body_mean.cpu().tolist() if exists(self.body_mean) else None,
            body_enc_scale = self.body_enc_scale
        )

    @classmethod
    def from_metadata(cls, meta: dict[str, Any], robot: Robot) -> MotionBank:
        kin_dim = int(meta.get('kin_dim', robot.kin_dim))
        m_mean = as_tensor(meta['motion_mean'], dtype = torch.float32) if exists(meta.get('motion_mean')) else torch.zeros(kin_dim)
        m_std = as_tensor(meta['motion_std'], dtype = torch.float32) if exists(meta.get('motion_std')) else torch.ones(kin_dim)
        b_mean = as_tensor(meta['body_mean'], dtype = torch.float32) if exists(meta.get('body_mean')) else None

        return cls(
            mode = meta.get('mode', 'clips'),
            window_len = int(meta.get('window_len', 16)),
            enc_idx = as_tensor(meta.get('enc_idx', [0, 4, 8, 12, 15]), dtype = torch.long),
            kin_dim = kin_dim,
            has_free_root = robot.has_free_root,
            motion_mean = m_mean,
            motion_std = m_std,
            body_mean = b_mean,
            body_enc_scale = float(meta.get('body_enc_scale', 1.0)),
            use_masks = bool(meta.get('use_masks', False))
        )

def masked_mean_std(vals: Tensor, mask: Tensor, min_std: Tensor) -> tuple[Tensor, Tensor]:
    w = mask.to(vals)
    cnt = w.sum(dim = 0)
    mean = (vals * w).sum(dim = 0) / cnt.clamp(min = 1.)
    var = ((vals - mean).square() * w).sum(dim = 0) / cnt.clamp(min = 1.)
    std = var.clamp(min = 0.).sqrt().clamp(min = min_std)
    return torch.where(cnt > 0., mean, torch.zeros_like(mean)), std

def compute_nominal_qpos(clips: list[dict], robot: Robot) -> Tensor:
    qpos0 = robot.qpos0.clone()
    all_q = cat([c['qpos'] for c in clips], dim = 0)
    all_jm = cat([c['joint_mask'] for c in clips], dim = 0)
    all_rm = cat([c['root_mask'] for c in clips], dim = 0)[:, 0] > 0

    if robot.has_free_root and all_rm.any():
        qpos0[2] = all_q[all_rm, 2].median()
        qpos0[3:7] = as_tensor([1., 0., 0., 0.])

    for j in range(robot.nj):
        valid = all_jm[:, j] > 0
        if valid.any():
            qpos0[robot.q_start + j] = all_q[valid, robot.q_start + j].median()

    return qpos0

def build_motion_bank(
    clips: list[dict],
    robot: Robot,
    *,
    mode: str = 'auto',
    imag_horizon: int = 16,
    clip_len: int | None = None,
    clip_stride: int | None = None,
    enc_frames: int = 5,
    mask_dropout: float = 0.0
) -> MotionBank:
    assert mode in ('auto', 'skills', 'clips'), f'unknown mode {mode!r}'
    if mode == 'auto':
        med_len = float(torch.tensor([len(c['qpos']) for c in clips], dtype = torch.float32).median().item())
        mode = 'skills' if (not robot.has_free_root and len(clips) >= 2 and med_len <= 2 * imag_horizon) else 'clips'

    has_bodies = exists(robot.fk)
    window_len = max(imag_horizon, int(default(clip_len, imag_horizon)))
    enc_idx = torch.linspace(0, window_len - 1, min(enc_frames, window_len)).round().long().unique()

    all_states = cat([c['states'] for c in clips], dim = 0)
    all_jmask = cat([c['joint_mask'] for c in clips], dim = 0)
    all_rmask = cat([c['root_mask'] for c in clips], dim = 0)
    all_bmask = cat([c['body_mask'] for c in clips], dim = 0) if has_bodies else None
    all_kmask = build_kin_mask(all_jmask, all_rmask, robot.has_free_root)

    has_partial = bool(
        (all_jmask < 0.5).any()
        or (robot.has_free_root and (all_rmask < 0.5).any())
        or (exists(all_bmask) and (all_bmask < 0.5).any())
    )
    use_masks = has_partial or (mask_dropout > 0.)

    if robot.has_free_root:
        min_mot = as_tensor([0.05] + [0.08] * 4 + [0.15] * robot.nj + [0.30] * 3 + [0.50] * 3, dtype = torch.float32)
    else:
        min_mot = torch.full((robot.nq,), 0.15)
    mot_mean, mot_std = masked_mean_std(all_states[:, :robot.kin_dim], all_kmask, min_mot)

    if has_bodies:
        all_bp = cat([c['body_pos'] for c in clips], dim = 0)
        bm_flat = all_bmask > 0.5
        if bm_flat.any():
            valid_pts = all_bp[bm_flat]
            body_mean = rearrange(valid_pts.mean(dim = 0), 'd -> 1 1 d')
            body_span = float((valid_pts.amax(dim = 0) - valid_pts.amin(dim = 0)).norm().item())
        else:
            body_mean = torch.zeros(1, 1, 3)
            body_span = 0.2
        body_enc_scale = max(0.05, 0.25 * body_span)
    else:
        all_bp = body_mean = None
        body_span = 0.0
        body_enc_scale = 1.0

    bank = MotionBank(
        mode = mode,
        window_len = window_len,
        enc_idx = enc_idx,
        kin_dim = robot.kin_dim,
        has_free_root = robot.has_free_root,
        motion_mean = mot_mean,
        motion_std = mot_std,
        body_mean = body_mean,
        body_span = body_span,
        body_enc_scale = body_enc_scale,
        use_masks = use_masks,
        has_partial = has_partial,
        nominal_qpos = compute_nominal_qpos(clips, robot)
    )

    prim_names, prim_feats = [], []
    prim_refs_s, prim_refs_b = [], []
    prim_refs_jm, prim_refs_rm, prim_refs_bm = [], [], []

    if mode == 'skills':
        rs = min(max(0, window_len - imag_horizon), 4) if window_len > imag_horizon else 0
        for clip in clips:
            s_w, b_w, jm_w, rm_w, bm_w = slice_clip_window(clip, 0, window_len, include_bodies = has_bodies)
            prim_names.append(clip['name'])
            prim_feats.append(bank.encode(s_w, b_w, jm_w, rm_w, bm_w))
            prim_refs_s.append(s_w[rs : rs + imag_horizon])
            prim_refs_jm.append(jm_w[rs : rs + imag_horizon])
            prim_refs_rm.append(rm_w[rs : rs + imag_horizon])
            if has_bodies:
                prim_refs_b.append(b_w[rs : rs + imag_horizon])
                prim_refs_bm.append(bm_w[rs : rs + imag_horizon])
    else:
        stride = max(1, int(default(clip_stride, max(1, window_len // 2))))
        for clip in clips:
            n = len(clip['qpos'])
            starts = list(range(0, max(1, n - window_len + 1), stride))
            if n > window_len and starts[-1] != n - window_len:
                starts.append(n - window_len)

            for st in starts:
                s_w, b_w, jm_w, rm_w, bm_w = slice_clip_window(clip, st, window_len, include_bodies = has_bodies)
                prim_names.append(f'{clip["name"]}@{st}' if len(starts) > 1 else clip['name'])
                prim_feats.append(bank.encode(s_w, b_w, jm_w, rm_w, bm_w))

    clip_lengths = torch.tensor([len(c['qpos']) for c in clips], dtype = torch.long)
    clip_offsets = cat((torch.zeros(1, dtype = torch.long), clip_lengths[:-1].cumsum(dim = 0)))

    bank.names = tuple(prim_names)
    bank.motion_feats = stack(prim_feats, dim = 0)
    bank.skill_ref_states = stack(prim_refs_s, dim = 0) if prim_refs_s else None
    bank.skill_ref_bodies = stack(prim_refs_b, dim = 0) if prim_refs_b else None
    bank.skill_ref_jmask = stack(prim_refs_jm, dim = 0) if prim_refs_jm else None
    bank.skill_ref_rmask = stack(prim_refs_rm, dim = 0) if prim_refs_rm else None
    bank.skill_ref_bmask = stack(prim_refs_bm, dim = 0) if prim_refs_bm else None
    bank.clip_lengths = clip_lengths
    bank.clip_offsets = clip_offsets
    bank.clip_ends = cat([
        torch.full((int(n),), int(off + n - 1), dtype = torch.long)
        for n, off in zip(clip_lengths, clip_offsets)
    ])
    bank.all_states = all_states
    bank.all_body_pos = all_bp
    bank.all_joint_mask = all_jmask
    bank.all_root_mask = all_rmask
    bank.all_body_mask = all_bmask
    return bank

def compute_action_scaling(clips: list[dict], robot: Robot):
    model = robot.model
    nu = robot.nu
    all_qpos = cat([c['qpos'] for c in clips], dim = 0)
    all_jmask = cat([c['joint_mask'] for c in clips], dim = 0)

    offset = torch.zeros(nu)
    scale = torch.ones(nu)
    ctrl_low = torch.full((nu,), -1e6)
    ctrl_high = torch.full((nu,), 1e6)

    for a in range(nu):
        if robot.use_pd:
            j = int(model.actuator_trnid[a, 0])
            qadr = int(model.jnt_qposadr[j])
            jidx = qadr - robot.q_start
            valid = all_jmask[:, jidx] > 0.5 if 0 <= jidx < robot.nj else torch.ones(len(all_qpos), dtype = torch.bool)
            qm = all_qpos[valid, qadr] if valid.any() else robot.qpos0[qadr : qadr + 1]

            q_def = float(qm.median().item())
            dev = max(0.35, 1.35 * float((qm - q_def).abs().max().item()))

            if bool(model.jnt_limited[j]):
                lo = float(model.jnt_range[j, 0])
                hi = float(model.jnt_range[j, 1])
                offset[a] = q_def
                scale[a] = min(dev, max(abs(hi - q_def), abs(q_def - lo), 0.1))
            else:
                lo = q_def - dev
                hi = q_def + dev
                offset[a] = q_def
                scale[a] = dev

            ctrl_low[a] = lo
            ctrl_high[a] = hi
        elif bool(model.actuator_ctrllimited[a]):
            lo = float(model.actuator_ctrlrange[a, 0])
            hi = float(model.actuator_ctrlrange[a, 1])
            offset[a] = 0.5 * (lo + hi)
            scale[a] = max(0.5 * (hi - lo), 1e-6)
            ctrl_low[a] = lo
            ctrl_high[a] = hi

    return offset, scale, ctrl_low, ctrl_high

# parallel MuJoCo envs, rollout & video rendering

def is_fallen(state: Tensor, fall_cfg: dict | None) -> bool:
    if not torch.isfinite(state).all():
        return True
    if not exists(fall_cfg):
        return False

    cos_tilt = 1.0 - 2.0 * float(state[2] ** 2 + state[3] ** 2)
    return float(state[0]) < fall_cfg['min_height'] or cos_tilt < 1.0 - fall_cfg['max_tilt']

def clip_tracking_metrics(
    states_run: list[Tensor], ref_clip: dict, robot: Robot, device: torch.device
) -> tuple[float | None, float | None]:
    n = len(states_run)
    pred = unpack_state(stack(states_run, dim = 0).to(device), robot, with_fk = True)

    jmask = ref_clip['joint_mask'][1 : n + 1].to(device)
    q_rmse = None
    if (jmask > 0.5).any():
        ref_j = ref_clip['qpos'][1 : n + 1, robot.q_start:].to(device)
        diff_sq = (pred['joint_pos'] - ref_j).square() * jmask
        q_rmse = float((diff_sq.sum() / jmask.sum().clamp(min = 1.)).sqrt().item())

    ee_mm = None
    if exists(robot.fk) and exists(ref_clip['body_pos']):
        bmask = ref_clip['body_mask'][1 : n + 1].to(device)
        if (bmask > 0.5).any():
            ref_b = ref_clip['body_pos'][1 : n + 1].to(device)
            dist = (pred['body_pos'] - ref_b).norm(dim = -1) * bmask
            ee_mm = float((dist.sum() / bmask.sum().clamp(min = 1.)).item() * 1000.0)

    return q_rmse, ee_mm

class MocapCollectEnvs:
    def __init__(
        self,
        robot: Robot,
        clips: list[dict],
        bank: MotionBank,
        action_offset: Tensor,
        action_scale: Tensor,
        ctrl_low: Tensor,
        ctrl_high: Tensor,
        fall_cfg = None,
        *,
        extra_state_fn = None,
        num_envs: int = 16,
        horizon: int = 96,
        switch_every: int = 24,
        rsi_pos_noise: float = 0.03,
        rsi_vel_noise: float = 0.10
    ):
        mujoco = require('mujoco')
        self.robot = robot
        self.clips = clips
        self.bank = bank
        self.fall_cfg = fall_cfg
        self.extra_state_fn = extra_state_fn

        self.action_offset = action_offset.cpu()
        self.action_scale = action_scale.cpu()
        self.ctrl_low = ctrl_low.cpu()
        self.ctrl_high = ctrl_high.cpu()

        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.switch_every = int(switch_every)
        self.rsi_pos_noise = float(rsi_pos_noise)
        self.rsi_vel_noise = float(rsi_vel_noise)

        self.num_prims = len(bank.names)
        weights = bank.clip_lengths.float().sqrt() if bank.has_partial else bank.clip_lengths.float()
        self.clip_probs = weights / weights.sum()
        self.nominal_qpos = default(bank.nominal_qpos, robot.qpos0).cpu()
        self.full_clip_ids = [
            i for i, c in enumerate(clips)
            if (c['joint_mask'] > 0.5).all() and (not robot.has_free_root or (c['root_mask'] > 0.5).all())
        ]

        self.datas = [mujoco.MjData(robot.model) for _ in range(self.num_envs)]
        self.clip_ids = torch.zeros(self.num_envs, dtype = torch.long)
        self.frames = torch.zeros(self.num_envs, dtype = torch.long)
        self.prim_ids = torch.zeros(self.num_envs, dtype = torch.long)
        self.steps = torch.zeros(self.num_envs, dtype = torch.long)

        sample = extract_sim_state(robot.model, self.datas[0], robot.has_free_root, extra_state_fn)
        self.dim_state = int(sample.shape[-1])
        self.states = torch.zeros((self.num_envs, self.dim_state))

        for i in range(self.num_envs):
            self.reset(i)

    def global_frames(self) -> Tensor:
        return self.bank.clip_offsets[self.clip_ids] + self.frames

    def reset(self, index: int):
        mujoco = require('mujoco')
        rob = self.robot
        cid = int(torch.multinomial(self.clip_probs, 1).item())
        clip = self.clips[cid]
        data = self.datas[index]

        margin = 1 if self.bank.mode == 'skills' else min(len(clip['qpos']) - 1, 20)
        frame = int(torch.randint(max(1, len(clip['qpos']) - margin), ()).item())

        mujoco.mj_resetData(rob.model, data)
        jm = clip['joint_mask'][frame]
        rm = clip['root_mask'][frame]
        has_kin = bool((jm > 0.5).any() or (rob.has_free_root and rm[0] > 0.5))

        if has_kin:
            qpos, qvel = compose_clip_pose(clip, frame, rob, self.nominal_qpos)
        elif self.full_clip_ids:
            ref_c = self.clips[self.full_clip_ids[int(torch.randint(len(self.full_clip_ids), ()).item())]]
            rf = int(torch.randint(len(ref_c['qpos']), ()).item())
            qpos = ref_c['qpos'][rf].clone()
            qvel = ref_c['qvel'][rf].clone()
        else:
            qpos = self.nominal_qpos.clone()
            qvel = torch.zeros(rob.nv)

        q_j = qpos[rob.q_start:] + torch.randn(rob.nj) * self.rsi_pos_noise
        lim = rob.jnt_limited
        q_j[lim] = q_j[lim].clamp(rob.jnt_low[lim], rob.jnt_high[lim])
        qpos[rob.q_start:] = q_j

        if rob.has_free_root:
            qpos[2] += float(torch.empty(()).uniform_(-0.015, 0.015))
            qvel[:6] += torch.randn(6) * self.rsi_vel_noise
            qvel[6:] += torch.randn(rob.nv - 6) * (self.rsi_vel_noise * 2.0)
        else:
            qvel = qvel * float(torch.empty(()).uniform_(0.3, 1.0))

        data.qpos[:] = qpos.numpy()
        data.qvel[:] = qvel.numpy()
        data.ctrl[:] = self.action_offset.clamp(self.ctrl_low, self.ctrl_high).numpy()
        mujoco.mj_forward(rob.model, data)

        self.clip_ids[index] = cid
        self.frames[index] = frame
        self.steps[index] = 0
        self.prim_ids[index] = int(torch.randint(self.num_prims, ()).item())
        self.states[index] = extract_sim_state(rob.model, data, rob.has_free_root, self.extra_state_fn)

    def step(self, actions_norm: Tensor) -> tuple[Tensor, Tensor]:
        mujoco = require('mujoco')
        rob = self.robot
        actions_cpu = actions_norm.detach().cpu().clamp(-1., 1.)
        ctrls = (self.action_offset + self.action_scale * actions_cpu).clamp(self.ctrl_low, self.ctrl_high).numpy()

        next_states = torch.empty_like(self.states)
        dones = torch.zeros(self.num_envs, dtype = torch.bool)

        for i, data in enumerate(self.datas):
            data.ctrl[:] = ctrls[i]
            mujoco.mj_step(rob.model, data, nstep = rob.substeps)

            ns = extract_sim_state(rob.model, data, rob.has_free_root, self.extra_state_fn)
            next_states[i] = ns
            self.steps[i] += 1
            if self.bank.mode == 'clips':
                self.frames[i] += 1

            end_of_clip = self.bank.mode == 'clips' and int(self.frames[i]) >= int(self.bank.clip_lengths[self.clip_ids[i]]) - 1
            if is_fallen(ns, self.fall_cfg) or int(self.steps[i]) >= self.horizon or end_of_clip:
                dones[i] = True
                self.reset(i)
            else:
                self.states[i] = ns
                if self.bank.mode == 'skills' and self.num_prims > 1 and divisible_by(int(self.steps[i]), self.switch_every):
                    cur = int(self.prim_ids[i])
                    nxt = int(torch.randint(self.num_prims - 1, ()).item())
                    self.prim_ids[i] = nxt + int(nxt >= cur)

        return next_states, dones

@torch.no_grad()
def rollout_policy(
    robot: Robot,
    policy: LatentPolicy,
    device: torch.device,
    num_steps: int,
    *,
    ref_clip = None,
    z_fixed = None,
    mask_fixed = None,
    use_policy_mask: bool = True,
    random_latent: bool = False,
    bank: MotionBank | None = None,
    fall_cfg = None,
    extra_state_fn = None,
    obs_fn = identity,
    switch_every: int = 24,
    command_every: int = 8
):
    mujoco = require('mujoco')
    data = mujoco.MjData(robot.model)
    mujoco.mj_resetData(robot.model, data)

    nominal = default(bank.nominal_qpos.cpu() if exists(bank) and exists(bank.nominal_qpos) else None, robot.qpos0.cpu())
    if exists(ref_clip):
        q0, v0 = compose_clip_pose(ref_clip, 0, robot, nominal)
        data.qpos[:] = q0.numpy()
        data.qvel[:] = v0.numpy()
    elif exists(bank):
        mm = bank.motion_mean.cpu()
        if robot.has_free_root:
            data.qpos[2] = max(0.3, float(mm[0]))
            data.qpos[3:7] = [1., 0., 0., 0.]
            data.qpos[7:] = mm[5 : 5 + robot.nj].numpy()
        else:
            data.qpos[:] = mm[:robot.nq].numpy()
    mujoco.mj_forward(robot.model, data)

    cache = policy.init_cache(1, device = device)
    states_run = []
    qposes = [data.qpos.copy()]
    survived = 0
    z_cur = z_fixed
    m_cur = mask_fixed

    for t in range(num_steps):
        if random_latent and (divisible_by(t, switch_every) or not exists(z_cur)):
            z_cur = policy.sample_latent(1, device = device)
            m_cur = None
        elif exists(ref_clip) and not exists(z_fixed) and (divisible_by(t, command_every) or not exists(z_cur)):
            s_w, b_w, jm_w, rm_w, bm_w = slice_clip_window(ref_clip, t, bank.window_len)
            feat = rearrange(bank.encode(s_w, b_w, jm_w, rm_w, bm_w).to(device), 'd -> 1 d')
            z_cur = policy.encode(feat, add_noise = False)
            if policy.dim_mask > 0 and use_policy_mask:
                m_cur = rearrange(
                    bank.policy_mask(jm_w[0], rm_w[0], bm_w[0] if exists(bm_w) else None).to(device),
                    'd -> 1 d'
                )

        st_t = rearrange(extract_sim_state(robot.model, data, robot.has_free_root, extra_state_fn).to(device), 'd -> 1 d')
        ctrl, cache = policy.step(obs_fn(st_t), z_cur, cache, mask = m_cur if (policy.dim_mask > 0 and use_policy_mask) else None)
        ctrl_cpu = ctrl[0].cpu()
        if exists(robot.ctrl_low):
            ctrl_cpu = ctrl_cpu.clamp(robot.ctrl_low.cpu(), robot.ctrl_high.cpu())

        data.ctrl[:] = ctrl_cpu.numpy()
        mujoco.mj_step(robot.model, data, nstep = robot.substeps)

        ns = extract_sim_state(robot.model, data, robot.has_free_root, extra_state_fn)
        states_run.append(ns)
        qposes.append(data.qpos.copy())
        if is_fallen(ns, fall_cfg):
            break
        survived += 1

    return states_run, qposes, survived

@torch.no_grad()
def evaluate_policy(
    robot: Robot,
    clips: list[dict],
    bank: MotionBank,
    policy: LatentPolicy,
    device: torch.device,
    fall_cfg = None,
    *,
    extra_state_fn = None,
    obs_fn = identity,
    max_eval_clips: int = 4,
    command_every: int = 8
) -> dict[str, float]:
    policy.eval()
    z_bank = policy.encode(bank.motion_feats.to(device), add_noise = False)

    min_z_dist = 0.0
    if len(z_bank) >= 2:
        pdist = torch.cdist(z_bank, z_bank)
        pdist.fill_diagonal_(float('inf'))
        min_z_dist = float(pdist.min().item())

    q_errs, ee_errs, survs = [], [], []
    eval_ids = torch.linspace(0, len(clips) - 1, min(len(clips), max_eval_clips)).long().tolist()

    for cid in eval_ids:
        clip = clips[cid]
        horizon = len(clip['qpos']) - 1

        m_fix = None
        if bank.mode == 'skills' and policy.dim_mask > 0:
            bm0 = clip['body_mask'][0] if exists(clip['body_mask']) else None
            m_fix = rearrange(bank.policy_mask(clip['joint_mask'][0], clip['root_mask'][0], bm0).to(device), 'd -> 1 d')

        states_run, _, survived = rollout_policy(
            robot,
            policy,
            device,
            horizon,
            ref_clip = clip,
            z_fixed = z_bank[cid : cid + 1] if bank.mode == 'skills' else None,
            mask_fixed = m_fix,
            bank = bank,
            fall_cfg = fall_cfg,
            extra_state_fn = extra_state_fn,
            obs_fn = obs_fn,
            command_every = command_every
        )
        if states_run:
            q_rmse, ee_mm = clip_tracking_metrics(states_run, clip, robot, device)
            if exists(q_rmse):
                q_errs.append(q_rmse)
            if exists(ee_mm):
                ee_errs.append(ee_mm)
        survs.append(float(survived / max(1, horizon)))

    return dict(
        q_rmse = sum(q_errs) / len(q_errs) if q_errs else 0.0,
        ee_mm = sum(ee_errs) / len(ee_errs) if ee_errs else 0.0,
        surv = sum(survs) / len(survs) if survs else 0.0,
        min_z_dist = min_z_dist
    )

def render_qpos_video(model, qposes, out_video: Path, fps: float, has_free_root: bool, size: int = 480):
    mujoco = require('mujoco')
    imageio = require('imageio')

    renderer = mujoco.Renderer(model, height = size, width = size)
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.distance = max(0.30, 2.2 * float(model.stat.extent))
    cam.elevation = -18.0
    cam.azimuth = 135.0

    frames = []
    for q in qposes:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [float(q[0]), float(q[1]), float(max(0.15, q[2]))] if has_free_root else model.stat.center
        renderer.update_scene(data, camera = cam)
        frames.append(renderer.render())
    renderer.close()

    out_video.parent.mkdir(parents = True, exist_ok = True)
    if out_video.suffix.lower() == '.webp':
        from PIL import Image
        Image.fromarray(frames[0]).save(
            out_video,
            save_all = True,
            append_images = [Image.fromarray(f) for f in frames[1:]],
            duration = int(1000 / max(1.0, fps)),
            loop = 0,
            quality = 85
        )
    else:
        imageio.mimsave(
            out_video,
            frames,
            fps = fps,
            codec = 'libx264',
            pixelformat = 'yuv420p',
            macro_block_size = 1,
            output_params = ['-movflags', '+faststart']
        )
