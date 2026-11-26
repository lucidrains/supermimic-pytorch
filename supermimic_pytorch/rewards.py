from __future__ import annotations
from collections import Counter, namedtuple
from typing import Callable, Sequence as Seq

import torch
from torch import Tensor, as_tensor, cat, is_tensor, stack
from torch.nn import Module, ModuleList
import torch.nn.functional as F

import einx
from einops import einsum, rearrange, reduce, repeat
from torch_einops_utils import orthogonal_component, safe_cat
from torch_einops_utils.shape import is_shape, shape

from supermimic_pytorch.supermimic import (
    State,
    action_smoothness_penalty,
    cast_tuple,
    compact,
    default,
    exists,
    select_key
)

# helpers & feature selection

def l2norm(t: Tensor, eps = 1e-8) -> Tensor:
    return F.normalize(t, dim = -1, eps = eps)

def cross(a: Tensor, b: Tensor) -> Tensor:
    return torch.linalg.cross(a, b, dim = -1)

def is_static_ref(pred: Tensor, target: Tensor) -> bool:
    return pred.ndim >= 3 and shape(pred, '[b] t [...rest]') == target

def align_time(pred: Tensor, target: Tensor) -> Tensor:
    return rearrange(target, 'b ... -> b 1 ...') if is_static_ref(pred, target) else target

def select(tree: State, key) -> Tensor:
    if not exists(key):
        return tree
    if callable(key):
        return key(tree)
    if isinstance(key, str):
        return select_key(tree, key)
    if isinstance(key, tuple):
        for step in key:
            tree = select(tree, step)
        return tree

    assert is_tensor(tree), f'cannot index non-tensor state with {key!r}'
    return rearrange(tree[..., key], '... -> ... 1') if isinstance(key, int) else tree[..., key]

def select_pair(states: State, refs: State, key, ref_key = None) -> tuple[Tensor, Tensor]:
    pred = select(states, key)

    if exists(ref_key):
        target = select(refs, ref_key)
    elif is_tensor(refs) and (refs.shape == pred.shape or is_static_ref(pred, refs)):
        target = refs
    else:
        target = select(refs, key)

    return pred, align_time(pred, target)

# differentiable geometry (SO(3), wxyz quaternions, SO(2) angles)

def quat_mul(q1: Tensor, q2: Tensor) -> Tensor:
    q1, q2 = torch.broadcast_tensors(q1, q2)
    w1, v1 = q1[..., :1], q1[..., 1:]
    w2, v2 = q2[..., :1], q2[..., 1:]

    w = w1 * w2 - reduce(v1 * v2, '... d -> ... 1', 'sum')
    v = w1 * v2 + w2 * v1 + cross(v1, v2)
    return cat((w, v), dim = -1)

def axis_angle_to_quat(axis: Tensor, angle: Tensor) -> Tensor:
    half = 0.5 * angle
    v = einsum(l2norm(axis), half.sin(), '... d, ... -> ... d')
    w = rearrange(half.cos().expand(shape(v, '[...] d').shape), '... -> ... 1')
    return cat((w, v), dim = -1)

def quat_rotate(q: Tensor, v: Tensor, inverse = False) -> Tensor:
    q = l2norm(q)
    w, u = q[..., :1], q[..., 1:]
    u, v = torch.broadcast_tensors(u, v)

    t = 2. * cross(u, v)
    return v + (-w if inverse else w) * t + cross(u, t)

def quat_rotate_inverse(q: Tensor, v: Tensor) -> Tensor:
    return quat_rotate(q, v, inverse = True)

def projected_gravity(q: Tensor) -> Tensor:
    return quat_rotate_inverse(q, q.new_tensor((0., 0., -1.)))

def world_to_heading_frame(q: Tensor, v: Tensor) -> Tensor:
    w, x, y, z = l2norm(q).unbind(dim = -1)
    yaw = torch.atan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))
    c, s = yaw.cos(), yaw.sin()

    vx, vy = v[..., 0], v[..., 1]
    rx = c * vx + s * vy
    ry = -s * vx + c * vy

    if is_shape(v, '... 2'):
        return stack((rx, ry), dim = -1)

    return stack((rx, ry, v[..., 2].expand_as(rx)), dim = -1)

def quat_to_rotmat(q: Tensor) -> Tensor:
    basis = torch.eye(3, device = q.device, dtype = q.dtype)
    cols = quat_rotate(rearrange(q, '... d -> ... 1 d'), basis)
    return rearrange(cols, '... col row -> ... row col')

def rot6d_to_rotmat(rot6d: Tensor) -> Tensor:
    b1 = l2norm(rot6d[..., :3])
    b2 = l2norm(orthogonal_component(rot6d[..., 3:6], b1, eps = 1e-8))
    return stack((b1, b2, cross(b1, b2)), dim = -1)

def sin_half_sq_to_angle_sq(sin_half_sq: Tensor) -> Tensor:
    # smooth, NaN-free exact map sin^2(theta / 2) -> theta^2 on [0, pi^2]
    u = sin_half_sq.clamp(min = 0., max = 1.)
    small = u < 1e-6
    safe_u = torch.where(small, 0.5, u)

    theta = 2. * torch.atan2(safe_u.sqrt(), (1. - safe_u).clamp(min = 1e-12).sqrt())
    return torch.where(small, 4. * u * (1. + u / 3.), theta.square())

def angle_diff(pred: Tensor, target: Tensor) -> Tensor:
    delta = target - pred
    return torch.atan2(delta.sin(), delta.cos())

def rotation_squared_error(pred: Tensor, target: Tensor) -> Tensor:
    # geodesic squared angle ||log(R_des R^T)||^2 across rotmat, quat, 6d rot, or 2d angle
    if is_shape(pred, '... 3 3'):
        sin_half_sq = reduce((target - pred).square(), '... i j -> ...', 'sum') / 8.
        return sin_half_sq_to_angle_sq(sin_half_sq)

    if is_shape(pred, '... 4'):
        p = l2norm(pred)
        t = l2norm(target)
        dot = einsum(p, t, '... d, ... d -> ...')
        p = einx.where('..., ... d, ... d -> ... d', dot >= 0., p, -p)
        sin_quarter_sq = reduce((t - p).square(), '... d -> ...', 'sum') / 4.
        return 4. * sin_half_sq_to_angle_sq(sin_quarter_sq)

    if is_shape(pred, '... 6'):
        return rotation_squared_error(rot6d_to_rotmat(pred), rot6d_to_rotmat(target))

    err = angle_diff(pred, target).square()
    return rearrange(err, '... 1 -> ...') if is_shape(err, '... 1') else err

# analytical reward terms

METRICS = dict(
    l2 = lambda p, t: (t - p).square(),
    l1 = lambda p, t: (t - p).abs(),
    angle = lambda p, t: angle_diff(p, t).square()
)

KERNELS = dict(
    exp = lambda x: torch.exp(-x),
    neg = lambda x: -x,
    rational = lambda x: 1. / (1. + x)
)

def weighted_reduce(t: Tensor, weights: Tensor | None, mode = 'sum') -> Tensor:
    if exists(weights):
        return einsum(t, weights.to(t), '... d, d -> ...')
    return reduce(t, '... d -> ...', mode)

def masked_weighted_reduce(t: Tensor, mask: Tensor, weights: Tensor | None = None, mode = 'sum') -> tuple[Tensor, Tensor]:
    mask = mask.to(t)

    if mask.shape[-1] == 1 and t.shape[-1] > 1:
        active = (rearrange(mask, '... 1 -> ...') > 0).to(t)
        return weighted_reduce(t, weights, mode), active

    w = mask * weights.to(t) if exists(weights) else mask
    w_sum = reduce(w, '... d -> ...', 'sum')
    active = (w_sum > 0).to(t)

    if exists(weights):
        scale = weights.to(t).sum() if mode == 'sum' else 1.
    else:
        scale = float(t.shape[-1]) if mode == 'sum' else 1.

    val = reduce(t * w, '... d -> ...', 'sum') * (scale / w_sum.clamp(min = 1e-8))
    return val, active

def select_optional(tree: State, key) -> Tensor | None:
    if not exists(key):
        return None
    if isinstance(key, str):
        return tree.get(key) if isinstance(tree, dict) else None
    try:
        return select(tree, key)
    except (KeyError, AssertionError):
        return None

def default_term_name(obj, fallback: str) -> str:
    name = obj if isinstance(obj, str) else getattr(obj, 'name', None) or getattr(obj, '__name__', None)
    return name if exists(name) and name != '<lambda>' else fallback

def resolve_named_terms(terms: Seq[Callable] | dict[str, Callable]) -> tuple[tuple[str, Callable], ...]:
    if isinstance(terms, dict):
        return tuple(terms.items())

    names = [default_term_name(t, f'term_{i}') for i, t in enumerate(terms)]
    counts = Counter(names)

    return tuple(
        (f'{name}_{i}' if counts[name] > 1 else name, term)
        for i, (name, term) in enumerate(zip(names, terms))
    )

def apply_transforms(x: State, transforms: tuple[Callable, ...]) -> State:
    for fn in transforms:
        x = fn(x)
    return x

class TrackingTerm(Module):
    def __init__(
        self,
        key = None,
        *,
        ref_key = None,
        mask_key = None,
        metric = 'l2',
        std = 1.,
        scale = None,
        kernel = 'exp',
        weight = 1.,
        reduce_items = False,
        item_weights = None,
        dim_weights = None,
        name = None
    ):
        super().__init__()
        assert callable(metric) or metric in ('rotation', 'so3', *METRICS), f'unknown metric {metric!r}'
        assert callable(kernel) or kernel in KERNELS, f'unknown kernel {kernel!r}'

        self.denom = float(default(scale, std ** 2))
        assert self.denom > 0.

        self.key = key
        self.ref_key = ref_key
        self.mask_key = mask_key
        self.metric = metric
        self.kernel = kernel
        self.weight = float(weight)
        self.reduce_items = reduce_items or exists(item_weights)
        self.name = default(name, default_term_name(key, 'track'))

        if exists(item_weights):
            item_weights = as_tensor(item_weights, dtype = torch.float32)
            item_weights = item_weights / item_weights.sum().clamp(min = 1e-8)

        if exists(dim_weights):
            dim_weights = as_tensor(dim_weights, dtype = torch.float32)

        self.register_buffer('item_weights', item_weights)
        self.register_buffer('dim_weights', dim_weights)

    def forward(self, states: State, references: State = None, *, return_active = False, **kwargs) -> Tensor:
        assert exists(references), f'TrackingTerm {self.name!r} requires `references`'
        pred, target = select_pair(states, references, self.key, self.ref_key)
        raw_mask = select_optional(references, self.mask_key)
        mask = align_time(pred, raw_mask) if exists(raw_mask) else None
        active = None

        if callable(self.metric):
            error = self.metric(pred, target)
        elif self.metric in ('rotation', 'so3'):
            error = rotation_squared_error(pred, target)
        elif exists(mask) and not self.reduce_items:
            error, active = masked_weighted_reduce(METRICS[self.metric](pred, target), mask, self.dim_weights, 'sum')
        else:
            error = weighted_reduce(METRICS[self.metric](pred, target), self.dim_weights, 'sum')

        if self.reduce_items:
            if exists(mask):
                error, active = masked_weighted_reduce(error, mask, self.item_weights, 'mean')
            else:
                error = weighted_reduce(error, self.item_weights, 'mean')
        elif exists(mask) and not exists(active):
            m = rearrange(mask, '... 1 -> ...') if is_shape(mask, '... 1') else mask
            active = (m.to(error) > 0).to(error)

        shaped = self.kernel(error) if callable(self.kernel) else KERNELS[self.kernel](error / self.denom)
        if exists(active):
            shaped = shaped * active

        out = self.weight * shaped
        if return_active:
            return out, default(active, torch.ones_like(out))
        return out

class StateTerm(Module):
    def __init__(self, fn: Callable, *, weight = 1., name = None):
        super().__init__()
        self.fn = fn
        self.weight = float(weight)
        self.name = default(name, default_term_name(fn, 'state'))

    def forward(self, states: State, references: State = None, **kwargs) -> Tensor:
        return self.weight * self.fn(states)

class JointLimitPenalty(Module):
    def __init__(
        self,
        lower,
        upper,
        *,
        key = 'joint_pos',
        margin = 0.,
        weight = -2.,
        name = 'joint_limit'
    ):
        super().__init__()
        lower = as_tensor(lower, dtype = torch.float32)
        upper = as_tensor(upper, dtype = torch.float32)
        assert lower.shape == upper.shape and (lower <= upper).all() and 0. <= margin < 1.

        shrink = 0.5 * (upper - lower) * margin

        self.key = key
        self.weight = float(weight)
        self.name = name

        self.register_buffer('lower', lower + shrink)
        self.register_buffer('upper', upper - shrink)

    def forward(self, states: State, references: State = None, **kwargs) -> Tensor:
        q = select(states, self.key)
        violation = F.relu(q - self.upper.to(q)) + F.relu(self.lower.to(q) - q)
        return self.weight * reduce(violation, '... d -> ...', 'sum')

class ActionSmoothnessTerm(Module):
    def __init__(self, weight = -0.1, order = 1, name = 'action_smoothness'):
        super().__init__()
        self.weight = float(weight)
        self.order = order
        self.name = name

    def forward(
        self,
        states = None,
        references = None,
        *,
        actions = None,
        past_actions = None,
        past_mask = None,
        mask = None,
        episode_starts = None,
        **kwargs
    ) -> Tensor | None:
        if not exists(actions):
            return None

        return self.weight * action_smoothness_penalty(
            actions,
            past_actions,
            past_mask = past_mask,
            mask = mask,
            order = self.order,
            episode_starts = episode_starts
        )

class AnalyticalReward(Module):
    def __init__(
        self,
        terms: Seq[Callable] | dict[str, Callable],
        *,
        state_transform = None,
        ref_transform = None,
        dt = 1.,
        normalize_masked = False
    ):
        super().__init__()
        self.named_terms = resolve_named_terms(terms)
        assert len(self.named_terms) > 0, '`terms` must be non-empty'

        self.state_transforms = compact(*cast_tuple(state_transform))
        self.ref_transforms = compact(*cast_tuple(ref_transform))
        self.dt = float(dt)
        self.normalize_masked = bool(normalize_masked)

        children = [*(fn for _, fn in self.named_terms), *self.state_transforms, *self.ref_transforms]
        self._submodules = ModuleList(dict.fromkeys(m for m in children if isinstance(m, Module)))

    def forward(
        self,
        states: State,
        references: State | None = None,
        *,
        return_breakdown = False,
        **kwargs
    ):
        states = apply_transforms(states, self.state_transforms)

        if exists(references):
            references = apply_transforms(references, self.ref_transforms)

        breakdown = {}
        tracked_names = []
        active_weight = None
        total_pos_weight = 0.

        for name, term in self.named_terms:
            is_masked_track = self.normalize_masked and isinstance(term, TrackingTerm) and term.weight > 0.
            if is_masked_track:
                val, active = term(states, references, return_active = True, **kwargs)
                tracked_names.append(name)
                total_pos_weight += term.weight
                term_w = active * term.weight
                active_weight = term_w if not exists(active_weight) else (active_weight + term_w)
            else:
                val = term(states, references, **kwargs)

            if exists(val):
                breakdown[name] = val * self.dt

        if self.normalize_masked and exists(active_weight) and total_pos_weight > 0.:
            scale = torch.where(active_weight > 0., total_pos_weight / active_weight.clamp(min = 1e-8), 0.)
            for name in tracked_names:
                breakdown[name] = breakdown[name] * scale

        vals = list(breakdown.values())
        assert len(vals) > 0, 'no reward terms produced a value'

        total = sum(vals[1:], vals[0])
        return (total, breakdown) if return_breakdown else total

# differentiable forward kinematics (any tree-structured robot)

Frame = namedtuple('Frame', ('pos', 'quat', 'lin_vel', 'ang_vel'))

Joint = namedtuple(
    'Joint',
    ('body', 'kind', 'qpos_adr', 'dof_adr', 'axis', 'pos', 'qpos0'),
    defaults = ((0., 0., 1.), (0., 0., 0.), 0.)
)

def vec3_buffer(vals) -> Tensor:
    return as_tensor(vals, dtype = torch.float32) if len(vals) > 0 else torch.empty(0, 3)

def default_hinge_joints(num_bodies: int, joint_axes = None, free_root = False) -> list[Joint]:
    q_off, d_off = (7, 6) if free_root else (0, 0)

    if exists(joint_axes):
        axes = as_tensor(joint_axes, dtype = torch.float32).tolist()
    else:
        axes = [(0., 0., 1.)] * (num_bodies - 1)

    assert len(axes) == num_bodies - 1

    joints = [Joint(0, 'free', 0, 0)] if free_root else []
    joints.extend(
        Joint(i + 1, 'hinge', q_off + i, d_off + i, axis)
        for i, axis in enumerate(axes)
    )
    return joints

class ForwardKinematics(Module):
    def __init__(
        self,
        parents: Seq[int],
        offsets,
        local_quats = None,
        joints: Seq[Joint] | None = None,
        *,
        joint_axes = None,
        free_root = False,
        tracked_bodies = None,
        free_ang_vel_in_body_frame = False
    ):
        super().__init__()
        num_bodies = len(parents)
        self.parents = tuple(parents)
        self.tracked_bodies = tuple(tracked_bodies) if exists(tracked_bodies) else None
        self.free_ang_vel_in_body_frame = free_ang_vel_in_body_frame

        self.register_buffer('offsets', as_tensor(offsets, dtype = torch.float32))

        if exists(local_quats):
            quats = l2norm(as_tensor(local_quats, dtype = torch.float32))
        else:
            quats = repeat(as_tensor((1., 0., 0., 0.)), 'd -> n d', n = num_bodies)

        self.register_buffer('local_quats', quats)

        if not exists(joints):
            joints = default_hinge_joints(num_bodies, joint_axes, free_root)

        joints = tuple(joints)
        assert all(j.kind in ('free', 'slide', 'hinge') for j in joints), 'unsupported joint kind'

        per_body = [[] for _ in range(num_bodies)]
        for idx, j in enumerate(joints):
            per_body[j.body].append((idx, j))

        self.body_joints = tuple(tuple(js) for js in per_body)
        self.has_free_joint = any(j.kind == 'free' for j in joints)

        self.register_buffer('joint_axes', l2norm(vec3_buffer([j.axis for j in joints])))
        self.register_buffer('joint_pos', vec3_buffer([j.pos for j in joints]))
        self.register_buffer('joint_qpos0', as_tensor([j.qpos0 for j in joints], dtype = torch.float32))

    @classmethod
    def from_mujoco(cls, model, tracked_body_names = None, include_world = False) -> ForwardKinematics:
        import mujoco

        start = 0 if include_world else 1
        kinds = {
            int(mujoco.mjtJoint.mjJNT_FREE): 'free',
            int(mujoco.mjtJoint.mjJNT_SLIDE): 'slide',
            int(mujoco.mjtJoint.mjJNT_HINGE): 'hinge'
        }

        parents, offsets, local_quats, names, joints = [], [], [], [], []

        for b in range(start, model.nbody):
            idx = b - start
            par = int(model.body_parentid[b])

            parents.append(-1 if b == 0 or par < start else par - start)
            offsets.append(model.body_pos[b].tolist())
            local_quats.append(model.body_quat[b].tolist())
            names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f'body_{b}')

            jadr = int(model.body_jntadr[b])
            jnum = int(model.body_jntnum[b])

            for j in range(jadr, jadr + jnum):
                jtype = int(model.jnt_type[j])
                assert jtype in kinds, f'unsupported MuJoCo joint type {jtype}'

                kind = kinds[jtype]
                qadr = int(model.jnt_qposadr[j])
                dadr = int(model.jnt_dofadr[j])
                q0 = 0. if kind == 'free' else float(model.qpos0[qadr])

                joints.append(Joint(
                    body = idx,
                    kind = kind,
                    qpos_adr = qadr,
                    dof_adr = dadr,
                    axis = model.jnt_axis[j].tolist(),
                    pos = model.jnt_pos[j].tolist(),
                    qpos0 = q0
                ))

        tracked = [names.index(name) for name in tracked_body_names] if exists(tracked_body_names) else None

        return cls(
            parents,
            offsets,
            local_quats,
            joints,
            tracked_bodies = tracked,
            free_ang_vel_in_body_frame = True
        )

    def resolve_inputs(self, state: State | None, qpos: Tensor | None, qvel: Tensor | None):
        if not isinstance(state, dict):
            qpos = default(qpos, state)
            assert exists(qpos), 'pass `state` dict or `qpos` tensor'
            return {}, qpos, qvel

        out = dict(state)
        qpos = default(qpos, out.get('qpos'))
        qvel = default(qvel, out.get('qvel'))

        if not exists(qpos):
            if self.has_free_joint:
                qpos = safe_cat((out['root_pos'], out['root_quat'], out.get('joint_pos')), dim = -1)
            else:
                qpos = out['joint_pos']

        if not exists(qvel):
            if not self.has_free_joint:
                qvel = out.get('joint_vel')
            elif 'root_lin_vel' in out and 'root_ang_vel' in out:
                qvel = safe_cat((out['root_lin_vel'], out['root_ang_vel'], out.get('joint_vel')), dim = -1)

        return out, qpos, qvel

    def apply_joint(
        self,
        frame: Frame,
        joint: Joint,
        qpos: Tensor,
        qvel: Tensor | None,
        axis: Tensor,
        jpos: Tensor,
        q0: Tensor
    ) -> Frame:
        p, q, v, w = frame
        has_vel = exists(qvel)

        if joint.kind == 'free':
            p = qpos[..., joint.qpos_adr : joint.qpos_adr + 3]
            q = l2norm(qpos[..., joint.qpos_adr + 3 : joint.qpos_adr + 7])

            if has_vel:
                v = qvel[..., joint.dof_adr : joint.dof_adr + 3]
                raw_w = qvel[..., joint.dof_adr + 3 : joint.dof_adr + 6]
                w = quat_rotate(q, raw_w) if self.free_ang_vel_in_body_frame else raw_w

            return Frame(p, q, v, w)

        dq = qpos[..., joint.qpos_adr] - q0
        qd = qvel[..., joint.dof_adr] if has_vel else None

        if joint.kind == 'slide':
            axis_w = quat_rotate(q, axis)
            p = p + einx.multiply('... d, ... -> ... d', axis_w, dq)
            if has_vel:
                v = v + einx.multiply('... d, ... -> ... d', axis_w, qd)
            return Frame(p, q, v, w)

        q_new = quat_mul(q, axis_angle_to_quat(axis, dq))
        w_new = w

        if has_vel:
            axis_w = quat_rotate(q, axis)
            w_new = w + einx.multiply('... d, ... -> ... d', axis_w, qd)

        if any(joint.pos):
            anchor = p + quat_rotate(q, jpos)
            p_new = anchor - quat_rotate(q_new, jpos)
            if has_vel:
                v = v + cross(w, anchor - p) + cross(w_new, p_new - anchor)
            p = p_new

        return Frame(p, q_new, v, w_new)

    def forward(
        self,
        state: State | None = None,
        qvel: Tensor | None = None,
        *,
        qpos: Tensor | None = None
    ) -> dict[str, Tensor]:
        out, qpos, qvel = self.resolve_inputs(state, qpos, qvel)
        has_vel = exists(qvel)
        lead = shape(qpos, '[...] d').shape

        offsets, local_quats, joint_axes, joint_pos, joint_qpos0 = (
            t.to(qpos) for t in (self.offsets, self.local_quats, self.joint_axes, self.joint_pos, self.joint_qpos0)
        )

        zero3 = qpos.new_zeros(*lead, 3)
        ident_q = qpos.new_tensor((1., 0., 0., 0.)).expand(*lead, 4)

        if self.has_free_joint:
            root = Frame(zero3, ident_q, zero3, zero3)
        else:
            root = Frame(
                out.get('root_pos', zero3),
                l2norm(out['root_quat']) if 'root_quat' in out else ident_q,
                out.get('root_lin_vel', zero3),
                out.get('root_ang_vel', zero3)
            )

        frames: list[Frame] = []

        for b, par in enumerate(self.parents):
            parent = root if par < 0 else frames[par]

            p = parent.pos + quat_rotate(parent.quat, offsets[b])
            q = quat_mul(parent.quat, local_quats[b])
            v = parent.lin_vel
            if has_vel:
                v = v + cross(parent.ang_vel, p - parent.pos)

            frame = Frame(p, q, v, parent.ang_vel)

            for idx, joint in self.body_joints[b]:
                frame = self.apply_joint(
                    frame, joint, qpos, qvel,
                    joint_axes[idx], joint_pos[idx], joint_qpos0[idx]
                )

            frames.append(frame)

        tracked = default(self.tracked_bodies, range(len(self.parents)))
        fields = Frame._fields if has_vel else Frame._fields[:2]

        for field in fields:
            out[f'body_{field}'] = stack([getattr(frames[b], field) for b in tracked], dim = -2)
            out.setdefault(f'root_{field}', getattr(frames[0], field))

        return out

# unitree humanoid presets (Tables A.1, A.2, A.6 & Appendix A)

UNITREE_G1_STATE_SPEC = dict(
    root_pos = 3,
    root_quat = 4,
    joint_pos = 29,
    root_lin_vel = 3,
    root_ang_vel = 3,
    joint_vel = 29
)

UNITREE_G1_STATE_DIM = sum(UNITREE_G1_STATE_SPEC.values())

UNITREE_G1_OBS_NOISE = dict(
    joint_pos = 0.01,
    joint_vel = 0.5,
    base_ang_vel = 0.2,
    projected_gravity = 0.05
)

UNITREE_MOTION_WEIGHTS = dict(
    root_pos = 0.5,
    root_orientation = 0.5,
    body_pos = 1.0,
    body_orientation = 1.0,
    body_lin_vel = 1.0,
    body_ang_vel = 1.0,
    action_smoothness = -0.1,
    joint_limit = -2.0
)

UNITREE_VELOCITY_WEIGHTS = dict(
    lin_vel = 2.0,
    yaw_rate = 2.0,
    action_smoothness = -0.3,
    joint_limit = -2.0,
    torso_sideways_lean = -4.0,
    torso_roll_pitch_rate = -0.05
)

def is_unitree_g1_tensor(t) -> bool:
    return is_shape(t, '... d', d = UNITREE_G1_STATE_DIM)

def unpack_unitree_g1_state(state: State, strict = True) -> State:
    if isinstance(state, dict):
        raw = state.get('state')
        if 'root_pos' not in state and is_unitree_g1_tensor(raw):
            return {**state, **unpack_unitree_g1_state(raw)}
        return state

    if is_unitree_g1_tensor(state):
        sizes = tuple(UNITREE_G1_STATE_SPEC.values())
        return dict(zip(UNITREE_G1_STATE_SPEC, state.split(sizes, dim = -1)))

    assert not strict, f'expected {UNITREE_G1_STATE_DIM}-d Unitree G1 state or dict'
    return state

def unitree_g1_state_to_obs(
    state: State,
    sim_context: Tensor | None = None,
    *,
    default_joint_pos: Tensor | None = None
) -> dict[str, Tensor]:
    # Table A.1 analytic obs g_omega(s_t, xi); Table A.3 sim_context = [friction(1), calib(29), com(3)]
    s = unpack_unitree_g1_state(state)
    joint_pos = s['joint_pos']

    if exists(default_joint_pos):
        joint_pos = joint_pos - default_joint_pos.to(joint_pos)

    if exists(sim_context):
        joint_pos = joint_pos - align_time(joint_pos, sim_context[..., 1:30].to(joint_pos))

    root_quat = s['root_quat']
    base_ang_vel = s.get('base_ang_vel')
    if not exists(base_ang_vel):
        base_ang_vel = quat_rotate_inverse(root_quat, s['root_ang_vel'])

    return dict(
        joint_pos = joint_pos,
        joint_vel = s['joint_vel'],
        base_ang_vel = base_ang_vel,
        projected_gravity = projected_gravity(root_quat)
    )

def resolve_weights(defaults: dict[str, float], overrides: dict[str, float] | None) -> dict[str, float]:
    if not exists(overrides):
        return dict(defaults)

    unknown = set(overrides) - set(defaults)
    assert not unknown, f'unknown reward weight keys: {unknown}'
    return {**defaults, **overrides}

def build_unitree_reward(terms, w, fk, joint_limits, dt, ref_transform = None) -> AnalyticalReward:
    active = {name: term for name, term in terms.items() if w[name] != 0.}

    if w['action_smoothness'] != 0.:
        active['action_smoothness'] = ActionSmoothnessTerm(weight = w['action_smoothness'])

    if exists(joint_limits) and w['joint_limit'] != 0.:
        active['joint_limit'] = JointLimitPenalty(*joint_limits, weight = w['joint_limit'])

    return AnalyticalReward(
        active,
        state_transform = (unpack_unitree_g1_state, fk),
        ref_transform = ref_transform,
        dt = dt
    )

def unitree_motion_tracking_reward(
    *,
    fk = None,
    joint_limits = None,
    weights = None,
    include_body_terms = True,
    dt = 0.02
) -> AnalyticalReward:
    # Table A.2
    w = resolve_weights(UNITREE_MOTION_WEIGHTS, weights)

    terms = dict(
        root_pos = TrackingTerm('root_pos', std = 0.3, weight = w['root_pos']),
        root_orientation = TrackingTerm('root_quat', metric = 'rotation', std = 0.4, weight = w['root_orientation'])
    )

    if include_body_terms:
        terms.update(
            body_pos = TrackingTerm('body_pos', std = 0.3, weight = w['body_pos'], reduce_items = True),
            body_orientation = TrackingTerm('body_quat', metric = 'rotation', std = 0.4, weight = w['body_orientation'], reduce_items = True),
            body_lin_vel = TrackingTerm('body_lin_vel', std = 1.0, weight = w['body_lin_vel'], reduce_items = True),
            body_ang_vel = TrackingTerm('body_ang_vel', std = 3.14, weight = w['body_ang_vel'], reduce_items = True)
        )

    return build_unitree_reward(terms, w, fk, joint_limits, dt)

def extract_velocity_command(references: State) -> dict[str, Tensor]:
    if isinstance(references, dict):
        return references
    assert is_shape(references, '... 3'), 'velocity command must be (..., 3) [vx, vy, wz] or dict'
    return dict(lin_vel = references[..., :2], yaw_rate = references[..., 2:3])

def heading_lin_vel(state: dict[str, Tensor]) -> Tensor:
    return world_to_heading_frame(state['root_quat'], state['root_lin_vel'])[..., :2]

def root_yaw_rate(state: dict[str, Tensor]) -> Tensor:
    return state['root_ang_vel'][..., 2:3]

def torso_sideways_lean(state: dict[str, Tensor]) -> Tensor:
    q = state.get('torso_quat', state['root_quat'])
    z_world = quat_rotate(q, q.new_tensor((0., 0., 1.)))
    return world_to_heading_frame(state['root_quat'], z_world)[..., 1].square()

def torso_roll_pitch_rate(state: dict[str, Tensor]) -> Tensor:
    omega = world_to_heading_frame(state['root_quat'], state.get('torso_ang_vel', state['root_ang_vel']))
    return reduce(omega[..., :2].square(), '... d -> ...', 'sum')

def unitree_velocity_tracking_reward(
    *,
    fk = None,
    joint_limits = None,
    weights = None,
    dt = 0.02
) -> AnalyticalReward:
    # Table A.6
    w = resolve_weights(UNITREE_VELOCITY_WEIGHTS, weights)

    terms = dict(
        lin_vel = TrackingTerm(heading_lin_vel, ref_key = 'lin_vel', std = 0.5, weight = w['lin_vel']),
        yaw_rate = TrackingTerm(root_yaw_rate, ref_key = 'yaw_rate', scale = 0.5, weight = w['yaw_rate']),
        torso_sideways_lean = StateTerm(torso_sideways_lean, weight = w['torso_sideways_lean']),
        torso_roll_pitch_rate = StateTerm(torso_roll_pitch_rate, weight = w['torso_roll_pitch_rate'])
    )

    return build_unitree_reward(terms, w, fk, joint_limits, dt, ref_transform = extract_velocity_command)

# analytical terminations (Appendix A & Table A.6)

def combine_margins(*margins: Tensor, temperature: float | None = None) -> Tensor:
    stacked = stack(margins, dim = 0)
    if not exists(temperature):
        return (stacked > 0.).any(dim = 0).float()

    assert temperature > 0.
    return 1. - reduce(torch.sigmoid(-stacked / temperature), 'n ... -> ...', 'prod')

def motion_tracking_termination(
    states: State,
    references: State,
    *,
    root_pos_key = 'root_pos',
    root_quat_key = 'root_quat',
    height_threshold = 0.25,
    tilt_threshold = 0.8,
    temperature = None
) -> Tensor:
    # Appendix A: |e_{p,z}| > 0.25 m or |u_z^ref - u_z| > 0.8
    states = unpack_unitree_g1_state(states, strict = False)
    references = unpack_unitree_g1_state(references, strict = False)
    p, p_ref = select_pair(states, references, root_pos_key)
    q, q_ref = select_pair(states, references, root_quat_key)

    return combine_margins(
        (p_ref[..., 2] - p[..., 2]).abs() - height_threshold,
        (projected_gravity(q_ref)[..., 2] - projected_gravity(q)[..., 2]).abs() - tilt_threshold,
        temperature = temperature
    )

def fall_termination(
    states: State,
    references: State | None = None,
    *,
    root_pos_key = 'root_pos',
    root_quat_key = 'root_quat',
    min_height = 0.35,
    tilt_threshold = 0.7,
    temperature = None
) -> Tensor:
    # Table A.6: h_root < 0.35 m or 1 + u_z > 0.7
    states = unpack_unitree_g1_state(states, strict = False)
    p = select(states, root_pos_key)
    q = select(states, root_quat_key)

    return combine_margins(
        min_height - p[..., 2],
        1. + projected_gravity(q)[..., 2] - tilt_threshold,
        temperature = temperature
    )
