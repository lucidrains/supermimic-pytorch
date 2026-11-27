# /// script
# requires-python = ">= 3.10"
# dependencies = [
#     "supermimic-pytorch",
#     "accelerate",
#     "memmap-replay-buffer>=0.3.0",
#     "mujoco>=3.2",
#     "pyarrow",
#     "ema-pytorch>=0.8.3",
#     "einops>=0.8.1",
#     "einx>=0.4.0",
#     "x-mlps-pytorch>=0.6.14",
#     "fire",
#     "numpy",
# ]
# ///

from __future__ import annotations

import importlib
import json
import math
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, as_tensor, cat

from supermimic_pytorch import (
    ActionSmoothnessTerm,
    AnalyticalReward,
    Critic,
    JointLimitPenalty,
    TrackingTerm,
    WorldModel,
    load_policy
)
from supermimic_pytorch.cli import ignore_return, pick_kwonly, train as train_cli
from supermimic_pytorch.mujoco_mocap import (
    MotionBank,
    Robot,
    build_robot,
    compute_qvel,
    encode_clip_windows,
    parse_csv,
    require,
    unpack_state
)
from supermimic_pytorch.supermimic import default, exists

# constants

HANDSON_REPO_ID = 'Cyan666/HandsOn-Datasets'
HANDSON_HF_BASE = f'https://huggingface.co/datasets/{HANDSON_REPO_ID}/resolve/main/'
WUJI_DESC_RAW = 'https://raw.githubusercontent.com/wuji-technology/wuji-hand-description/main/'

SIDES = ('right', 'left')
HANDSON_FPS = 30.0
HANDSON_CHUNK_SIZE = 1000
DEFAULT_CACHE_DIR = '~/.cache/supermimic/wuji_handson'

# Ry(+pi/2): maps Wuji rest frame (fingers +Z, palm +X) to HandsOn tabletop frame (fingers +X, palm -Z)
PALM_BASE_QUAT_WXYZ = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
PALM_BASE_ROT = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0]
], dtype = np.float64)

PALM_JOINT_SUFFIXES = ('palm_tx', 'palm_ty', 'palm_tz', 'palm_rx', 'palm_ry', 'palm_rz')
FINGER_JOINT_SUFFIXES = tuple(f'finger{f}_joint{j}' for f in range(1, 6) for j in range(1, 5))

BIMANUAL_JOINT_NAMES = tuple(
    f'{side}_{suffix}'
    for side in SIDES
    for suffix in (*PALM_JOINT_SUFFIXES, *FINGER_JOINT_SUFFIXES)
)

BIMANUAL_TRACKED_BODIES = tuple(
    name
    for side in SIDES
    for name in (f'{side}_palm_link', *(f'{side}_finger{f}_tip' for f in range(1, 6)))
)

BIMANUAL_CENTER_SLICES = ((0, 3), (26, 29))

DEFAULT_BIMANUAL_EPISODES = (
    5300,   # dual-hand drawer pull
    5900,   # hold bottle (left) + brush bands (right)
    6300,   # pass pen from right hand to left hand
    7200,   # dual-hand salt & pepper pour
    14200,  # bimanual ruler unfold
    15000   # bimanual fan unfold
)

DEFAULT_EVAL_EPISODES = (
    5301,   # held-out dual-hand drawer pull
    14201   # held-out bimanual ruler unfold
)

PALM_JOINT_RANGES = dict(
    tx = (-2.0, 2.0), ty = (-2.0, 2.0), tz = (0.0, 2.5),
    rx = (-2.60, 2.60), ry = (-2.00, 2.00), rz = (-3.14, 3.14)
)

# SO(3) <-> continuous intrinsic-XYZ Euler conversion

def wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + math.pi) % (2.0 * math.pi) - math.pi

def rotvec_to_rotmat(rv: np.ndarray) -> np.ndarray:
    rv = np.asarray(rv, dtype = np.float64)
    lead = rv.shape[:-1]
    v = rv.reshape(-1, 3)

    theta = np.linalg.norm(v, axis = -1, keepdims = True)
    small = theta < 1e-6
    safe_t = np.where(small, 1.0, theta)

    k1 = np.where(small, 1.0 - theta ** 2 / 6.0 + theta ** 4 / 120.0, np.sin(safe_t) / safe_t)
    k2 = np.where(small, 0.5 - theta ** 2 / 24.0 + theta ** 4 / 720.0, (1.0 - np.cos(safe_t)) / (safe_t ** 2))

    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    z = np.zeros_like(vx)
    K = np.stack((
        np.stack((z, -vz, vy), axis = -1),
        np.stack((vz, z, -vx), axis = -1),
        np.stack((-vy, vx, z), axis = -1)
    ), axis = -2)

    R = np.eye(3, dtype = np.float64) + k1[..., None] * K + k2[..., None] * (K @ K)
    return R.reshape(*lead, 3, 3)

def rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype = np.float64)
    lead = R.shape[:-2]
    M = R.reshape(-1, 3, 3)

    theta = np.arccos(np.clip(0.5 * (M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] - 1.0), -1.0, 1.0))
    skew = np.stack((
        M[:, 2, 1] - M[:, 1, 2],
        M[:, 0, 2] - M[:, 2, 0],
        M[:, 1, 0] - M[:, 0, 1]
    ), axis = -1)

    out = np.zeros((len(M), 3), dtype = np.float64)
    small = theta < 1e-5
    near_pi = theta > (math.pi - 1e-4)
    mid = (~small) & (~near_pi)

    if np.any(small):
        out[small] = 0.5 * skew[small] * (1.0 + theta[small, None] ** 2 / 6.0)

    if np.any(mid):
        tm = theta[mid, None]
        out[mid] = 0.5 * (tm / np.sin(tm)) * skew[mid]

    for i in np.nonzero(near_pi)[0]:
        axis = np.sqrt(np.maximum(0.5 * (np.diag(M[i]) + 1.0), 0.0))
        k = int(np.argmax(axis))
        for j in range(3):
            if j != k and M[i, k, j] + M[i, j, k] < 0.0:
                axis[j] = -axis[j]
        out[i] = (axis / max(np.linalg.norm(axis), 1e-12)) * theta[i]

    return out.reshape(*lead, 3)

def euler_xyz_to_rotmat(euler: np.ndarray) -> np.ndarray:
    e = np.asarray(euler, dtype = np.float64)
    ca, sa = np.cos(e[..., 0]), np.sin(e[..., 0])
    cb, sb = np.cos(e[..., 1]), np.sin(e[..., 1])
    cc, sc = np.cos(e[..., 2]), np.sin(e[..., 2])

    return np.stack((
        np.stack((cb * cc, -cb * sc, sb), axis = -1),
        np.stack((sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb), axis = -1),
        np.stack((-ca * sb * cc + sa * sc, ca * sb * sc + sa * cc, ca * cb), axis = -1)
    ), axis = -2)

def rotmat_to_euler_xyz_branches(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, dtype = np.float64)
    b1 = np.arcsin(np.clip(R[..., 0, 2], -1.0, 1.0))
    gimbal = np.abs(np.cos(b1)) < 1e-7

    a1 = np.where(gimbal, np.arctan2(R[..., 2, 1], R[..., 1, 1]), np.arctan2(-R[..., 1, 2], R[..., 2, 2]))
    c1 = np.where(gimbal, 0.0, np.arctan2(-R[..., 0, 1], R[..., 0, 0]))

    b2 = wrap_pi(math.pi - b1)
    a2 = wrap_pi(a1 + math.pi)
    c2 = wrap_pi(c1 + math.pi)
    return np.stack((a1, b1, c1), axis = -1), np.stack((a2, b2, c2), axis = -1)

def unwrap_euler_xyz(R_seq: np.ndarray) -> np.ndarray:
    b1, b2 = rotmat_to_euler_xyz_branches(R_seq)
    if len(b1) == 0:
        return b1

    out = np.empty_like(b1)
    out[0] = b1[0] if np.linalg.norm(b1[0]) <= np.linalg.norm(b2[0]) else b2[0]

    for t in range(1, len(b1)):
        prev = out[t - 1]
        c1 = prev + wrap_pi(b1[t] - prev)
        c2 = prev + wrap_pi(b2[t] - prev)
        out[t] = c1 if np.linalg.norm(c1 - prev) <= np.linalg.norm(c2 - prev) else c2

    return out

def transform_bimanual_rot(values: np.ndarray | Tensor, rot_fn) -> np.ndarray:
    arr = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    unbatched = arr.ndim == 1
    if unbatched:
        arr = arr[None, :]
    assert arr.ndim == 2 and arr.shape[-1] == 52, f'expected (T, 52), got {arr.shape}'

    hands = arr.astype(np.float64).reshape(-1, 2, 26)
    parts = [
        np.concatenate((hands[:, s, 0:3], rot_fn(hands[:, s, 3:6]), hands[:, s, 6:26]), axis = -1)
        for s in range(2)
    ]
    out = np.concatenate(parts, axis = -1).astype(np.float32)
    return out[0] if unbatched else out

def eef52_to_bimanual_qpos(states_52: np.ndarray | Tensor) -> np.ndarray:
    """Convert HandsOn EEF52 (T, 52) [right26, left26] -> bimanual Wuji MuJoCo qpos (T, 52)."""
    return transform_bimanual_rot(states_52, lambda rv: unwrap_euler_xyz(PALM_BASE_ROT.T @ rotvec_to_rotmat(rv)))

def bimanual_qpos_to_eef52(qpos_52: np.ndarray | Tensor) -> np.ndarray:
    """Inverse of `eef52_to_bimanual_qpos`: bimanual MuJoCo qpos (T, 52) -> HandsOn EEF52 (T, 52)."""
    return transform_bimanual_rot(qpos_52, lambda eul: rotmat_to_rotvec(PALM_BASE_ROT @ euler_xyz_to_rotmat(eul)))

# bimanual Wuji Hand 1 MJCF builder

def download_file(url: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents = True, exist_ok = True)
    tmp = dst.with_suffix(dst.suffix + '.part')
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)
    return dst

def ensure_wuji_hand_assets(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    root = Path(cache_dir).expanduser().resolve() / 'wuji_hand_description'
    mesh_tasks = []

    for side in SIDES:
        xml_path = download_file(f'{WUJI_DESC_RAW}mjcf/{side}.xml', root / 'mjcf' / f'{side}.xml')
        for mesh_el in ET.parse(xml_path).findall('.//asset/mesh'):
            fname = mesh_el.attrib['file']
            dst = root / 'meshes' / side / fname
            if not (dst.exists() and dst.stat().st_size > 0):
                mesh_tasks.append((f'{WUJI_DESC_RAW}meshes/{side}/{fname}', dst))

    if mesh_tasks:
        with ThreadPoolExecutor(max_workers = 8) as pool:
            list(pool.map(lambda item: download_file(*item), mesh_tasks))

    return root

def add_palm_joints(palm_body: ET.Element, side: str, *, palm_armature = 0.01, rot_armature = 0.002):
    palm_body.set('pos', '0 0 0')
    palm_body.set('quat', ' '.join(f'{v:.12g}' for v in PALM_BASE_QUAT_WXYZ))

    # slide axes in palm's Ry(+pi/2) frame -> exact world X, Y, Z translations
    joint_defs = (
        ('tx', 'slide', '0 0 1',  palm_armature, 0.5),
        ('ty', 'slide', '0 1 0',  palm_armature, 0.5),
        ('tz', 'slide', '-1 0 0', palm_armature, 0.5),
        ('rx', 'hinge', '1 0 0',  rot_armature,  0.05),
        ('ry', 'hinge', '0 1 0',  rot_armature,  0.05),
        ('rz', 'hinge', '0 0 1',  rot_armature,  0.05)
    )

    for idx, (ax, jtype, axis, arm, damp) in enumerate(joint_defs):
        lo, hi = PALM_JOINT_RANGES[ax]
        palm_body.insert(idx, ET.Element('joint', dict(
            name = f'{side}_palm_{ax}',
            type = jtype,
            pos = '0 0 0',
            axis = axis,
            limited = 'true',
            range = f'{lo} {hi}',
            armature = str(arm),
            damping = str(damp)
        )))

def style_hand_geoms_and_add_tips(palm_body: ET.Element, side: str, *, enable_collisions: bool = False):
    palm_rgba = '0.32 0.56 0.86 1' if side == 'right' else '0.18 0.68 0.62 1'
    finger_rgba = '0.62 0.78 0.94 1' if side == 'right' else '0.56 0.86 0.80 1'

    for body in palm_body.iter('body'):
        bname = body.attrib.get('name', '')
        tip_pos = None

        for g in list(body.findall('geom')):
            mesh_name = g.attrib.get('mesh', '')
            if mesh_name.endswith('_tip_link') and 'pos' in g.attrib and not exists(tip_pos):
                tip_pos = g.attrib['pos']

            if not enable_collisions:
                if g.attrib.get('group') != '1':
                    body.remove(g)
                    continue
                g.set('group', '0')
                g.set('contype', '0')
                g.set('conaffinity', '0')

            if mesh_name == f'{side}_palm_link':
                g.set('rgba', palm_rgba)
            elif mesh_name.endswith(('link3', 'link4', 'tip_link')):
                g.set('rgba', finger_rgba)

        if bname.startswith(f'{side}_finger') and bname.endswith('_link4'):
            tip = ET.SubElement(body, 'body', dict(
                name = bname.replace('_link4', '_tip'),
                pos = default(tip_pos, '0 0 0.0267')
            ))
            ET.SubElement(tip, 'inertial', dict(pos = '0 0 0', mass = '1e-6', diaginertia = '1e-9 1e-9 1e-9'))

def build_bimanual_wuji_mjcf(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    fps: float = HANDSON_FPS,
    substeps: int = 8,
    palm_vel_kv: float = 150.0,
    palm_vel_limit: float = 0.65,
    palm_pos_force: float = 100.0,
    palm_rot_kp: float = 24.0,
    palm_rot_kv: float = 2.4,
    palm_rot_force: float = 12.0,
    enable_collisions: bool = False,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    out_xml: str | Path | None = None
) -> Path:
    """Download official Wuji Hand 1 left/right descriptions and assemble a 52-DoF translation-equivariant MJCF."""
    desc_root = ensure_wuji_hand_assets(cache_dir)
    out_path = Path(out_xml).expanduser().resolve() if exists(out_xml) else (desc_root / 'mjcf' / 'bimanual_wuji.xml')
    out_path.parent.mkdir(parents = True, exist_ok = True)

    root = ET.Element('mujoco', dict(model = 'wujihand-bimanual'))
    ET.SubElement(root, 'compiler', dict(angle = 'radian', meshdir = str((desc_root / 'meshes').resolve())))
    ET.SubElement(root, 'option', dict(
        integrator = 'implicitfast',
        timestep = f'{1.0 / (float(fps) * int(substeps)):.8f}',
        gravity = ' '.join(f'{g:g}' for g in gravity)
    ))

    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'headlight', dict(diffuse = '0.55 0.55 0.55', ambient = '0.32 0.32 0.32', specular = '0.15 0.15 0.15'))
    ET.SubElement(visual, 'global', dict(offwidth = '1280', offheight = '960'))
    ET.SubElement(root, 'statistic', dict(center = '-0.34 0.0 1.01', extent = '0.34'))

    default_el = ET.SubElement(root, 'default')
    ET.SubElement(default_el, 'joint', dict(armature = '0.0004', damping = '0.002'))

    asset_el = ET.SubElement(root, 'asset')
    ET.SubElement(asset_el, 'texture', dict(name = 'sky', type = 'skybox', builtin = 'gradient', rgb1 = '0.94 0.96 0.99', rgb2 = '0.82 0.86 0.92', width = '64', height = '64'))
    ET.SubElement(asset_el, 'texture', dict(name = 'table_tex', type = '2d', builtin = 'checker', mark = 'edge', rgb1 = '0.86 0.88 0.91', rgb2 = '0.80 0.83 0.87', markrgb = '0.72 0.75 0.80', width = '256', height = '256'))
    ET.SubElement(asset_el, 'material', dict(name = 'table_mat', texture = 'table_tex', texuniform = 'true', texrepeat = '6 6', reflectance = '0.08'))

    worldbody = ET.SubElement(root, 'worldbody')
    ET.SubElement(worldbody, 'light', dict(pos = '0.5 -0.6 2.2', dir = '-0.3 0.3 -1.0', diffuse = '0.85 0.85 0.85', specular = '0.25 0.25 0.25', castshadow = 'false'))
    ET.SubElement(worldbody, 'light', dict(pos = '-0.8 0.6 2.0', dir = '0.4 -0.3 -1.0', diffuse = '0.45 0.48 0.52', castshadow = 'false'))
    ET.SubElement(worldbody, 'geom', dict(name = 'tabletop', type = 'plane', pos = '-0.35 0.0 0.78', size = '2.5 2.5 0.02', material = 'table_mat', contype = '0', conaffinity = '0'))

    actuator_el = ET.SubElement(root, 'actuator')

    for side in SIDES:
        src = ET.parse(desc_root / 'mjcf' / f'{side}.xml').getroot()

        for mesh in src.findall('./asset/mesh'):
            m = ET.SubElement(asset_el, 'mesh', dict(mesh.attrib))
            m.set('file', f'{side}/{mesh.attrib["file"]}')

        palm = src.find(f'./worldbody/body[@name="{side}_palm_link"]')
        assert exists(palm), f'missing {side}_palm_link in {side}.xml'
        for body in palm.iter('body'):
            body.set('gravcomp', '1')
        add_palm_joints(palm, side)
        style_hand_geoms_and_add_tips(palm, side, enable_collisions = enable_collisions)
        worldbody.append(palm)

        # velocity-controlled wrists -> exact 3D translation equivariance in MuJoCo physics
        for ax in ('tx', 'ty', 'tz'):
            ET.SubElement(actuator_el, 'velocity', dict(
                name = f'{side}_palm_{ax}_actuator',
                joint = f'{side}_palm_{ax}',
                kv = str(palm_vel_kv),
                ctrllimited = 'true',
                ctrlrange = f'{-palm_vel_limit} {palm_vel_limit}',
                forcelimited = 'true',
                forcerange = f'{-palm_pos_force} {palm_pos_force}'
            ))

        for ax in ('rx', 'ry', 'rz'):
            lo, hi = PALM_JOINT_RANGES[ax]
            ET.SubElement(actuator_el, 'position', dict(
                name = f'{side}_palm_{ax}_actuator',
                joint = f'{side}_palm_{ax}',
                kp = str(palm_rot_kp),
                kv = str(palm_rot_kv),
                ctrllimited = 'true',
                ctrlrange = f'{lo} {hi}',
                forcelimited = 'true',
                forcerange = f'{-palm_rot_force} {palm_rot_force}'
            ))

        for act in src.findall('./actuator/position'):
            attrib = dict(act.attrib)
            attrib.setdefault('ctrllimited', 'true')
            attrib.setdefault('forcelimited', 'true')
            ET.SubElement(actuator_el, 'position', attrib)

    ET.indent(root, space = '  ')
    ET.ElementTree(root).write(out_path, encoding = 'utf-8', xml_declaration = True)
    return out_path

def build_bimanual_wuji_spec(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    fps: float = HANDSON_FPS,
    substeps: int = 8,
    enable_collisions: bool = False
):
    """Factory compatible with `supermimic train --robot wuji_handson:build_bimanual_wuji_spec`."""
    mujoco = require('mujoco')
    xml_path = build_bimanual_wuji_mjcf(cache_dir, fps = fps, substeps = substeps, enable_collisions = enable_collisions)
    return mujoco.MjSpec.from_file(str(xml_path))

def build_bimanual_wuji_model(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    fps: float = HANDSON_FPS,
    substeps: int = 8,
    enable_collisions: bool = False
):
    mujoco = require('mujoco')
    xml_path = build_bimanual_wuji_mjcf(cache_dir, fps = fps, substeps = substeps, enable_collisions = enable_collisions)
    return mujoco.MjModel.from_xml_path(str(xml_path)), xml_path

# HandsOn-Datasets parquet reading & subset preparation

def handson_episode_relpath(episode_index: int) -> str:
    chunk = int(episode_index) // HANDSON_CHUNK_SIZE
    return f'data/chunk-{chunk:03d}/episode_{int(episode_index):06d}.parquet'

def load_handson_parquet(path: str | Path, *, source: str = 'state', name: str | None = None) -> dict[str, Any]:
    """Read a HandsOn LeRobot v2.1 `.parquet` episode and convert EEF52 -> 52-DoF bimanual Wuji qpos."""
    require('pyarrow')
    pq = importlib.import_module('pyarrow.parquet')
    p = Path(path).expanduser().resolve()
    col = 'observation.state' if source == 'state' else 'action'
    cols = [c for c in (col, 'task_index', 'episode_index') if c in pq.read_schema(p).names]
    table = pq.read_table(p, columns = cols)

    eef52 = np.asarray(table[col].to_pylist(), dtype = np.float32)
    assert eef52.ndim == 2 and eef52.shape[-1] == 52, f'{p.name}: expected (T, 52) {col!r}, got {eef52.shape}'
    qpos = eef52_to_bimanual_qpos(eef52)

    r_pos, r_j = eef52[:, 0:3], eef52[:, 6:26]
    l_pos, l_j = eef52[:, 26:29], eef52[:, 32:52]
    stats = dict(
        right_palm_disp = float(np.linalg.norm(r_pos - r_pos[0], axis = -1).max()),
        left_palm_disp = float(np.linalg.norm(l_pos - l_pos[0], axis = -1).max()),
        right_joint_span = float((r_j.max(axis = 0) - r_j.min(axis = 0)).max()),
        left_joint_span = float((l_j.max(axis = 0) - l_j.min(axis = 0)).max())
    )

    return dict(
        name = default(name, p.stem),
        qpos = qpos,
        eef52 = eef52,
        fps = HANDSON_FPS,
        joint_names = BIMANUAL_JOINT_NAMES,
        episode_index = int(table['episode_index'].to_numpy()[0]) if 'episode_index' in table.column_names else -1,
        task_index = int(table['task_index'].to_numpy()[0]) if 'task_index' in table.column_names else -1,
        bimanual_stats = stats
    )

def is_bimanual_active(stats: dict[str, float], *, min_palm_disp: float = 0.04, min_joint_span: float = 0.25) -> bool:
    return (
        (stats['right_palm_disp'] >= min_palm_disp or stats['right_joint_span'] >= min_joint_span)
        and (stats['left_palm_disp'] >= min_palm_disp or stats['left_joint_span'] >= min_joint_span)
    )

def fetch_handson_episodes_index(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    data_dir: str | Path | None = None
) -> dict[int, dict[str, Any]]:
    idx_path = (
        Path(data_dir).expanduser().resolve() / 'meta' / 'episodes.jsonl'
        if exists(data_dir) else
        download_file(f'{HANDSON_HF_BASE}meta/episodes.jsonl', Path(cache_dir).expanduser().resolve() / 'meta' / 'episodes.jsonl')
    )
    if not idx_path.exists():
        return {}
    return {
        int(row['episode_index']): row
        for line in idx_path.read_text().splitlines()
        if line.strip() and (row := json.loads(line))
    }

def parse_episode_ids(episodes) -> list[int] | None:
    if not exists(episodes):
        return None
    if isinstance(episodes, int):
        return [episodes]
    items = episodes if isinstance(episodes, (list, tuple)) else parse_csv(episodes)
    return [int(x) for x in items]

def select_candidate_episodes(ep_index: dict[int, dict[str, Any]], *, explicit_ids, data_dir, task_query, num_episodes) -> list[int]:
    if exists(explicit_ids):
        return explicit_ids
    if exists(data_dir):
        parquets = sorted((Path(data_dir).expanduser().resolve() / 'data').rglob('episode_*.parquet'))
        return [int(p.stem.split('_')[-1]) for p in parquets]
    if exists(task_query):
        q = task_query.lower()
        return [eid for eid, row in ep_index.items() if any(q in str(t).lower() for t in row.get('tasks', ()))]
    if exists(num_episodes):
        seen, diverse = set(), []
        for eid, row in ep_index.items():
            t = row['tasks'][0] if row.get('tasks') else str(eid)
            if t not in seen:
                seen.add(t)
                diverse.append(eid)
        return [*DEFAULT_BIMANUAL_EPISODES, *(e for e in diverse if e not in DEFAULT_BIMANUAL_EPISODES)]
    return list(DEFAULT_BIMANUAL_EPISODES)

def prepare_handson_dataset(
    out_dir: str | Path = './handson_prepared',
    *,
    episodes = None,
    num_episodes: int | None = None,
    data_dir: str | Path | None = None,
    task_query: str | None = None,
    bimanual_only: bool = True,
    source: str = 'state',
    max_frames: int | None = None,
    min_frames: int = 32,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    enable_collisions: bool = False
) -> dict[str, Any]:
    """Download/load a HandsOn-Datasets subset, convert to SuperMimic `.npz` clips, and build `bimanual_wuji.xml`."""
    out_root = Path(out_dir).expanduser().resolve()
    clips_dir = out_root / 'clips'
    clips_dir.mkdir(parents = True, exist_ok = True)

    robot_xml = build_bimanual_wuji_mjcf(
        cache_dir = cache_dir,
        enable_collisions = enable_collisions,
        out_xml = out_root / 'bimanual_wuji.xml'
    )

    ep_index = fetch_handson_episodes_index(cache_dir, data_dir = data_dir)
    explicit_ids = parse_episode_ids(episodes)
    candidate_ids = select_candidate_episodes(
        ep_index,
        explicit_ids = explicit_ids,
        data_dir = data_dir,
        task_query = task_query,
        num_episodes = num_episodes
    )

    target_count = default(num_episodes, len(candidate_ids))
    local_root = Path(data_dir).expanduser().resolve() if exists(data_dir) else None
    parquet_cache = Path(cache_dir).expanduser().resolve() / 'parquets'
    manifest_rows: list[dict[str, Any]] = []

    for eid in candidate_ids:
        if len(manifest_rows) >= target_count:
            break

        rel = handson_episode_relpath(eid)
        if exists(local_root):
            pq_path = local_root / rel
            if not pq_path.exists():
                continue
        else:
            pq_path = download_file(f'{HANDSON_HF_BASE}{rel}', parquet_cache / rel)

        clip = load_handson_parquet(pq_path, source = source)
        if len(clip['qpos']) < min_frames:
            continue
        if bimanual_only and not exists(explicit_ids) and not is_bimanual_active(clip['bimanual_stats']):
            continue

        qpos = clip['qpos'][:max_frames] if exists(max_frames) else clip['qpos']
        eef52 = clip['eef52'][:max_frames] if exists(max_frames) else clip['eef52']
        tasks = ep_index.get(eid, {}).get('tasks', ())
        task_text = str(tasks[0]) if tasks else ''

        clip_name = f'ep_{eid:06d}'
        npz_path = clips_dir / f'{clip_name}.npz'
        np.savez_compressed(
            npz_path,
            qpos = qpos,
            eef52 = eef52,
            fps = np.array(HANDSON_FPS, dtype = np.float32),
            joint_names = np.array(BIMANUAL_JOINT_NAMES),
            episode_index = np.array(eid, dtype = np.int64),
            task = np.array(task_text)
        )

        manifest_rows.append(dict(
            name = clip_name,
            episode_index = eid,
            frames = int(len(qpos)),
            task = task_text,
            path = str(npz_path),
            **clip['bimanual_stats']
        ))

    assert manifest_rows, 'no HandsOn episodes matched the requested filters'

    manifest = dict(
        dataset = HANDSON_REPO_ID,
        robot_xml = str(robot_xml),
        clips_dir = str(clips_dir),
        num_clips = len(manifest_rows),
        total_frames = sum(r['frames'] for r in manifest_rows),
        fps = HANDSON_FPS,
        tracked_bodies = list(BIMANUAL_TRACKED_BODIES),
        episodes = manifest_rows
    )
    (out_root / 'manifest.json').write_text(json.dumps(manifest, indent = 2))
    return manifest

# translation-equivariant bimanual obs, world model, critic & reward hooks

def bimanual_centroid(state: Tensor) -> Tensor:
    return 0.5 * (state[..., 0:3] + state[..., 26:29])

def shift_bimanual_state(state: Tensor, offset: Tensor) -> Tensor:
    out = state.clone()
    out[..., 0:3] = state[..., 0:3] + offset
    out[..., 26:29] = state[..., 26:29] + offset
    return out

def bimanual_relative_obs(state: Tensor) -> Tensor:
    """Exact 3D translation-invariant bimanual state: subtracts midpoint of right & left palms."""
    return shift_bimanual_state(state, -bimanual_centroid(state))

class TranslationEquivariantWorldModel(WorldModel):
    def forward(self, state, action, sim_context = None, next_state = None, *args, **kwargs):
        c = bimanual_centroid(state)
        s_rel = shift_bimanual_state(state, -c)
        ns_rel = shift_bimanual_state(next_state, -c) if exists(next_state) else None
        out = super().forward(s_rel, action, sim_context, ns_rel, *args, **kwargs)
        if hasattr(out, '_replace'):
            return out._replace(state = shift_bimanual_state(out.state, c))
        loss, pred, losses = out
        return loss, pred._replace(state = shift_bimanual_state(pred.state, c)), losses

    def rollout(self, state, actions, next_states, *args, **kwargs):
        c0 = bimanual_centroid(state)
        s0 = shift_bimanual_state(state, -c0)
        ns = shift_bimanual_state(next_states, -rearrange(c0, 'b d -> b 1 d'))
        loss, preds, losses = super().rollout(s0, actions, ns, *args, **kwargs)
        c0_rep = c0.repeat_interleave(len(preds[0].state) // len(c0), dim = 0)
        preds = [p._replace(state = shift_bimanual_state(p.state, c0_rep)) for p in preds]
        return loss, preds, losses

class RelativeBimanualCritic(Critic):
    def forward(self, state, *args, **kwargs):
        return super().forward(bimanual_relative_obs(state), *args, **kwargs)

def build_bimanual_world_model(**kwargs) -> TranslationEquivariantWorldModel:
    kwargs.pop('robot', None)
    kwargs.pop('bank', None)
    return TranslationEquivariantWorldModel(**kwargs)

def build_bimanual_critic(**kwargs) -> RelativeBimanualCritic:
    kwargs.pop('robot', None)
    kwargs.pop('bank', None)
    return RelativeBimanualCritic(**kwargs)

def add_relative_bimanual_terms(unpacked: dict[str, Tensor]) -> dict[str, Tensor]:
    q = unpacked['joint_pos']
    qd = unpacked['joint_vel']
    c = bimanual_centroid(q)

    rel_q = shift_bimanual_state(q, -c)
    palm_vel = cat((qd[..., 0:3], qd[..., 26:29]), dim = -1)
    centroid_disp = (c - c[..., :1, :]) if c.ndim >= 2 else torch.zeros_like(c)

    out = {**unpacked, 'rel_joint_pos': rel_q, 'palm_vel': palm_vel, 'centroid_disp': centroid_disp}
    if 'body_pos' in unpacked:
        out['rel_body_pos'] = unpacked['body_pos'] - rearrange(c, '... d -> ... 1 d')
    return out

def bimanual_handson_reward(
    robot: Robot,
    bank: MotionBank,
    *,
    interhand_weight: float = 2.5,
    palm_rot_weight: float = 4.0,
    finger_joint_weight: float = 2.5,
    body_weight: float = 3.0,
    body_coarse_weight: float = 2.0,
    body_std: float = 0.035,
    body_coarse_std: float = 0.14,
    palm_vel_weight: float = 3.5,
    palm_vel_std: float = 0.10,
    centroid_disp_weight: float = 4.0,
    centroid_disp_std: float = 0.020,
    joint_limit_weight: float = -0.5,
    action_smoothness_weight: float = -0.012
) -> AnalyticalReward:
    """Translation-invariant tracking reward: inter-hand geometry, palm rotations, fingers, relative 3D bodies, velocity & displacement."""
    assert robot.nj == 52, f'expected 52-DoF bimanual Wuji robot, got {robot.nj}'

    dim_w = torch.empty(52, dtype = torch.float32)
    for s in range(2):
        off = s * 26
        dim_w[off : off + 3] = interhand_weight * 20.0
        dim_w[off + 3 : off + 6] = palm_rot_weight
        dim_w[off + 6 : off + 26] = finger_joint_weight
    dim_w = dim_w / dim_w.sum()

    terms: dict[str, Any] = dict(
        rel_joint_pos = TrackingTerm('rel_joint_pos', mask_key = 'joint_mask', std = 0.30, weight = 3.0, dim_weights = dim_w),
        palm_vel = TrackingTerm('palm_vel', std = float(palm_vel_std), weight = palm_vel_weight, dim_weights = torch.full((6,), 1.0 / 6.0)),
        centroid_disp = TrackingTerm('centroid_disp', std = float(centroid_disp_std), weight = centroid_disp_weight),
        action_smoothness = ActionSmoothnessTerm(weight = action_smoothness_weight)
    )

    if robot.jnt_limited.any() and joint_limit_weight != 0.0:
        low = robot.jnt_low.clone()
        high = robot.jnt_high.clone()
        for lo, hi in BIMANUAL_CENTER_SLICES:
            low[lo:hi], high[lo:hi] = -1e6, 1e6
        terms['joint_limit'] = JointLimitPenalty(low, high, key = 'joint_pos', margin = 0.02, weight = joint_limit_weight)

    if exists(robot.fk):
        if body_coarse_weight > 0.0:
            terms['rel_body_pos_coarse'] = TrackingTerm('rel_body_pos', mask_key = 'body_mask', std = float(body_coarse_std), weight = body_coarse_weight, reduce_items = True)
        if body_weight > 0.0:
            terms['rel_body_pos'] = TrackingTerm('rel_body_pos', mask_key = 'body_mask', std = float(body_std), weight = body_weight, reduce_items = True)

    return AnalyticalReward(
        terms,
        state_transform = lambda s: add_relative_bimanual_terms(unpack_state(s, robot, with_fk = True)),
        ref_transform = add_relative_bimanual_terms,
        normalize_masked = True
    )

# one-liner latent encoding & standalone cli

def load_bimanual_qpos(source: str | Path | np.ndarray | Tensor, input_format: str = 'auto') -> np.ndarray:
    if isinstance(source, (str, Path)):
        p = Path(source).expanduser().resolve()
        suf = p.suffix.lower()
        if suf == '.parquet':
            return load_handson_parquet(p)['qpos']
        if suf == '.npz':
            with np.load(p, allow_pickle = True) as data:
                return data['qpos'].astype(np.float32) if 'qpos' in data else eef52_to_bimanual_qpos(data['eef52'])
        raw = np.load(p) if suf == '.npy' else torch.load(p, map_location = 'cpu', weights_only = False)
        return eef52_to_bimanual_qpos(raw) if input_format == 'eef52' else np.asarray(raw, dtype = np.float32)

    arr = source.detach().cpu().numpy() if torch.is_tensor(source) else np.asarray(source, dtype = np.float32)
    return eef52_to_bimanual_qpos(arr) if input_format == 'eef52' else arr.astype(np.float32)

@torch.no_grad()
def encode_handson_trajectory(
    policy,
    source: str | Path | np.ndarray | Tensor,
    *,
    input_format: str = 'auto',
    stride: int | None = None,
    robot: Robot | None = None,
    device = None
) -> dict[str, Any]:
    """Encode any HandsOn `.parquet`, `.npz`, `(T, 52)` EEF52 or qpos trajectory into `policy`'s latent space."""
    meta = policy.metadata
    dev = default(device, policy.obs_mean.device)

    if not exists(robot):
        robot_path = meta.get('robot_path')
        assert exists(robot_path), 'pass `robot` or use a checkpoint saved with `robot_path` metadata'
        robot = build_robot(robot_path, **meta.get('robot_cfg', {}))

    bank = MotionBank.from_metadata(meta, robot)
    qpos = as_tensor(load_bimanual_qpos(source, input_format), dtype = torch.float32)
    qvel = compute_qvel(qpos, 1.0 / robot.fps, has_free_root = False)
    states = cat((qpos, qvel), dim = -1)
    body_pos = robot.fk(qpos = qpos)['body_pos'].float() if exists(robot.fk) else None

    num_frames = len(qpos)
    clip = dict(
        states = states,
        body_pos = body_pos,
        joint_mask = torch.ones((num_frames, robot.nj)),
        root_mask = torch.ones((num_frames, 1)),
        body_mask = torch.ones((num_frames, robot.nb)) if exists(body_pos) else None
    )

    latents, starts = encode_clip_windows(clip, bank, policy, dev, stride = stride)
    return dict(latents = latents, starts = starts, window_len = bank.window_len, qpos = qpos, states = states, body_pos = body_pos)

def train_handson(
    out: str = './handson_latent_policy.pt', *,
    out_dir: str | None = None, episodes = None, num_episodes: int | None = None, data_dir: str | None = None,
    task_query: str | None = None, bimanual_only: bool = True, source: str = 'state',
    max_frames: int | None = None, min_frames: int = 32,
    cache_dir: str = DEFAULT_CACHE_DIR, enable_collisions: bool = False,
    dim_latent: int = 8, latent_space: str = 'box', clip_len: int = 16, clip_stride: int = 8,
    enc_frames: int = 5, uniformity_weight: float = 1.5, latent_noise: float = 0.08,
    iterations: int = 80, num_envs: int = 16, env_horizon: int = 96, warmup_steps: int = 128, warmup_wm_updates: int = 350,
    collect_steps: int = 16, buffer_capacity: int = 64_000, batch_size: int = 256, wm_horizon: int = 8, imag_horizon: int = 16,
    wm_rollout_loss_decay: float = 0.90, wm_state_noise_scale: float = 0.01,
    wm_updates: int = 24, policy_updates: int = 10, critic_updates: int = 4, command_every: int = 8,
    obs_window: int = 2, action_window: int = 1,
    encoder_hiddens = (256, 256), policy_hiddens = (512, 512, 256), wm_hiddens = (512, 512, 512), critic_hiddens = (256, 256),
    dim_explorative_latent: int = 4, num_explorative_rollouts: int = 2,
    wm_lr: float = 6e-4, actor_lr: float = 5e-4, critic_lr: float = 3e-4,
    use_median_grad_clip: bool = True, eval_every: int = 10,
    mixed_precision: str = 'no', log_with: str | None = None, seed: int = 42, device: str = 'auto'
) -> dict[str, Any]:
    """Fetch/load a HandsOn bimanual Wuji subset and train a translation-equivariant SuperMimic LatentPolicy."""
    args = locals()
    out_path = Path(out).expanduser()
    prep_dir = Path(out_dir).expanduser() if exists(out_dir) else (out_path.parent / f'{out_path.stem}_data')
    manifest = prepare_handson_dataset(prep_dir, **pick_kwonly(prepare_handson_dataset, args))

    train_kw = pick_kwonly(train_cli, args)
    train_kw.update(
        mode = 'clips',
        control = 'native',
        tracked_bodies = ','.join(BIMANUAL_TRACKED_BODIES),
        center_pos_slices = BIMANUAL_CENTER_SLICES,
        body_std = 0.035,
        obs_fn = 'wuji_handson:bimanual_relative_obs',
        world_model_fn = 'wuji_handson:build_bimanual_world_model',
        critic_fn = 'wuji_handson:build_bimanual_critic',
        reward_fn = 'wuji_handson:bimanual_handson_reward'
    )
    result = train_cli(manifest['robot_xml'], manifest['clips_dir'], str(out_path), **train_kw)
    result['manifest'] = manifest
    return result

def encode_cli(
    checkpoint: str,
    mocap: str,
    out: str | None = None,
    *,
    stride: int | None = None,
    device: str = 'cpu'
):
    """Encode a HandsOn `.parquet`/`.npz` episode or directory into latent trajectories."""
    dev = torch.device(device)
    policy = load_policy(checkpoint, map_location = dev).to(dev)
    src = Path(mocap).expanduser().resolve()
    files = sorted(p for p in src.rglob('*') if p.is_file() and p.suffix.lower() in ('.parquet', '.npz', '.npy', '.pt')) if src.is_dir() else [src]

    encoded = {}
    for p in files:
        enc = encode_handson_trajectory(policy, p, stride = stride, device = dev)
        encoded[p.stem] = dict(latents = enc['latents'].cpu(), starts = enc['starts'], window_len = enc['window_len'])

    payload = dict(checkpoint = str(Path(checkpoint).expanduser().resolve()), dim_latent = policy.dim_latent, clips = encoded)
    if exists(out):
        out_path = Path(out).expanduser()
        out_path.parent.mkdir(parents = True, exist_ok = True)
        torch.save(payload, out_path)
        print(f'[wuji_handson] encoded {len(encoded)} clip(s) -> {out_path}')
    else:
        total_w = sum(len(v['latents']) for v in encoded.values())
        print(f'[wuji_handson] encoded {len(encoded)} clip(s) ({total_w} latent windows, dim={policy.dim_latent})')
    return payload

if __name__ == '__main__':
    require('fire').Fire(dict(
        prepare = ignore_return(prepare_handson_dataset),
        train = ignore_return(train_handson),
        encode = ignore_return(encode_cli)
    ))
