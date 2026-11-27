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
import time
from bisect import bisect_right
from pathlib import Path

import fire
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from supermimic_pytorch import load_policy
from supermimic_pytorch.cli import load_callable
from supermimic_pytorch.mujoco_mocap import (
    MotionBank,
    build_robot,
    load_mocap_clips,
    rollout_policy,
    unpack_state
)
from supermimic_pytorch.supermimic import divisible_by, exists
from supermimic_pytorch.welford_normalizer import identity

from wuji_handson import (
    DEFAULT_BIMANUAL_EPISODES,
    DEFAULT_EVAL_EPISODES,
    bimanual_centroid,
    encode_handson_trajectory,
    prepare_handson_dataset,
    shift_bimanual_state,
    train_handson
)

# constants

SHORT_TASK_LABELS = {
    5300: 'Dual Drawer Pull',
    5301: 'Dual Drawer (Held-Out)',
    5900: 'Hold Bottle & Brush',
    5901: 'Hold Bottle (Held-Out)',
    6300: 'Pen Handoff (R -> L)',
    6301: 'Pen Handoff (Held-Out)',
    7200: 'Dual Salt/Pepper Pour',
    7201: 'Dual Pour (Held-Out)',
    14200: 'Bimanual Ruler Unfold',
    14201: 'Ruler Unfold (Held-Out)',
    15000: 'Bimanual Fan Unfold',
    15001: 'Fan Unfold (Held-Out)'
}

TASK_COLORS = ('#2563eb', '#16a34a', '#ea580c', '#7c3aed', '#db2777', '#0891b2', '#d97706', '#4f46e5')
FONT_PATHS = (
    '/System/Library/Fonts/Helvetica.ttc',
    '/System/Library/Fonts/SFNS.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
)

# helpers

def select_device(device: str = 'auto') -> torch.device:
    if device != 'auto':
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def load_font(size: int):
    for p in FONT_PATHS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

def short_label(ep_id: int, fallback: str = '') -> str:
    return SHORT_TASK_LABELS.get(ep_id) or (fallback[:28] if fallback else f'ep_{ep_id:06d}')

# closed-loop evaluation & latent verification

@torch.no_grad()
def eval_bimanual_clip(robot, policy, bank, clip, device, *, obs_fn = identity, command_every: int = 8) -> dict:
    horizon = len(clip['qpos']) - 1
    states_run, qposes, _ = rollout_policy(
        robot, policy, device, horizon,
        ref_clip = clip, bank = bank, obs_fn = obs_fn, command_every = command_every
    )
    pred = unpack_state(torch.stack(states_run, dim = 0).to(device), robot, with_fk = True)
    ref_q = clip['qpos'][1 : len(states_run) + 1].to(device)
    ref_b = clip['body_pos'][1 : len(states_run) + 1].to(device)

    pred_q = pred['joint_pos']
    pred_b = pred['body_pos']
    pred_c = bimanual_centroid(pred_q).unsqueeze(-2)
    ref_c = bimanual_centroid(ref_q).unsqueeze(-2)

    # world-frame & centroid-relative 3D body errors in mm (12 bodies: 0=R_palm, 1..5=R_tips, 6=L_palm, 7..11=L_tips)
    b_err_mm = (pred_b - ref_b).norm(dim = -1) * 1000.0
    rel_b_err_mm = ((pred_b - pred_c) - (ref_b - ref_c)).norm(dim = -1) * 1000.0
    rel_q_err = shift_bimanual_state(pred_q, -pred_c[:, 0]) - shift_bimanual_state(ref_q, -ref_c[:, 0])

    return dict(
        q_rmse = float(rel_q_err.square().mean().sqrt().item()),
        r_finger_rmse = float((pred_q[:, 6:26] - ref_q[:, 6:26]).square().mean().sqrt().item()),
        l_finger_rmse = float((pred_q[:, 32:52] - ref_q[:, 32:52]).square().mean().sqrt().item()),
        r_palm_mm = float(b_err_mm[:, 0].mean().item()),
        l_palm_mm = float(b_err_mm[:, 6].mean().item()),
        r_tip_mm = float(b_err_mm[:, 1:6].mean().item()),
        l_tip_mm = float(b_err_mm[:, 7:12].mean().item()),
        body_mm = float(b_err_mm.mean().item()),
        rel_body_mm = float(rel_b_err_mm.mean().item()),
        qposes = np.asarray(qposes, dtype = np.float32),
        ref_qpos = clip['qpos'].cpu().numpy(),
        per_frame_mm = b_err_mm.mean(dim = -1).cpu().numpy(),
        per_frame_rel_mm = rel_b_err_mm.mean(dim = -1).cpu().numpy()
    )

@torch.no_grad()
def eval_split(robot, policy, bank, entries, clips, device, *, obs_fn = identity, tag: str, held_out: bool = False) -> list[dict]:
    results = []
    for entry, clip in zip(entries, clips):
        ev = eval_bimanual_clip(robot, policy, bank, clip, device, obs_fn = obs_fn)
        ev.update(
            episode_index = entry['episode_index'],
            label = short_label(entry['episode_index'], entry['task']),
            path = entry['path'],
            held_out = held_out
        )
        results.append(ev)
        print(
            f'  [{tag:8s}] {ev["label"]:24s} | rel_q={ev["q_rmse"]:.3f}rad '
            f'| rel_12b={ev["rel_body_mm"]:4.1f}mm | world_12b={ev["body_mm"]:4.1f}mm '
            f'(R_tips={ev["r_tip_mm"]:4.1f}mm L_tips={ev["l_tip_mm"]:4.1f}mm)',
            flush = True
        )
    return results

@torch.no_grad()
def verify_latent_space(robot, policy, bank, train_entries, eval_entries, eval_clips, device, *, obs_fn = identity) -> dict:
    train_encs = [encode_handson_trajectory(policy, e['path'], robot = robot, device = device) for e in train_entries]
    eval_encs = [encode_handson_trajectory(policy, e['path'], robot = robot, device = device) for e in eval_entries]

    train_zs = [enc['latents'].cpu().numpy() for enc in train_encs]
    eval_zs = [enc['latents'].cpu().numpy() for enc in eval_encs]

    all_z = np.concatenate(train_zs, axis = 0)
    all_lbl = np.concatenate([np.full(len(z), i, dtype = np.int64) for i, z in enumerate(train_zs)])

    # pairwise latent spread & leave-one-out 1-NN task consistency
    dmat = np.linalg.norm(all_z[:, None] - all_z[None, :], axis = -1)
    np.fill_diagonal(dmat, np.inf)
    nn_task_acc = float(np.mean(all_lbl[np.argmin(dmat, axis = -1)] == all_lbl))

    # check midpoint interpolation & random latent rollouts remain finite in MuJoCo
    z_a = torch.from_numpy(train_zs[0][len(train_zs[0]) // 2 : len(train_zs[0]) // 2 + 1]).to(device)
    z_b = torch.from_numpy(train_zs[-1][len(train_zs[-1]) // 2 : len(train_zs[-1]) // 2 + 1]).to(device)
    z_mid = policy.project_latent(0.5 * (z_a + z_b))

    interp_states, _, _ = rollout_policy(robot, policy, device, 48, z_fixed = z_mid, bank = bank, obs_fn = obs_fn)
    rand_states, _, _ = rollout_policy(robot, policy, device, 60, random_latent = True, bank = bank, obs_fn = obs_fn, switch_every = 20)

    # verify exact 3D translation equivariance by shifting a held-out clip by (+30, +25, +15) cm
    shift_delta_mm = None
    shift_eval = None
    shift_video_eval = None
    if eval_clips:
        base_clip = eval_clips[0]
        base_label = short_label(eval_entries[0]['episode_index'], eval_entries[0]['task'])

        def eval_shifted(offset_xyz: tuple[float, float, float], label: str) -> dict:
            shift = torch.tensor(offset_xyz, dtype = torch.float32)
            shifted = dict(base_clip)
            shifted['qpos'] = shift_bimanual_state(base_clip['qpos'], shift)
            shifted['states'] = shift_bimanual_state(base_clip['states'], shift)
            shifted['body_pos'] = base_clip['body_pos'] + shift
            ev = eval_bimanual_clip(robot, policy, bank, shifted, device, obs_fn = obs_fn)
            ev.update(episode_index = eval_entries[0]['episode_index'], label = label, held_out = True)
            return ev

        ev_orig = eval_bimanual_clip(robot, policy, bank, base_clip, device, obs_fn = obs_fn)
        shift_eval = eval_shifted((0.30, 0.25, 0.15), f'{base_label} (+30,+25,+15cm)')
        shift_delta_mm = abs(shift_eval['body_mm'] - ev_orig['body_mm'])
        shift_video_eval = eval_shifted((0.10, 0.12, 0.14), f'{base_label} (+10,+12,+14cm Shift)')

    # 2D PCA projection of latent bank
    mean_z = all_z.mean(axis = 0, keepdims = True)
    _, _, vt = np.linalg.svd(all_z - mean_z, full_matrices = False)
    proj = vt[:2].T

    return dict(
        train_zs = train_zs,
        eval_zs = eval_zs,
        train_starts = [enc['starts'] for enc in train_encs],
        eval_starts = [enc['starts'] for enc in eval_encs],
        train_pca = [(z - mean_z) @ proj for z in train_zs],
        eval_pca = [(z - mean_z) @ proj for z in eval_zs],
        nn_task_acc = nn_task_acc,
        min_latent_dist = float(dmat.min()),
        mean_latent_dist = float(dmat[np.isfinite(dmat)].mean()),
        interp_finite = all(torch.isfinite(s).all().item() for s in interp_states),
        rand_finite = all(torch.isfinite(s).all().item() for s in rand_states),
        shift_equivariance_delta_mm = shift_delta_mm,
        shift_eval = shift_eval,
        shift_video_eval = shift_video_eval
    )

# plotting & side-by-side MuJoCo rendering

def make_convergence_and_latent_plot(history, train_results, eval_results, latent_info, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize = (12.5, 8.6), dpi = 160)
    fig.patch.set_facecolor('#f8fafc')

    steps = [h['steps'] / 1000.0 for h in history]
    wm = [h['wm_loss'] for h in history]
    q_err = [h['q_rmse'] for h in history]
    ee_err = [h['ee_mm'] for h in history]

    ax = axes[0, 0]
    ax.plot(steps, wm, color = '#2563eb', lw = 2.2, marker = 'o', ms = 4)
    ax.set_title('World Model Rollout Loss (Noise-Regularized)', fontsize = 11, fontweight = 'bold')
    ax.set_xlabel('Simulator Steps (k)')
    ax.set_ylabel('MSE Loss')
    ax.grid(True, alpha = 0.25)

    ax = axes[0, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(steps, ee_err, color = '#0d9488', lw = 2.2, marker = 's', ms = 4, label = '12-Body 3D Error (mm)')
    l2 = ax2.plot(steps, q_err, color = '#ea580c', lw = 2.0, ls = '--', marker = '^', ms = 4, label = 'Joint RMSE (rad)')
    ax.set_title('Bimanual Latent Actor Convergence', fontsize = 11, fontweight = 'bold')
    ax.set_xlabel('Simulator Steps (k)')
    ax.set_ylabel('3D Palm + Fingertip Error (mm)', color = '#0d9488')
    ax2.set_ylabel('Joint RMSE (rad)', color = '#ea580c')
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc = 'upper right', fontsize = 8.5)
    ax.grid(True, alpha = 0.25)

    ax = axes[1, 0]
    shift_ev = latent_info.get('shift_eval')
    all_res = [*train_results, *eval_results, *([shift_ev] if exists(shift_ev) else [])]
    rel_mm = [r['rel_body_mm'] for r in all_res]
    world_mm = [r['body_mm'] for r in all_res]
    x, w = np.arange(len(all_res)), 0.36
    ax.bar(x - w / 2, rel_mm, w, color = '#14b8a6', label = 'Relative 12-Body (mm)')
    ax.bar(x + w / 2, world_mm, w, color = '#3b82f6', label = 'World 12-Body (mm)')
    if eval_results:
        ax.axvline(len(train_results) - 0.5, color = '#64748b', ls = ':', lw = 1.5)
        ax.text(len(train_results) - 0.35, max(*rel_mm, *world_mm) * 0.90, 'Held-out\n& Shifted', fontsize = 8, color = '#475569')
    if exists(latent_info.get('shift_equivariance_delta_mm')):
        ax.set_title(f'Closed-Loop Accuracy (3D Shift |Δ| = {latent_info["shift_equivariance_delta_mm"]:.4f} mm)', fontsize = 10.5, fontweight = 'bold')
    else:
        ax.set_title('Per-Task Closed-Loop Bimanual Accuracy', fontsize = 11, fontweight = 'bold')
    ax.set_xticks(x)
    ax.set_xticklabels([r['label'] for r in all_res], rotation = 26, ha = 'right', fontsize = 7.8)
    ax.set_ylabel('Mean 12-Body Error (mm)')
    ax.legend(fontsize = 8.5, loc = 'upper left')
    ax.grid(True, axis = 'y', alpha = 0.25)

    ax = axes[1, 1]
    for i, (pts, res) in enumerate(zip(latent_info['train_pca'], train_results)):
        col = TASK_COLORS[i % len(TASK_COLORS)]
        ax.plot(pts[:, 0], pts[:, 1], color = col, lw = 1.3, alpha = 0.65)
        ax.scatter(pts[:, 0], pts[:, 1], color = col, s = 22, alpha = 0.85, label = res['label'])
    for j, (pts, res) in enumerate(zip(latent_info['eval_pca'], eval_results)):
        marker, edge = ('D', '#111827') if j == 0 else ('^', '#dc2626')
        ax.scatter(pts[:, 0], pts[:, 1], facecolors = 'none', edgecolors = edge, s = 44, lw = 1.2, marker = marker, label = res['label'])
    ax.set_title(f'Learned Latent Space (PCA, 1-NN task acc = {latent_info["nn_task_acc"] * 100:.1f}%)', fontsize = 11, fontweight = 'bold')
    ax.set_xlabel('Latent PC 1')
    ax.set_ylabel('Latent PC 2')
    ax.legend(fontsize = 7.5, loc = 'best', ncol = 2, framealpha = 0.85)
    ax.grid(True, alpha = 0.25)

    plt.tight_layout(pad = 1.6)
    out_path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(out_path, dpi = 160)
    plt.close(fig)

def render_side_by_side_video(
    model: mujoco.MjModel,
    segments: list[dict],
    out_mp4: Path,
    out_webp: Path,
    out_preview_png: Path,
    *,
    fps: float = 30.0,
    width: int = 440,
    height: int = 340,
    webp_stride: int = 2
):
    renderer = mujoco.Renderer(model, height = height, width = width)
    data = mujoco.MjData(model)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [-0.34, 0.04, 1.06]
    cam.distance = 1.24
    cam.elevation = -32.0
    cam.azimuth = 180.0

    def render_pose(q: np.ndarray) -> Image.Image:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera = cam)
        return Image.fromarray(renderer.render())

    font_title, font_sub, font_small = load_font(17), load_font(13), load_font(11)
    header_h, footer_h = 54, 34
    total_w = width * 2 + 8
    total_h = header_h + height + footer_h
    if total_h % 2 == 1:
        footer_h += 1
        total_h += 1

    mp4_frames, webp_frames, preview_tiles = [], [], []

    for seg in segments:
        ref_q, pol_q, err_mm, rel_mm = seg['ref_qpos'], seg['qposes'], seg['per_frame_mm'], seg['per_frame_rel_mm']
        latents, starts = seg['latents'], seg['starts']
        n = min(len(ref_q), len(pol_q)) - 1
        preview_t = n // 2
        tag = ' [HELD-OUT ZERO-SHOT]' if seg.get('held_out') else ''

        for t in range(n):
            canvas = Image.new('RGB', (total_w, total_h), (18, 24, 38))
            canvas.paste(render_pose(ref_q[t]), (0, header_h))
            canvas.paste(render_pose(pol_q[t]), (width + 8, header_h))

            draw = ImageDraw.Draw(canvas)
            cur_err = float(err_mm[min(t, len(err_mm) - 1)])
            cur_rel = float(rel_mm[min(t, len(rel_mm) - 1)])
            draw.text((14, 8), 'HandsOn Reference (2 Wuji Hands)', fill = (191, 219, 254), font = font_title)
            draw.text((width + 22, 8), f'SuperMimic Latent Actor{tag}', fill = (153, 246, 228), font = font_title)
            draw.text((14, 32), f'Task: {seg["label"]}', fill = (226, 232, 240), font = font_sub)
            draw.text((width + 22, 32), f'12b err: {cur_err:4.1f}mm (rel {cur_rel:4.1f}mm) | f {t:03d}/{n:03d}', fill = (226, 232, 240), font = font_sub)

            z = latents[min(max(0, bisect_right(starts, t) - 1), len(latents) - 1)]
            draw.text((14, header_h + height + 10), f'Latent z ({len(z)}d):', fill = (148, 163, 184), font = font_small)
            bar_x0 = 112
            bar_w = min(64, (total_w - bar_x0 - 20) // max(1, len(z)) - 6)
            mid_y = header_h + height + footer_h // 2

            for di, val in enumerate(z):
                bx = bar_x0 + di * (bar_w + 6)
                cx = bx + bar_w // 2
                fill_x = int(cx + np.clip(val, -1.0, 1.0) * (bar_w // 2 - 1))
                col = (56, 189, 248) if val >= 0 else (244, 114, 182)
                draw.rectangle([bx, mid_y - 8, bx + bar_w, mid_y + 8], fill = (30, 41, 59), outline = (71, 85, 105))
                draw.rectangle([min(cx, fill_x), mid_y - 6, max(cx, fill_x), mid_y + 6], fill = col)

            mp4_frames.append(np.asarray(canvas))
            if divisible_by(t, webp_stride):
                webp_frames.append(canvas)
            if t == preview_t:
                preview_tiles.append(canvas.copy())

    renderer.close()

    out_mp4.parent.mkdir(parents = True, exist_ok = True)
    imageio.mimsave(str(out_mp4), mp4_frames, fps = fps, codec = 'libx264', pixelformat = 'yuv420p', macro_block_size = 1)

    out_webp.parent.mkdir(parents = True, exist_ok = True)
    webp_frames[0].save(
        out_webp,
        save_all = True,
        append_images = webp_frames[1:],
        duration = int(round(1000.0 * webp_stride / fps)),
        loop = 0,
        quality = 82
    )

    if preview_tiles:
        grid = Image.new('RGB', (total_w, total_h * len(preview_tiles)), (18, 24, 38))
        for i, tile in enumerate(preview_tiles):
            grid.paste(tile, (0, i * total_h))
        out_preview_png.parent.mkdir(parents = True, exist_ok = True)
        grid.save(out_preview_png)

# entrypoint

def main(
    out_dir: str = './results_wuji_handson',
    episodes = DEFAULT_BIMANUAL_EPISODES,
    eval_episodes = DEFAULT_EVAL_EPISODES,
    dim_latent: int = 8,
    iterations: int = 65,
    num_envs: int = 16,
    batch_size: int = 256,
    wm_updates: int = 24,
    policy_updates: int = 12,
    eval_every: int = 5,
    seed: int = 42,
    device: str = 'auto',
    eval_only: bool = False
):
    t0 = time.time()
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents = True, exist_ok = True)
    ckpt_path = out_root / 'handson_latent_policy.pt'

    # 1. train (or reload existing checkpoint) via `wuji_handson.train_handson`

    if eval_only:
        assert ckpt_path.exists(), f'checkpoint not found for --eval-only: {ckpt_path}'
        train_manifest = json.loads((out_root / 'train_data' / 'manifest.json').read_text())
        prev_summary = json.loads((out_root / 'summary.json').read_text()) if (out_root / 'summary.json').exists() else {}
        res = dict(checkpoint = str(ckpt_path), history = prev_summary.get('history', []), manifest = train_manifest)
    else:
        res = train_handson(
            out = str(ckpt_path),
            out_dir = str(out_root / 'train_data'),
            episodes = episodes,
            dim_latent = dim_latent,
            iterations = iterations,
            num_envs = num_envs,
            batch_size = batch_size,
            wm_updates = wm_updates,
            policy_updates = policy_updates,
            eval_every = eval_every,
            seed = seed,
            device = device
        )

    # 2. prepare held-out episodes & reload portable LatentPolicy

    eval_manifest = prepare_handson_dataset(out_dir = out_root / 'eval_data', episodes = eval_episodes)
    dev = select_device(device)

    policy = load_policy(ckpt_path, map_location = dev).to(dev)
    meta = policy.metadata
    robot = build_robot(meta['robot_path'], **meta['robot_cfg'])
    bank = MotionBank.from_metadata(meta, robot)
    obs_fn = load_callable(meta['obs_fn']) if exists(meta.get('obs_fn')) else identity

    train_manifest = res['manifest']
    train_clips = load_mocap_clips(train_manifest['clips_dir'], robot)
    eval_clips = load_mocap_clips(eval_manifest['clips_dir'], robot)

    # 3. closed-loop evaluation & latent verification

    print('\n=== Closed-Loop Bimanual MuJoCo Evaluation ===', flush = True)
    train_results = eval_split(robot, policy, bank, train_manifest['episodes'], train_clips, dev, obs_fn = obs_fn, tag = 'TRAIN')
    eval_results = eval_split(robot, policy, bank, eval_manifest['episodes'], eval_clips, dev, obs_fn = obs_fn, tag = 'HELD-OUT', held_out = True)

    latent_info = verify_latent_space(robot, policy, bank, train_manifest['episodes'], eval_manifest['episodes'], eval_clips, dev, obs_fn = obs_fn)
    shift_s = f'\n  (+30,+25,+15)cm shift equivariance |delta| = {latent_info["shift_equivariance_delta_mm"]:.4f} mm' if exists(latent_info['shift_equivariance_delta_mm']) else ''
    print(
        f'\n=== Latent Space Verification ({policy.dim_latent}d {policy.latent_space}) ===\n'
        f'  1-NN task consistency = {latent_info["nn_task_acc"] * 100:.1f}%\n'
        f'  min|dz| = {latent_info["min_latent_dist"]:.3f} | mean|dz| = {latent_info["mean_latent_dist"]:.3f}\n'
        f'  midpoint interpolation stable = {latent_info["interp_finite"]} | random latent rollout stable = {latent_info["rand_finite"]}{shift_s}',
        flush = True
    )

    # 4. render plots & side-by-side comparison videos (.mp4 + .webp)

    plot_path = out_root / 'wuji_handson_convergence.png'
    mp4_path = out_root / 'handson_ref_vs_supermimic.mp4'
    webp_path = out_root / 'handson_ref_vs_supermimic.webp'
    preview_path = out_root / 'handson_frames_preview.png'

    make_convergence_and_latent_plot(res['history'], train_results, eval_results, latent_info, plot_path)

    showcase = [
        (train_results[2], latent_info['train_zs'][2], latent_info['train_starts'][2]),
        (train_results[0], latent_info['train_zs'][0], latent_info['train_starts'][0])
    ]
    if len(eval_results) > 1:
        showcase.append((eval_results[1], latent_info['eval_zs'][1], latent_info['eval_starts'][1]))
    if eval_results:
        showcase.append((eval_results[0], latent_info['eval_zs'][0], latent_info['eval_starts'][0]))
    if exists(latent_info.get('shift_video_eval')):
        showcase.append((latent_info['shift_video_eval'], latent_info['eval_zs'][0], latent_info['eval_starts'][0]))

    video_segments = [
        dict(
            label = ev['label'],
            held_out = ev['held_out'],
            ref_qpos = ev['ref_qpos'],
            qposes = ev['qposes'],
            per_frame_mm = ev['per_frame_mm'],
            per_frame_rel_mm = ev['per_frame_rel_mm'],
            latents = z_seq,
            starts = starts
        )
        for ev, z_seq, starts in showcase
    ]
    render_side_by_side_video(robot.model, video_segments, mp4_path, webp_path, preview_path, fps = robot.fps, webp_stride = 2)

    summary = dict(
        checkpoint = str(ckpt_path),
        elapsed_sec = round(time.time() - t0, 1),
        dim_latent = policy.dim_latent,
        train_mean_q_rmse = float(np.mean([r['q_rmse'] for r in train_results])),
        train_mean_rel_body_mm = float(np.mean([r['rel_body_mm'] for r in train_results])),
        train_mean_body_mm = float(np.mean([r['body_mm'] for r in train_results])),
        train_mean_r_tip_mm = float(np.mean([r['r_tip_mm'] for r in train_results])),
        train_mean_l_tip_mm = float(np.mean([r['l_tip_mm'] for r in train_results])),
        eval_mean_q_rmse = float(np.mean([r['q_rmse'] for r in eval_results])) if eval_results else None,
        eval_mean_rel_body_mm = float(np.mean([r['rel_body_mm'] for r in eval_results])) if eval_results else None,
        eval_mean_body_mm = float(np.mean([r['body_mm'] for r in eval_results])) if eval_results else None,
        shift_equivariance_delta_mm = latent_info['shift_equivariance_delta_mm'],
        nn_task_acc = latent_info['nn_task_acc'],
        min_latent_dist = latent_info['min_latent_dist'],
        mean_latent_dist = latent_info['mean_latent_dist'],
        interp_finite = latent_info['interp_finite'],
        rand_finite = latent_info['rand_finite'],
        plot = str(plot_path),
        mp4 = str(mp4_path),
        video = str(webp_path),
        preview = str(preview_path),
        history = res['history']
    )
    (out_root / 'summary.json').write_text(json.dumps(summary, indent = 2))
    print(f'\nDone in {summary["elapsed_sec"]:.1f}s! Summary -> {out_root / "summary.json"}', flush = True)
    print(f'  Train mean: rel_q={summary["train_mean_q_rmse"]:.3f} rad, rel_12b={summary["train_mean_rel_body_mm"]:.1f} mm, world_12b={summary["train_mean_body_mm"]:.1f} mm (R tips {summary["train_mean_r_tip_mm"]:.1f} mm, L tips {summary["train_mean_l_tip_mm"]:.1f} mm)', flush = True)
    if eval_results:
        print(f'  Held-out mean (zero-shot): rel_q={summary["eval_mean_q_rmse"]:.3f} rad, rel_12b={summary["eval_mean_rel_body_mm"]:.1f} mm, world_12b={summary["eval_mean_body_mm"]:.1f} mm', flush = True)

if __name__ == '__main__':
    fire.Fire(main)
