from __future__ import annotations

import math

import pytest
import torch

from supermimic_pytorch import (
    LatentMotionActor,
    LatentPolicy,
    load_policy
)
from wuji_handson import (
    BIMANUAL_CENTER_SLICES,
    BIMANUAL_JOINT_NAMES,
    BIMANUAL_TRACKED_BODIES,
    PALM_BASE_ROT,
    RelativeBimanualCritic,
    TranslationEquivariantWorldModel,
    bimanual_handson_reward,
    bimanual_qpos_to_eef52,
    bimanual_relative_obs,
    eef52_to_bimanual_qpos,
    encode_cli,
    encode_handson_trajectory,
    euler_xyz_to_rotmat,
    load_handson_parquet,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
    shift_bimanual_state,
    unwrap_euler_xyz
)

def tiny_bimanual_xml() -> str:
    sides_xml, acts_xml = [], []

    for side in ('right', 'left'):
        j_xml = [
            f'<joint name="{side}_palm_tx" type="slide" axis="0 0 1" range="-1 1" limited="true"/>',
            f'<joint name="{side}_palm_ty" type="slide" axis="0 1 0" range="-1 1" limited="true"/>',
            f'<joint name="{side}_palm_tz" type="slide" axis="-1 0 0" range="0.3 2" limited="true"/>',
            f'<joint name="{side}_palm_rx" type="hinge" axis="1 0 0" range="-3.2 3.2" limited="true"/>',
            f'<joint name="{side}_palm_ry" type="hinge" axis="0 1 0" range="-3.2 3.2" limited="true"/>',
            f'<joint name="{side}_palm_rz" type="hinge" axis="0 0 1" range="-3.2 3.2" limited="true"/>'
        ]
        acts_xml.extend(
            f'<position name="{side}_palm_{ax}_act" joint="{side}_palm_{ax}" kp="50" kv="5" ctrllimited="true" ctrlrange="-1.5 1.5"/>'
            for ax in ('tx', 'ty', 'tz', 'rx', 'ry', 'rz')
        )

        fingers_xml = []
        for f in range(1, 6):
            inner = f'<body name="{side}_finger{f}_tip" pos="0 0 0.02"><inertial pos="0 0 0" mass="1e-4" diaginertia="1e-7 1e-7 1e-7"/></body>'
            for j in range(4, 0, -1):
                jname = f'{side}_finger{f}_joint{j}'
                acts_xml.append(f'<position name="{jname}_act" joint="{jname}" kp="1" kv="0.1" ctrllimited="true" ctrlrange="-0.5 1.6"/>')
                inner = (
                    f'<body name="{side}_finger{f}_link{j}" pos="0 0 0.02">'
                    f'<inertial pos="0 0 0.01" mass="0.01" diaginertia="1e-6 1e-6 1e-6"/>'
                    f'<joint name="{jname}" type="hinge" axis="0 1 0" range="-0.5 1.6" limited="true"/>'
                    f'{inner}</body>'
                )
            fingers_xml.append(inner)

        sides_xml.append(
            f'<body name="{side}_palm_link" pos="0 0 0" quat="0.7071067811865476 0 0.7071067811865476 0">'
            f'<inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>'
            + ''.join(j_xml)
            + ''.join(fingers_xml)
            + '</body>'
        )

    return f'<mujoco model="tiny_bimanual"><worldbody>{"".join(sides_xml)}</worldbody><actuator>{"".join(acts_xml)}</actuator></mujoco>'

def test_rotvec_and_eef52_roundtrip():
    np = pytest.importorskip('numpy')
    rng = np.random.default_rng(0)

    t = np.linspace(0.0, 2.0 * math.pi, 40)
    r_eul = np.stack((0.6 * np.sin(t), 0.4 * np.cos(t), 1.8 * np.sin(0.5 * t)), axis = -1)
    l_eul = np.stack((-0.5 * np.cos(t), 0.35 * np.sin(t), -1.5 * np.cos(0.5 * t)), axis = -1)

    eef = np.zeros((40, 52), dtype = np.float32)
    eef[:, 0:3] = rng.uniform(-0.5, 0.2, size = (40, 3))
    eef[:, 3:6] = rotmat_to_rotvec(PALM_BASE_ROT @ euler_xyz_to_rotmat(r_eul))
    eef[:, 6:26] = rng.uniform(-0.2, 1.2, size = (40, 20))
    eef[:, 26:29] = rng.uniform(-0.5, 0.2, size = (40, 3))
    eef[:, 29:32] = rotmat_to_rotvec(PALM_BASE_ROT @ euler_xyz_to_rotmat(l_eul))
    eef[:, 32:52] = rng.uniform(-0.2, 1.2, size = (40, 20))

    qpos = eef52_to_bimanual_qpos(eef)
    assert qpos.shape == (40, 52) and np.isfinite(qpos).all()
    assert np.abs(np.diff(qpos[:, 3:6], axis = 0)).max() < 0.35
    assert np.abs(np.diff(qpos[:, 29:32], axis = 0)).max() < 0.35

    eef_rt = bimanual_qpos_to_eef52(qpos)
    for cols in (slice(0, 3), slice(6, 26), slice(26, 29), slice(32, 52)):
        assert np.allclose(eef_rt[:, cols], eef[:, cols], atol = 1e-5)

    for s in range(2):
        R1 = rotvec_to_rotmat(eef[:, s * 26 + 3 : s * 26 + 6])
        R2 = rotvec_to_rotmat(eef_rt[:, s * 26 + 3 : s * 26 + 6])
        assert np.allclose(R1, R2, atol = 1e-5)

def test_unwrap_euler_across_pi_wrap():
    np = pytest.importorskip('numpy')
    angles = np.linspace(-math.pi - 0.5, math.pi + 0.5, 50)
    eul = np.stack((np.zeros_like(angles), 0.2 * np.sin(angles), angles), axis = -1)
    R_seq = euler_xyz_to_rotmat(eul)
    unwrapped = unwrap_euler_xyz(R_seq)

    assert np.abs(np.diff(unwrapped, axis = 0)).max() < 0.25
    assert np.allclose(euler_xyz_to_rotmat(unwrapped), R_seq, atol = 1e-6)

def test_bimanual_parquet_mocap_reward_and_encode(tmp_path):
    mujoco = pytest.importorskip('mujoco')
    np = pytest.importorskip('numpy')
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')

    from supermimic_pytorch.cli import encode as cli_encode
    from supermimic_pytorch.mujoco_mocap import build_motion_bank, build_robot, load_mocap_clips

    xml_path = tmp_path / 'tiny_bimanual.xml'
    xml_path.write_text(tiny_bimanual_xml())

    rob = build_robot(xml_path, tracked_bodies = ','.join(BIMANUAL_TRACKED_BODIES))
    assert rob.nq == 52 and rob.nu == 52 and not rob.has_free_root
    assert rob.joint_names == BIMANUAL_JOINT_NAMES

    T = 24
    base_rv = rotmat_to_rotvec(PALM_BASE_ROT[None])[0]
    eef = np.zeros((T, 52), dtype = np.float32)
    eef[:, 0:3], eef[:, 3:6] = (-0.35, -0.25, 1.00), base_rv
    eef[:, 26:29], eef[:, 29:32] = (-0.35, 0.25, 1.00), base_rv
    eef[:, 6:26] = 0.2 * np.sin(np.linspace(0, 2, T))[:, None]
    eef[:, 32:52] = 0.2 * np.cos(np.linspace(0, 2, T))[:, None]

    pq_path = tmp_path / 'episode_000001.parquet'
    pq.write_table(pa.table({
        'observation.state': pa.array([row.tolist() for row in eef]),
        'episode_index': pa.array([1] * T, type = pa.int64()),
        'task_index': pa.array([0] * T, type = pa.int64())
    }), pq_path)

    pq_clip = load_handson_parquet(pq_path)
    npz_path = tmp_path / 'episode_000001.npz'
    np.savez_compressed(npz_path, qpos = pq_clip['qpos'], fps = np.array(pq_clip['fps'], dtype = np.float32), joint_names = np.array(BIMANUAL_JOINT_NAMES))

    clips = load_mocap_clips(str(npz_path), rob)
    assert len(clips) == 1
    assert clips[0]['qpos'].shape == (T, 52) and clips[0]['body_pos'].shape == (T, 12, 3)

    data = mujoco.MjData(rob.model)
    data.qpos[:] = clips[0]['qpos'][0].numpy()
    mujoco.mj_forward(rob.model, data)
    r_pid = mujoco.mj_name2id(rob.model, mujoco.mjtObj.mjOBJ_BODY, 'right_palm_link')
    l_pid = mujoco.mj_name2id(rob.model, mujoco.mjtObj.mjOBJ_BODY, 'left_palm_link')
    assert np.allclose(data.xpos[r_pid], eef[0, 0:3], atol = 1e-5)
    assert np.allclose(data.xpos[l_pid], eef[0, 26:29], atol = 1e-5)

    bank = build_motion_bank(clips, rob, mode = 'clips', imag_horizon = 8)
    rew_fn = bimanual_handson_reward(rob, bank)
    st = clips[0]['states'][:8].unsqueeze(0).clone().requires_grad_(True)
    acts = torch.zeros(1, 8, 52, requires_grad = True)
    refs = dict(
        joint_pos = clips[0]['qpos'][:8].unsqueeze(0),
        joint_vel = clips[0]['qvel'][:8].unsqueeze(0),
        joint_mask = clips[0]['joint_mask'][:8].unsqueeze(0),
        body_pos = clips[0]['body_pos'][:8].unsqueeze(0),
        body_mask = clips[0]['body_mask'][:8].unsqueeze(0)
    )
    rew = rew_fn(st, references = refs, actions = acts)
    assert rew.shape == (1, 8)
    rew.sum().backward()
    assert st.grad is not None and torch.isfinite(st.grad).all()

    # verify exact 3D translation equivariance / invariance across obs, encoder, reward, world model & critic
    delta = torch.tensor([0.35, -0.22, 0.18], dtype = torch.float32)
    assert torch.allclose(bimanual_relative_obs(shift_bimanual_state(st.detach(), delta)), bimanual_relative_obs(st.detach()), atol = 1e-6)

    centered_bank = build_motion_bank(clips, rob, mode = 'clips', imag_horizon = 8, center_pos_slices = BIMANUAL_CENTER_SLICES)
    f_orig = centered_bank.encode(clips[0]['states'][:8], clips[0]['body_pos'][:8])
    f_shift = centered_bank.encode(shift_bimanual_state(clips[0]['states'][:8], delta), clips[0]['body_pos'][:8] + delta)
    assert torch.allclose(f_orig, f_shift, atol = 1e-5)

    refs_shift = {
        **refs,
        'joint_pos': shift_bimanual_state(refs['joint_pos'], delta),
        'body_pos': refs['body_pos'] + delta
    }
    rew_shift = rew_fn(shift_bimanual_state(st.detach(), delta), references = refs_shift, actions = acts.detach())
    assert torch.allclose(rew.detach(), rew_shift, atol = 1e-5)

    wm = TranslationEquivariantWorldModel(dim_state = 104, dim_action = 52, dim_sim_context = 0, dim_hiddens = (32,), pred_terminal = False)
    wm.eval()
    s0 = clips[0]['states'][:2]
    a0 = torch.randn(2, 52)
    p1 = wm(s0, a0).state
    p2 = wm(shift_bimanual_state(s0, delta), a0).state
    assert torch.allclose(p2, shift_bimanual_state(p1, delta), atol = 1e-5)

    critic = RelativeBimanualCritic(dim_state = 104, dim_goal = 4, dim_sim_context = 0, dim_hiddens = (32,))
    critic.eval()
    z_g = torch.randn(2, 4)
    assert torch.allclose(critic(s0, z_g), critic(shift_bimanual_state(s0, delta), z_g), atol = 1e-6)

    actor = LatentMotionActor(
        dim_obs = 104, dim_action = 52, dim_motion = bank.dim_motion, dim_latent = 4,
        encoder_hiddens = (32,), policy_hiddens = (32,)
    )
    policy = LatentPolicy(actor, metadata = dict(robot_path = str(xml_path), robot_cfg = rob.cfg, **bank.to_metadata()))
    ckpt = tmp_path / 'bimanual_policy.pt'
    policy.save(ckpt)
    loaded = load_policy(ckpt)

    enc = encode_handson_trajectory(loaded, pq_path)
    assert enc['latents'].ndim == 2 and enc['latents'].shape[-1] == 4
    assert (enc['latents'].abs() <= 1.0 + 1e-6).all()

    enc_eef = encode_handson_trajectory(loaded, eef, input_format = 'eef52')
    assert torch.allclose(enc['latents'], enc_eef['latents'], atol = 1e-5)

    out_latents = tmp_path / 'latents.pt'
    cli_res = cli_encode(str(ckpt), str(npz_path), str(out_latents), device = 'cpu')
    assert out_latents.exists() and 'episode_000001' in cli_res['clips']
    assert torch.allclose(cli_res['clips']['episode_000001']['latents'], enc['latents'], atol = 1e-5)

    pq_latents = tmp_path / 'pq_latents.pt'
    pq_res = encode_cli(str(ckpt), str(pq_path), str(pq_latents), device = 'cpu')
    assert pq_latents.exists() and 'episode_000001' in pq_res['clips']
    assert torch.allclose(pq_res['clips']['episode_000001']['latents'], enc['latents'], atol = 1e-5)
