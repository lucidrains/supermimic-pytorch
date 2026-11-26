from __future__ import annotations

import pytest
import torch

from supermimic_pytorch import LatentMotionActor, LatentPolicy, load_policy

# minimal inline MJCF robots for fast CPU tests

FIXED_ARM_XML = """
<mujoco model="tiny_arm">
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom type="sphere" size="0.03" mass="0.2"/>
      <body name="link1" pos="0 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" range="-1.5 1.5" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02" mass="0.3"/>
        <body name="tip" pos="0 0 0.25">
          <joint name="j2" type="hinge" axis="0 1 0" range="-1.5 1.5" limited="true"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.015" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

FLOATING_BOT_XML = """
<mujoco model="tiny_walker">
  <worldbody>
    <body name="torso" pos="0 0 0.6">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.08 0.06" mass="2.0"/>
      <body name="left_leg" pos="0 0.08 -0.06">
        <joint name="l_hip" type="hinge" axis="0 1 0" range="-1.0 1.0" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.025" mass="0.5"/>
      </body>
      <body name="right_leg" pos="0 -0.08 -0.06">
        <joint name="r_hip" type="hinge" axis="0 1 0" range="-1.0 1.0" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.025" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m_l" joint="l_hip" ctrllimited="true" ctrlrange="-20 20"/>
    <motor name="m_r" joint="r_hip" ctrllimited="true" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""

@pytest.mark.parametrize('latent_space', ('box', 'sphere', 'unbounded'))
def test_latent_policy_roundtrip_and_streaming(tmp_path, latent_space):
    torch.manual_seed(0)
    actor = LatentMotionActor(
        dim_obs = 6,
        dim_action = 3,
        dim_motion = 10,
        dim_latent = 4,
        num_obs_frames = 2,
        num_action_frames = 1,
        dim_explorative_latent = 2,
        latent_space = latent_space,
        encoder_hiddens = (16,),
        policy_hiddens = (16,),
    )
    obs_mean = torch.tensor([0.5, -0.2, 0.1, 0.0, 1.0, -1.0])
    obs_std = torch.tensor([0.5, 0.8, 1.2, 2.0, 0.4, 1.5])
    act_off = torch.tensor([0.2, -0.3, 0.5])
    act_scale = torch.tensor([2.0, 0.5, 1.5])
    clip_latents = actor.project_latent(torch.randn(3, 4))

    policy = LatentPolicy(
        actor,
        obs_mean = obs_mean,
        obs_std = obs_std,
        action_offset = act_off,
        action_scale = act_scale,
        clip_latents = clip_latents,
        clip_names = ('walk', 'trot', 'turn'),
        metadata = dict(fps = 30.0, state_layout = 'qpos_qvel'),
    )
    policy.eval()

    ckpt = tmp_path / 'policy.pt'
    policy.save(ckpt)
    loaded = load_policy(ckpt)

    assert loaded.clip_names == ('walk', 'trot', 'turn')
    assert loaded.metadata['fps'] == 30.0
    assert torch.allclose(loaded.clip_latents, clip_latents)

    z = loaded.sample_latent(2)
    assert z.shape == (2, 4)
    if latent_space == 'box':
        assert (z.abs() <= 1.0 + 1e-6).all()
    elif latent_space == 'sphere':
        assert torch.allclose(z.norm(dim = -1), torch.ones(2), atol = 1e-5)

    # 2D and 3D stateless forward parity
    obs_2d = torch.randn(2, 6)
    obs_3d = obs_2d.unsqueeze(1).expand(2, 2, 6)
    a2 = loaded(obs_2d, z)
    a3 = loaded(obs_3d, z)
    assert a2.shape == (2, 3)
    assert torch.allclose(a2, a3, atol = 1e-5)

    # streaming step matches stateless forward with past_action re-scaling
    obs0, obs1 = torch.randn(2, 6), torch.randn(2, 6)
    act0, cache = loaded.step(obs0, z)
    act1, cache = loaded.step(obs1, z, cache)
    obs_win = torch.stack((obs0, obs1), dim = 1)
    act1_stateless = loaded(obs_win, z, past_action = act0.unsqueeze(1))
    assert torch.allclose(act1, act1_stateless, atol = 1e-5)

    # unbatched 1D step + reset
    single_act, scache = loaded.step(obs0[0], z[0])
    assert single_act.shape == (3,)
    single_act2, _ = loaded.step(obs0[0], z[0], scache, reset = torch.tensor([True]))
    assert torch.allclose(single_act, single_act2, atol = 1e-5)

def test_latent_policy_wrap_env(tmp_path):
    pytest.importorskip('env_ssl_wrapper')
    gym = pytest.importorskip('gymnasium')
    np = pytest.importorskip('numpy')

    class DummyTaskEnv(gym.Env):
        observation_space = gym.spaces.Box(-10., 10., shape = (10,), dtype = np.float32)
        action_space = gym.spaces.Box(-2., 2., shape = (3,), dtype = np.float32)

        def reset(self, *, seed = None, options = None):
            super().reset(seed = seed)
            self.t = 0
            return np.zeros(10, dtype = np.float32), {}

        def step(self, action):
            self.t += 1
            return np.zeros(10, dtype = np.float32), float(np.asarray(action).mean()), self.t >= 16, False, {}

    actor = LatentMotionActor(
        dim_obs = 6, dim_action = 3, dim_motion = 10, dim_latent = 2,
        num_obs_frames = 2, num_action_frames = 1, encoder_hiddens = (16,), policy_hiddens = (16,)
    )
    policy = LatentPolicy(
        actor,
        obs_mean = torch.zeros(6),
        obs_std = torch.ones(6),
        action_offset = torch.zeros(3),
        action_scale = torch.full((3,), 2.0),
    )
    policy.save(tmp_path / 'p.pt')
    loaded = load_policy(tmp_path / 'p.pt')

    wrapped = loaded.wrap_env([DummyTaskEnv(), DummyTaskEnv()], horizon = 4, obs_fn = slice(0, 6))
    obs, _ = wrapped.reset(seed = 0)
    assert obs.shape == (2, 10)
    assert wrapped.action_space.shape == (2,)

    obs, _rew, _, _, info = wrapped.step(torch.zeros(2, 2))
    assert obs.shape == (2, 10)
    assert info['duration'].tolist() == [4.0, 4.0]

def test_mocap_loader_formats(tmp_path):
    pytest.importorskip('mujoco')
    np = pytest.importorskip('numpy')
    from supermimic_pytorch.cli import build_robot, load_mocap_clips

    arm_xml = tmp_path / 'arm.xml'
    arm_xml.write_text(FIXED_ARM_XML)
    arm_rob = build_robot(arm_xml, control = 'pd')
    assert not arm_rob['has_free_root']
    assert arm_rob['nq'] == 2 and arm_rob['nu'] == 2
    assert arm_rob['tracked_bodies'] == ('tip',)

    t = np.linspace(0., 1., 16, dtype = np.float32)
    q = np.stack((0.5 * np.sin(2 * np.pi * t), 0.3 * np.cos(2 * np.pi * t)), axis = -1)

    # .npz, .npy, .pt, .csv
    np.savez(tmp_path / 'c1.npz', qpos = q)
    np.save(tmp_path / 'c2.npy', q)
    torch.save(dict(dof_pos = torch.from_numpy(q)), tmp_path / 'c3.pt')
    np.savetxt(tmp_path / 'c4.csv', q, delimiter = ',')

    clips = load_mocap_clips(str(tmp_path), arm_rob)
    assert len(clips) == 4
    for c in clips:
        assert c['qpos'].shape == (16, 2)
        assert c['qvel'].shape == (16, 2)
        assert c['states'].shape == (16, 4)
        assert c['body_pos'].shape == (16, 1, 3)

    # split-key floating-base with xyzw quaternion & FPS resampling
    walk_xml = tmp_path / 'walker.xml'
    walk_xml.write_text(FLOATING_BOT_XML)
    walk_rob = build_robot(walk_xml, fps = 30.0, control = 'pd')
    assert walk_rob['has_free_root']
    assert walk_rob['nq'] == 9 and walk_rob['nu'] == 2

    split_dir = tmp_path / 'split'
    split_dir.mkdir()
    T60 = 61
    rp = np.stack((np.linspace(0., 1., T60), np.zeros(T60), np.full(T60, 0.55)), axis = -1).astype(np.float32)
    rq_xyzw = np.tile(np.array([0., 0., 0., 1.], dtype = np.float32), (T60, 1))
    jp = np.zeros((T60, 2), dtype = np.float32)
    np.savez(split_dir / 'walk.npz', root_pos = rp, root_quat = rq_xyzw, dof_pos = jp, fps = np.array(60.0))

    wclips = load_mocap_clips(str(split_dir), walk_rob, quat_order = 'xyzw', target_fps = 30.0)
    assert len(wclips) == 1
    assert wclips[0]['qpos'].shape == (31, 9)
    assert np.allclose(wclips[0]['qpos'][:, 3], 1.0, atol = 1e-5) # wxyz w == 1

    # dim mismatch raises a clear error
    bad = tmp_path / 'bad.npy'
    np.save(bad, np.zeros((10, 5), dtype = np.float32))
    with pytest.raises(AssertionError, match = 'qpos dim'):
        load_mocap_clips(str(bad), arm_rob)

def test_cli_train_eval_fixed_base_skills(tmp_path):
    pytest.importorskip('mujoco')
    pytest.importorskip('memmap_replay_buffer')
    np = pytest.importorskip('numpy')
    from supermimic_pytorch.cli import eval as cli_eval, inspect as cli_inspect, train as cli_train

    arm_xml = tmp_path / 'arm.xml'
    arm_xml.write_text(FIXED_ARM_XML)

    clips_dir = tmp_path / 'skills'
    clips_dir.mkdir()
    for name, target in (('up', (0.6, 0.4)), ('down', (-0.6, -0.4)), ('mid', (0.0, 0.0))):
        q = np.tile(np.array(target, dtype = np.float32), (10, 1))
        np.savez(clips_dir / f'{name}.npz', qpos = q)

    info = cli_inspect(str(arm_xml), str(clips_dir), imag_horizon = 6)
    assert info['mode'] == 'skills'
    assert info['num_primitives'] == 3
    assert info['tracked_bodies'] == ('tip',)

    ckpt_path = tmp_path / 'arm_policy.pt'
    res = cli_train(
        str(arm_xml), str(clips_dir), str(ckpt_path),
        dim_latent = 2,
        iterations = 2,
        num_envs = 2,
        env_horizon = 16,
        warmup_steps = 12,
        warmup_wm_updates = 2,
        collect_steps = 6,
        batch_size = 8,
        wm_horizon = 3,
        imag_horizon = 6,
        wm_updates = 2,
        policy_updates = 2,
        critic_updates = 1,
        encoder_hiddens = (16,),
        policy_hiddens = (16,),
        wm_hiddens = (32,),
        critic_hiddens = (16,),
        eval_every = 2,
        device = 'cpu',
    )
    assert ckpt_path.exists()
    assert len(res['history']) == 1

    policy = load_policy(ckpt_path)
    assert policy.dim_latent == 2
    assert policy.clip_latents.shape == (3, 2)
    assert set(policy.clip_names) == {'up', 'down', 'mid'}

    ev_clip = cli_eval(str(ckpt_path), clip = 'up', steps = 10, device = 'cpu')
    assert ev_clip['steps'] == 10

    ev_rand = cli_eval(str(ckpt_path), random_latent = True, steps = 10, device = 'cpu')
    assert ev_rand['steps'] == 10

    ev_mocap = cli_eval(str(ckpt_path), mocap = str(clips_dir / 'up.npz'), device = 'cpu')
    assert 'q_rmse' in ev_mocap and 'ee_mm' in ev_mocap

def test_cli_train_eval_floating_base_clips(tmp_path):
    pytest.importorskip('mujoco')
    pytest.importorskip('memmap_replay_buffer')
    np = pytest.importorskip('numpy')
    from supermimic_pytorch.cli import eval as cli_eval, train as cli_train

    walk_xml = tmp_path / 'walker.xml'
    walk_xml.write_text(FLOATING_BOT_XML)

    T = 40
    t = np.linspace(0., 1.3, T, dtype = np.float32)
    qpos = np.zeros((T, 9), dtype = np.float32)
    qpos[:, 0] = 0.2 * t
    qpos[:, 2] = 0.55
    qpos[:, 3] = 1.0
    qpos[:, 7] = 0.25 * np.sin(2 * np.pi * t)
    qpos[:, 8] = -0.25 * np.sin(2 * np.pi * t)
    mocap_file = tmp_path / 'walk.npz'
    np.savez(mocap_file, qpos = qpos)

    ckpt_path = tmp_path / 'walker_policy.pt'
    cli_train(
        str(walk_xml), str(mocap_file), str(ckpt_path),
        control = 'pd',
        dim_latent = 4,
        latent_space = 'sphere',
        iterations = 2,
        num_envs = 2,
        env_horizon = 20,
        warmup_steps = 12,
        warmup_wm_updates = 2,
        collect_steps = 6,
        batch_size = 8,
        wm_horizon = 3,
        imag_horizon = 6,
        wm_updates = 2,
        policy_updates = 2,
        critic_updates = 1,
        encoder_hiddens = (16,),
        policy_hiddens = (16,),
        wm_hiddens = (32,),
        critic_hiddens = (16,),
        eval_every = 2,
        device = 'cpu',
    )
    assert ckpt_path.exists()

    policy = load_policy(ckpt_path)
    assert policy.latent_space == 'sphere'
    assert policy.metadata['floating_base'] is True
    assert policy.metadata['state_layout'] == 'heading'
    assert torch.allclose(policy.clip_latents.norm(dim = -1), torch.ones(len(policy.clip_latents)), atol = 1e-5)

    ev = cli_eval(str(ckpt_path), mocap = str(mocap_file), steps = 12, device = 'cpu')
    assert ev['steps'] <= 12
    assert 'q_rmse' in ev

def test_cli_config_and_python_hooks(tmp_path):
    pytest.importorskip('mujoco')
    pytest.importorskip('memmap_replay_buffer')
    np = pytest.importorskip('numpy')
    from supermimic_pytorch.cli import eval as cli_eval, train as cli_train

    arm_xml = tmp_path / 'arm.xml'
    arm_xml.write_text(FIXED_ARM_XML)

    q = np.zeros((16, 2), dtype = np.float32)
    q[:, 0] = np.linspace(-0.4, 0.4, 16)
    mocap_file = tmp_path / 'wave.npz'
    np.savez(mocap_file, qpos = q)

    hooks_py = tmp_path / 'my_hooks.py'
    hooks_py.write_text(
        f"""
import mujoco
from supermimic_pytorch import LatentMotionActor, WorldModel

CALLED = {{'robot': 0, 'reward': 0, 'actor': 0, 'wm': 0}}

def build_spec():
    CALLED['robot'] += 1
    return mujoco.MjSpec.from_file({str(arm_xml)!r})

def make_reward(default_reward):
    CALLED['reward'] += 1
    def _rew(states, references, **kwargs):
        return default_reward(states, references, **kwargs) - 0.001 * kwargs['actions'].square().sum(dim = -1)
    return _rew

def make_actor(dim_obs, dim_action, dim_motion, dim_latent, num_obs_frames, num_action_frames, dim_explorative_latent, latent_space):
    CALLED['actor'] += 1
    return LatentMotionActor(
        dim_obs = dim_obs, dim_action = dim_action, dim_motion = dim_motion, dim_latent = dim_latent,
        num_obs_frames = num_obs_frames, num_action_frames = num_action_frames,
        dim_explorative_latent = dim_explorative_latent, latent_space = latent_space,
        encoder_hiddens = (16,), policy_hiddens = (16,),
    )

def make_wm(dim_state, dim_action):
    CALLED['wm'] += 1
    return WorldModel(dim_state = dim_state, dim_action = dim_action, dim_sim_context = 0, dim_hiddens = (24,), pred_terminal = False)
"""
    )

    cfg_toml = tmp_path / 'run.toml'
    ckpt_path = tmp_path / 'hooked_policy.pt'
    cfg_toml.write_text(
        f"""
robot = "{hooks_py}:build_spec"
mocap = "{mocap_file}"
out = "{ckpt_path}"
reward_fn = "{hooks_py}:make_reward"
actor_fn = "{hooks_py}:make_actor"
world_model_fn = "{hooks_py}:make_wm"
dim_latent = 3
obs_noise = 0.01
iterations = 99
num_envs = 2
env_horizon = 16
warmup_steps = 10
warmup_wm_updates = 2
collect_steps = 6
batch_size = 8
wm_horizon = 3
imag_horizon = 6
wm_updates = 2
policy_updates = 2
critic_updates = 1
critic_hiddens = [16]
eval_every = 2
device = "cpu"
"""
    )

    # explicit kwarg `iterations = 2` must override `iterations = 99` in run.toml
    res = cli_train(config = str(cfg_toml), iterations = 2)
    assert ckpt_path.exists()
    assert len(res['history']) == 1

    policy = load_policy(ckpt_path)
    assert policy.dim_latent == 3

    # eval automatically re-invokes the saved `hooks_py:build_spec` robot hook
    ev = cli_eval(str(ckpt_path), random_latent = True, steps = 8, device = 'cpu')
    assert ev['steps'] == 8

def test_cli_extra_sim_sensors_and_asymmetric_obs(tmp_path):
    pytest.importorskip('mujoco')
    pytest.importorskip('memmap_replay_buffer')
    np = pytest.importorskip('numpy')
    from supermimic_pytorch.cli import eval as cli_eval, train as cli_train

    touch_xml = """
<mujoco model="touch_walker">
  <worldbody>
    <body name="torso" pos="0 0 0.6">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.08 0.06" mass="2.0"/>
      <body name="left_leg" pos="0 0.08 -0.06">
        <joint name="l_hip" type="hinge" axis="0 1 0" range="-1.0 1.0" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.025" mass="0.5"/>
        <site name="l_foot_site" pos="0 0 -0.35" size="0.03"/>
      </body>
      <body name="right_leg" pos="0 -0.08 -0.06">
        <joint name="r_hip" type="hinge" axis="0 1 0" range="-1.0 1.0" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.025" mass="0.5"/>
        <site name="r_foot_site" pos="0 0 -0.35" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <touch name="l_touch" site="l_foot_site"/>
    <touch name="r_touch" site="r_foot_site"/>
  </sensor>
  <actuator>
    <motor name="m_l" joint="l_hip" ctrllimited="true" ctrlrange="-20 20"/>
    <motor name="m_r" joint="r_hip" ctrllimited="true" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""
    bot_xml = tmp_path / 'touch_walker.xml'
    bot_xml.write_text(touch_xml)

    # mocap has ONLY robot qpos (9 dims), zero touch/contact channels
    T = 32
    t = np.linspace(0., 1.0, T, dtype = np.float32)
    qpos = np.zeros((T, 9), dtype = np.float32)
    qpos[:, 2] = 0.42
    qpos[:, 3] = 1.0
    qpos[:, 7] = 0.2 * np.sin(2 * np.pi * t)
    qpos[:, 8] = -0.2 * np.sin(2 * np.pi * t)
    mocap_file = tmp_path / 'walk.npz'
    np.savez(mocap_file, qpos = qpos)

    sensor_hooks = tmp_path / 'sensor_hooks.py'
    sensor_hooks.write_text(
        """
import numpy as np
import torch

def read_touch(model, data):
    # 2 touch sensors -> symlog compressed, appended to kinematic sim state (15 -> 17 dims)
    s = np.asarray(data.sensordata, dtype = np.float32)
    return np.sign(s) * np.log1p(np.abs(s))

def proprio_plus_touch(raw_state):
    # drop privileged root_z / root_lin_vel; keep [joint_pos(2), root_ang_vel(3), touch(2)] = 7 dims
    joint_pos = raw_state[..., 5:7]
    root_ang_vel = raw_state[..., 10:13]
    touch = raw_state[..., 15:17]
    return torch.cat((joint_pos, root_ang_vel, touch), dim = -1)
"""
    )

    ckpt_path = tmp_path / 'touch_policy.pt'
    cli_train(
        str(bot_xml), str(mocap_file), str(ckpt_path),
        control = 'pd',
        extra_state_fn = f'{sensor_hooks}:read_touch',
        obs_fn = f'{sensor_hooks}:proprio_plus_touch',
        dim_latent = 4,
        iterations = 2,
        num_envs = 2,
        env_horizon = 16,
        warmup_steps = 10,
        warmup_wm_updates = 2,
        collect_steps = 6,
        batch_size = 8,
        wm_horizon = 3,
        imag_horizon = 6,
        wm_updates = 2,
        policy_updates = 2,
        critic_updates = 1,
        encoder_hiddens = (16,),
        policy_hiddens = (16,),
        wm_hiddens = (32,),
        critic_hiddens = (16,),
        eval_every = 2,
        device = 'cpu',
    )

    policy = load_policy(ckpt_path)
    # robot obs (7) != sim state (17) != mocap kin_dim (13)
    assert policy.dim_obs == 7
    assert policy.obs_mean.shape == (7,)
    assert len(policy.metadata['motion_mean']) == 13

    # eval replays both mocap-tracking and random-latent rollouts using saved hooks & separate motion stats
    ev_mocap = cli_eval(str(ckpt_path), mocap = str(mocap_file), steps = 10, device = 'cpu')
    assert 'q_rmse' in ev_mocap

    ev_rand = cli_eval(str(ckpt_path), random_latent = True, steps = 10, device = 'cpu')
    assert ev_rand['steps'] <= 10

def test_masked_tracking_reward_and_scale_normalization():
    from supermimic_pytorch import AnalyticalReward, TrackingTerm

    states = dict(
        joint_pos = torch.tensor([[[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]]]),
        body_pos = torch.zeros(1, 2, 2, 3),
        root_z = torch.ones(1, 2, 1),
    )
    refs_partial = dict(
        joint_pos = torch.zeros(1, 2, 3),
        joint_mask = torch.tensor([[[1., 0., 1.], [0., 0., 0.]]]),
        body_pos = torch.zeros(1, 2, 2, 3),
        body_mask = torch.tensor([[[1., 1.], [1., 0.]]]),
        root_z = torch.zeros(1, 2, 1),
        root_mask = torch.zeros(1, 2, 1),
    )

    reward = AnalyticalReward(
        dict(
            joint = TrackingTerm('joint_pos', mask_key = 'joint_mask', std = 0.5, weight = 2.0, dim_weights = torch.full((3,), 1 / 3)),
            body = TrackingTerm('body_pos', mask_key = 'body_mask', std = 0.5, weight = 3.0, reduce_items = True),
            root = TrackingTerm('root_z', mask_key = 'root_mask', std = 0.5, weight = 1.0),
        ),
        normalize_masked = True,
    )
    total, bd = reward(states, refs_partial, return_breakdown = True)

    assert torch.allclose(bd['root'], torch.zeros(1, 2))
    assert torch.allclose(bd['joint'][:, 1], torch.zeros(1))
    assert torch.allclose(total, torch.full((1, 2), 6.0), atol = 1e-5)

def test_cli_heterogeneous_masked_mocap(tmp_path):
    pytest.importorskip('mujoco')
    pytest.importorskip('memmap_replay_buffer')
    np = pytest.importorskip('numpy')
    from supermimic_pytorch.cli import eval as cli_eval, inspect as cli_inspect, train as cli_train

    walk_xml = tmp_path / 'walker.xml'
    walk_xml.write_text(FLOATING_BOT_XML)

    clips_dir = tmp_path / 'hetero_clips'
    clips_dir.mkdir()

    T = 24
    t = np.linspace(0., 0.8, T, dtype = np.float32)
    qpos = np.zeros((T, 9), dtype = np.float32)
    qpos[:, 0] = 0.2 * t
    qpos[:, 2] = 0.55
    qpos[:, 3] = 1.0
    qpos[:, 7] = 0.25 * np.sin(2 * np.pi * t)
    qpos[:, 8] = -0.25 * np.sin(2 * np.pi * t)

    # 1. full-body clip
    np.savez(clips_dir / 'full.npz', qpos = qpos)
    # 2. partial joint clip (left hip only, no right hip, no root)
    np.savez(clips_dir / 'left_only.npz', joint_pos = qpos[:, 7:8], joint_names = np.array(['l_hip']))
    # 3. keypoint-only clip (tracked leaf bodies only, zero qpos)
    body_pts = np.zeros((T, 2, 3), dtype = np.float32)
    body_pts[:, 0, 1], body_pts[:, 1, 1] = 0.08, -0.08
    body_pts[:, :, 2] = 0.14
    np.savez(clips_dir / 'feet_only.npz', body_pos = body_pts, body_names = np.array(['left_leg', 'right_leg']))

    info = cli_inspect(str(walk_xml), str(clips_dir), control = 'pd', imag_horizon = 6)
    assert info['num_clips'] == 3
    assert info['partial_masks'] is True

    ckpt = tmp_path / 'hetero_policy.pt'
    cli_train(
        str(walk_xml), str(clips_dir), str(ckpt),
        control = 'pd',
        dim_latent = 4,
        mask_dropout = 0.3,
        policy_mask = True,
        policy_mask_dropout = 0.5,
        iterations = 2,
        num_envs = 2,
        env_horizon = 16,
        warmup_steps = 10,
        warmup_wm_updates = 2,
        collect_steps = 6,
        batch_size = 8,
        wm_horizon = 3,
        imag_horizon = 6,
        wm_updates = 2,
        policy_updates = 2,
        critic_updates = 1,
        encoder_hiddens = (16,),
        policy_hiddens = (16,),
        wm_hiddens = (32,),
        critic_hiddens = (16,),
        eval_every = 2,
        device = 'cpu',
    )

    policy = load_policy(ckpt)
    assert policy.dim_mask == 1 + 2 + 2
    assert policy.metadata['use_masks'] is True

    # mask-free Stage-2 step works cleanly (mask = None -> zeros)
    act_no_mask, _ = policy.step(torch.zeros(policy.dim_obs), policy.sample_latent(1)[0])
    assert act_no_mask.shape == (2,)

    ev_full = cli_eval(str(ckpt), mocap = str(clips_dir / 'full.npz'), steps = 8, device = 'cpu')
    assert 'q_rmse' in ev_full and 'ee_mm' in ev_full

    ev_partial = cli_eval(str(ckpt), mocap = str(clips_dir / 'left_only.npz'), steps = 8, device = 'cpu')
    assert 'q_rmse' in ev_partial

    ev_kp = cli_eval(str(ckpt), mocap = str(clips_dir / 'feet_only.npz'), steps = 8, device = 'cpu')
    assert 'q_rmse' not in ev_kp and 'ee_mm' in ev_kp

    ev_rand = cli_eval(str(ckpt), random_latent = True, steps = 8, device = 'cpu')
    assert ev_rand['steps'] <= 8

    # explicit mask-free rollout via `use_policy_mask = False`
    from supermimic_pytorch.mujoco_mocap import MotionBank, build_robot, load_mocap_clips, rollout_policy
    rob = build_robot(walk_xml, **policy.metadata['robot_cfg'])
    bank = MotionBank.from_metadata(policy.metadata, rob)
    kp_clip = load_mocap_clips(str(clips_dir / 'feet_only.npz'), rob, target_fps = rob.fps)[0]
    states_mf, _, _ = rollout_policy(rob, policy, torch.device('cpu'), 4, ref_clip = kp_clip, bank = bank, use_policy_mask = False)
    assert len(states_mf) == 4

def test_cli_e2e_entrypoint_with_mock_video_and_json_config(tmp_path, monkeypatch):
    """End-to-end example of the full `supermimic` CLI workflow (inspect -> train -> eval -> Stage-2 load_policy)."""
    pytest.importorskip('mujoco')
    pytest.importorskip('memmap_replay_buffer')
    pytest.importorskip('fire')
    np = pytest.importorskip('numpy')
    import json
    import subprocess
    import sys
    import supermimic_pytorch.cli as cli_mod

    arm_xml = tmp_path / 'mock_robot.xml'
    arm_xml.write_text(FIXED_ARM_XML)

    clips_dir = tmp_path / 'mock_clips'
    clips_dir.mkdir()
    t = np.linspace(0., 1., 12, dtype = np.float32)
    np.savez(clips_dir / 'reach_a.npz', qpos = np.stack((0.4 * np.sin(2 * np.pi * t), 0.2 * np.cos(2 * np.pi * t)), axis = -1))
    np.savez(clips_dir / 'reach_b.npz', qpos = np.stack((-0.4 * np.sin(2 * np.pi * t), -0.2 * np.cos(2 * np.pi * t)), axis = -1))

    term_hook = tmp_path / 'mock_hooks.py'
    term_hook.write_text(
        """
import torch

def never_terminate(states, references = None):
    return torch.zeros(states.shape[:-1], device = states.device, dtype = states.dtype)
"""
    )

    ckpt_path = tmp_path / 'e2e_policy.pt'
    cfg_json = tmp_path / 'run.json'
    cfg_json.write_text(json.dumps(dict(
        robot = str(arm_xml),
        mocap = str(clips_dir),
        out = str(ckpt_path),
        termination_fn = f'{term_hook}:never_terminate',
        control = 'pd',
        dim_latent = 2,
        latent_space = 'box',
        mixed_precision = 'no',
        iterations = 1,
        num_envs = 2,
        env_horizon = 12,
        warmup_steps = 8,
        warmup_wm_updates = 1,
        collect_steps = 4,
        batch_size = 8,
        wm_horizon = 2,
        imag_horizon = 4,
        wm_updates = 1,
        policy_updates = 1,
        critic_updates = 1,
        encoder_hiddens = [16],
        policy_hiddens = [16],
        wm_hiddens = [24],
        critic_hiddens = [16],
        eval_every = 1,
        device = 'cpu',
    )))

    def run_cli(*args: str):
        proc = subprocess.run([sys.executable, '-m', 'supermimic_pytorch.cli', *args], capture_output = True, text = True)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    # 1. `supermimic inspect` via module entrypoint (`python -m supermimic_pytorch.cli inspect ...`)
    ins_out = run_cli('inspect', '--robot', str(arm_xml), '--mocap', str(clips_dir), '--imag-horizon', '4')
    assert 'Robot:' in ins_out and 'Mocap: 2 clip(s)' in ins_out

    # 2. `supermimic train --config run.json` via module entrypoint (same target used by `accelerate launch -m supermimic_pytorch.cli`)
    trn_out = run_cli('train', '--config', str(cfg_json))
    assert ckpt_path.exists()
    assert 'saved portable latent policy' in trn_out

    # 3. `supermimic eval` via module entrypoint (both held-out mocap and random-latent exploration)
    ev1_out = run_cli('eval', '--checkpoint', str(ckpt_path), '--mocap', str(clips_dir / 'reach_a.npz'), '--steps', '6', '--device', 'cpu')
    assert 'q_rmse=' in ev1_out

    ev2_out = run_cli('eval', '--checkpoint', str(ckpt_path), '--random-latent', '--steps', '6', '--device', 'cpu')
    assert 'completed 6 steps' in ev2_out

    # 4. mocked video export path in `eval(..., video = ...)` (works headlessly without OpenGL / ffmpeg)
    rendered = {}
    def fake_render(model, qposes, out_video, fps, has_free_root, size = 480):
        rendered.update(num_frames = len(qposes), path = out_video, fps = fps, size = size)
        out_video.parent.mkdir(parents = True, exist_ok = True)
        out_video.write_bytes(b'mock-mp4')

    monkeypatch.setattr(cli_mod, 'render_qpos_video', fake_render)
    vid_path = tmp_path / 'rollout.mp4'
    ev_vid = cli_mod.eval(str(ckpt_path), mocap = str(clips_dir / 'reach_a.npz'), steps = 6, video = str(vid_path), video_size = 128, device = 'cpu')
    assert vid_path.exists() and ev_vid['video'] == str(vid_path)
    assert rendered['num_frames'] == 7 and rendered['size'] == 128

    # 5. Stage-2 deployment: load portable checkpoint without MuJoCo and step closed-loop with z
    policy = load_policy(ckpt_path)
    obs = torch.zeros(policy.dim_obs)
    z = policy.sample_latent(1)[0]
    action, cache = policy.step(obs, z)
    assert action.shape == (2,)
    action2, _ = policy.step(obs, z, cache)
    assert action2.shape == (2,)
