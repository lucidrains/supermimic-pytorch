from functools import partial
import pickle

import pytest
import torch
from einops import rearrange

from supermimic_pytorch import (
    ActionSmoothnessTerm,
    AnalyticalReward,
    Critic,
    ForwardKinematics,
    HumanoidActor,
    StateTerm,
    SuperMimic,
    TrackingTerm,
    UNITREE_G1_OBS_NOISE,
    WorldModel,
    fall_termination,
    motion_tracking_termination,
    projected_gravity,
    quat_rotate,
    quat_rotate_inverse,
    rotation_squared_error,
    unitree_g1_state_to_obs,
    unitree_motion_tracking_reward,
    unitree_velocity_tracking_reward,
    unpack_unitree_g1_state,
    world_to_heading_frame
)
from supermimic_pytorch.rewards import axis_angle_to_quat, quat_to_rotmat

def has_grads(module):
    return any(p.grad is not None for p in module.parameters())

def has_nonzero_grads(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

# 1. differentiable geometry (SO(3), quaternions, 6D rot, SO(2) angles)

def test_rotation_squared_error_representations_and_gradients():
    torch.manual_seed(0)

    axes = torch.randn(4, 3)
    angles = torch.tensor([0., 1e-5, 0.7, torch.pi - 1e-2])

    q_id = torch.tensor([[1., 0., 0., 0.]]).expand(4, 4)
    q_rot = axis_angle_to_quat(axes, angles).requires_grad_(True)

    err_quat = rotation_squared_error(q_rot, q_id)
    err_neg = rotation_squared_error(-q_rot, q_id)

    R_id, R_rot = quat_to_rotmat(q_id), quat_to_rotmat(q_rot)
    err_mat = rotation_squared_error(R_rot, R_id)
    err_6d = rotation_squared_error(
        rearrange(R_rot[..., :, :2], 'b i j -> b (j i)'),
        rearrange(R_id[..., :, :2], 'b i j -> b (j i)')
    )

    expected = angles.square()
    for err in (err_quat, err_neg, err_mat, err_6d):
        assert torch.allclose(err, expected, atol = 1e-5)

    # NaN-free gradients everywhere including theta = 0 and theta -> pi

    err_quat.sum().backward()
    assert torch.isfinite(q_rot.grad).all()
    assert torch.allclose(q_rot.grad[0], torch.zeros(4), atol = 1e-6)

    q_pi = torch.tensor([[0., 1., 0., 0.]], requires_grad = True)
    e_pi = rotation_squared_error(q_pi, torch.tensor([[1., 0., 0., 0.]]))
    assert torch.allclose(e_pi, torch.tensor([torch.pi ** 2]), atol = 1e-4)
    e_pi.sum().backward()
    assert torch.isfinite(q_pi.grad).all()

    # 2D angle wraparound (-pi and +pi are identical)

    a = torch.tensor([torch.pi - 0.1], requires_grad = True)
    e_ang = rotation_squared_error(a, torch.tensor([-torch.pi + 0.1]))
    assert torch.allclose(e_ang, torch.tensor([0.2 ** 2]), atol = 1e-5)
    e_ang.sum().backward()
    assert torch.isfinite(a.grad).all()

def test_quaternion_frames_broadcast_and_gravity():
    q_up = torch.tensor([[1., 0., 0., 0.]])
    q_pitch = axis_angle_to_quat(torch.tensor([[0., 1., 0.]]), torch.tensor([torch.pi / 2]))
    q_flip = axis_angle_to_quat(torch.tensor([[1., 0., 0.]]), torch.tensor([torch.pi]))

    assert torch.allclose(projected_gravity(q_up), torch.tensor([[0., 0., -1.]]), atol = 1e-6)
    assert torch.allclose(projected_gravity(q_pitch), torch.tensor([[1., 0., 0.]]), atol = 1e-6)
    assert torch.allclose(projected_gravity(q_flip), torch.tensor([[0., 0., 1.]]), atol = 1e-6)

    q_batch = torch.cat((q_up, q_pitch, q_flip), dim = 0)
    v_single = torch.tensor([1.2, -0.4, 0.7])
    rotated = quat_rotate(q_batch, v_single)
    assert rotated.shape == (3, 3)
    assert torch.allclose(quat_rotate_inverse(q_batch, rotated), v_single.expand(3, 3), atol = 1e-6)
    assert torch.allclose(quat_rotate_inverse(q_pitch[0], quat_rotate(q_pitch[0], rotated)), rotated, atol = 1e-6)

    q_yaw90 = axis_angle_to_quat(torch.tensor([[0., 0., 1.]]), torch.tensor([torch.pi / 2]))
    v_heading = world_to_heading_frame(q_yaw90, torch.tensor([[0., 2., 0.5]]))
    assert torch.allclose(v_heading, torch.tensor([[2., 0., 0.5]]), atol = 1e-6)

# 2. differentiable ForwardKinematics verified against real MuJoCo

@pytest.mark.parametrize('env_name', ('HalfCheetah-v5', 'Humanoid-v5'))
def test_forward_kinematics_matches_mujoco(env_name):
    import gymnasium as gym
    import mujoco
    import numpy as np

    env = gym.make(env_name)
    model, data = env.unwrapped.model, env.unwrapped.data
    fk = ForwardKinematics.from_mujoco(model)
    fk_world = ForwardKinematics.from_mujoco(model, include_world = True)

    rng = np.random.default_rng(7)
    qpos_batch, qvel_batch, mj_pos, mj_quat, mj_lin_vel, mj_ang_vel = [], [], [], [], [], []

    for _ in range(4):
        qpos = model.qpos0.copy() + rng.normal(scale = 0.25, size = model.nq)
        if fk.has_free_joint:
            qpos[3:7] /= np.linalg.norm(qpos[3:7])
        qvel = rng.normal(scale = 0.6, size = model.nv)

        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(model, data)

        qpos_batch.append(qpos)
        qvel_batch.append(qvel)
        mj_pos.append(data.xpos[1:].copy())
        mj_quat.append(data.xquat[1:].copy())

        lvels, avels, vel6 = [], [], np.zeros(6)
        for b in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, vel6, 0)
            avels.append(vel6[:3].copy())
            lvels.append(vel6[3:].copy())
        mj_lin_vel.append(lvels)
        mj_ang_vel.append(avels)

    env.close()

    qpos_t = torch.tensor(np.asarray(qpos_batch), dtype = torch.float32, requires_grad = True)
    qvel_t = torch.tensor(np.asarray(qvel_batch), dtype = torch.float32, requires_grad = True)

    out = fk(qpos = qpos_t, qvel = qvel_t)
    out_world = fk_world(qpos = qpos_t.detach(), qvel = qvel_t.detach())

    assert torch.allclose(out['body_pos'], torch.tensor(np.asarray(mj_pos), dtype = torch.float32), atol = 1e-5)
    assert torch.allclose(out['body_lin_vel'], torch.tensor(np.asarray(mj_lin_vel), dtype = torch.float32), atol = 1e-5)
    assert torch.allclose(out['body_ang_vel'], torch.tensor(np.asarray(mj_ang_vel), dtype = torch.float32), atol = 1e-5)
    assert torch.allclose(out_world['body_pos'][:, 1:], out['body_pos'], atol = 1e-5)

    quat_err = rotation_squared_error(out['body_quat'], torch.tensor(np.asarray(mj_quat), dtype = torch.float32))
    assert torch.allclose(quat_err, torch.zeros_like(quat_err), atol = 1e-5)

    loss = out['body_pos'].square().sum() + out['body_lin_vel'].square().sum() + out['body_ang_vel'].square().sum()
    loss.backward()
    assert torch.isfinite(qpos_t.grad).all() and qpos_t.grad.abs().sum() > 0
    assert torch.isfinite(qvel_t.grad).all() and qvel_t.grad.abs().sum() > 0

def test_forward_kinematics_fixed_base_arm():
    fk = ForwardKinematics([-1, 0, 1], [[0., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
    q = torch.tensor([[torch.pi / 2, 0.]], requires_grad = True)
    out = fk(q)
    assert torch.allclose(out['body_pos'][0, 2], torch.tensor([1., 1., 0.]), atol = 1e-5)
    out['body_pos'].sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0

# 3. generalized analytical rewards (flat subset, nested dicts, multi-body, custom callables, name disambiguation)

def test_generalized_analytical_rewards():
    batch_size, horizon = 3, 6
    states = torch.randn(batch_size, horizon, 10, requires_grad = True)
    actions = torch.randn(batch_size, horizon, 4, requires_grad = True)
    ref_subset = torch.cat((states.detach()[..., :3], states.detach()[..., 5:6]), dim = -1) + 0.1

    reward_fn = AnalyticalReward([
        TrackingTerm(slice(0, 3), ref_key = slice(0, 3), std = 0.5, weight = 1.0, name = 'pos'),
        TrackingTerm(5, ref_key = 3, metric = 'angle', std = 0.4, weight = 0.5, name = 'angle'),
        ActionSmoothnessTerm(weight = -0.1),
        lambda s, ref, **kw: -0.01 * s.square().mean(dim = -1)
    ])

    total, breakdown = reward_fn(states, ref_subset, actions = actions, return_breakdown = True)
    assert total.shape == (batch_size, horizon)
    assert set(breakdown) == {'pos', 'angle', 'action_smoothness', 'term_3'}
    assert torch.allclose(total, sum(breakdown.values()))

    _, bd_no_act = reward_fn(states, ref_subset, return_breakdown = True)
    assert 'action_smoothness' not in bd_no_act

    # multiple unnamed TrackingTerms and lambdas never collide or drop terms

    multi_unnamed = AnalyticalReward([
        TrackingTerm(0, ref_key = 0, std = 1.0, weight = 1.0),
        TrackingTerm(1, ref_key = 1, std = 1.0, weight = 2.0),
        lambda s, ref, **kw: torch.ones(s.shape[:-1]),
        lambda s, ref, **kw: torch.full(s.shape[:-1], 3.0)
    ])
    tot_u, bd_u = multi_unnamed(states.detach(), states.detach(), return_breakdown = True)
    assert len(bd_u) == 4 and torch.allclose(tot_u, sum(bd_u.values()))
    assert torch.allclose(tot_u, torch.full((batch_size, horizon), 7.0))

    # auto-matching subset reference, static broadcast goal, nested dict + weighted multi-body

    r_sub = TrackingTerm([0, 2, 5], std = 0.5, kernel = 'rational')(states, states.detach()[..., [0, 2, 5]])
    assert torch.allclose(r_sub, torch.ones(batch_size, horizon))

    # negative integer selector works cleanly (including -1)

    r_last = TrackingTerm(-1, kernel = 'neg')(states, torch.zeros(batch_size, horizon, 1))
    assert torch.allclose(r_last, -states[..., -1].square())

    r_static = TrackingTerm(slice(0, 3), std = 1.0, kernel = 'neg')(states, torch.zeros(batch_size, 3))
    assert torch.allclose(r_static, -states[..., :3].square().sum(dim = -1))

    nested_s = dict(obs = dict(bodies = torch.zeros(batch_size, horizon, 3, 2, requires_grad = True)))
    nested_r = dict(obs = dict(bodies = torch.ones(batch_size, horizon, 3, 2)))
    term_bodies = TrackingTerm(('obs', 'bodies'), kernel = 'neg', item_weights = [1., 2., 1.], dim_weights = [1., 3.])
    assert torch.allclose(term_bodies(nested_s, nested_r), torch.full((batch_size, horizon), -4.0))

    # StateTerm works standalone and zero-weight overrides drop inactive terms

    st = StateTerm(lambda s: s.square().mean(dim = -1), weight = -0.5, name = 'energy')
    assert torch.allclose(st(states.detach()), -0.5 * states.detach().square().mean(dim = -1))

    total.sum().backward()
    assert torch.isfinite(states.grad).all() and states.grad.abs().sum() > 0
    assert torch.isfinite(actions.grad).all() and actions.grad.abs().sum() > 0

# 4. Unitree G1 motion-tracking, velocity-tracking, analytic obs, terminations & e2e SuperMimic BPTT

def make_unitree_synthetic_fk(num_joints = 29):
    parents = [-1] + [0 if i % 5 == 0 else i for i in range(num_joints)]
    offsets = [[0., 0., 0.]] + [[0.05 * ((i % 3) - 1), 0.05 * ((i % 2) * 2 - 1), -0.15] for i in range(num_joints)]
    return ForwardKinematics(
        parents,
        offsets,
        joint_axes = [[0., 1., 0.]] * num_joints,
        free_root = True,
        tracked_bodies = range(0, 30, 3)
    )

def test_unitree_motion_and_velocity_rewards_exact_and_bptt():
    torch.manual_seed(1)
    batch_size, horizon = 2, 5

    fk = make_unitree_synthetic_fk()
    lower, upper = torch.full((29,), -1.5), torch.full((29,), 1.5)
    motion_reward = unitree_motion_tracking_reward(fk = fk, joint_limits = (lower, upper))
    assert fk in set(motion_reward.modules())
    assert pickle.loads(pickle.dumps(motion_reward)) is not None

    # exact tracking yields max Table A.2 reward * dt, and weights override cleanly

    state71 = torch.zeros(batch_size, horizon, 71)
    state71[..., 2], state71[..., 3] = 0.8, 1.0
    ref_dict = fk(unpack_unitree_g1_state(state71))
    actions = torch.zeros(batch_size, horizon, 29)

    r_perfect, bd = motion_reward(state71, ref_dict, actions = actions, return_breakdown = True)
    assert torch.allclose(r_perfect, torch.full((batch_size, horizon), 5.0 * 0.02), atol = 1e-6)
    assert torch.allclose(bd['joint_limit'], torch.zeros(batch_size, horizon))
    assert torch.allclose(bd['action_smoothness'], torch.zeros(batch_size, horizon))

    override_fn = unitree_motion_tracking_reward(fk = fk, weights = dict(root_pos = 2.0))
    r_override = override_fn(state71, ref_dict, actions = actions)
    assert torch.allclose(r_override, torch.full((batch_size, horizon), 6.5 * 0.02), atol = 1e-6)

    # joint limit violation applies -2 * violation * dt

    state_oob = state71.clone()
    state_oob[..., 7] = 2.0
    _, bd_oob = motion_reward(state_oob, ref_dict, actions = actions, return_breakdown = True)
    assert torch.allclose(bd_oob['joint_limit'], torch.full((batch_size, horizon), -2.0 * 0.5 * 0.02), atol = 1e-6)

    # Table A.6 velocity reward: forward pitch unpenalized, sideways roll penalized, static (b, 3) command broadcasts

    vel_reward = unitree_velocity_tracking_reward(joint_limits = (lower, upper))
    assert pickle.loads(pickle.dumps(vel_reward)) is not None
    cmd = torch.tensor([[1.0, 0.0, 0.5]]).expand(batch_size, 3)

    s_pitch = state71.clone()
    s_pitch[..., 3:7] = axis_angle_to_quat(torch.tensor([0., 1., 0.]), torch.tensor(0.4))
    s_pitch[..., 36], s_pitch[..., 41] = 1.0, 0.5
    _, bd_pitch = vel_reward(s_pitch, cmd, actions = actions, return_breakdown = True)
    assert torch.allclose(bd_pitch['torso_sideways_lean'], torch.zeros(batch_size, horizon), atol = 1e-6)
    assert torch.allclose(bd_pitch['lin_vel'], torch.full((batch_size, horizon), 2.0 * 0.02), atol = 1e-5)
    assert torch.allclose(bd_pitch['yaw_rate'], torch.full((batch_size, horizon), 2.0 * 0.02), atol = 1e-5)

    s_roll = state71.clone()
    s_roll[..., 3:7] = axis_angle_to_quat(torch.tensor([1., 0., 0.]), torch.tensor(0.5))
    _, bd_roll = vel_reward(s_roll, cmd, actions = actions, return_breakdown = True)
    expected_lean = -4.0 * (torch.sin(torch.tensor(0.5)) ** 2) * 0.02
    assert torch.allclose(bd_roll['torso_sideways_lean'], torch.full((batch_size, horizon), expected_lean), atol = 1e-5)

    # Appendix A & Table A.6 terminations (hard + differentiable soft)

    s_fallen = state71.clone()
    s_fallen[..., 2] = 0.2
    assert (motion_tracking_termination(s_fallen, ref_dict) == 1.).all()
    assert (fall_termination(s_fallen) == 1.).all()
    assert (motion_tracking_termination(state71, ref_dict) == 0.).all()
    assert (fall_termination(state71) == 0.).all()

    s_soft = state71.clone().requires_grad_(True)
    fall_termination(s_soft, temperature = 0.05).sum().backward()
    assert torch.isfinite(s_soft.grad).all() and s_soft.grad.abs().sum() > 0

    # full e2e SuperMimic BPTT with analytic obs head + analytic reward + soft analytic termination

    world_model = WorldModel(
        dim_state = 71,
        dim_action = 29,
        dim_sim_context = 33,
        dim_hiddens = (32,),
        pred_reward = False,
        pred_terminal = False
    )
    actor = HumanoidActor(
        dim_obs = 64,
        num_obs_frames = 4,
        num_action_frames = 1,
        num_goal_frames = 4,
        dim_hiddens = (16,)
    )
    critic = Critic(dim_state = 71, dim_goal = 71, dim_sim_context = 33, dim_hiddens = (16,))

    supermimic = SuperMimic(
        actor = actor,
        world_model = world_model,
        critic = critic,
        state_to_obs = unitree_g1_state_to_obs,
        obs_history = 4,
        action_history = 1,
        goal_window = 4,
        reward_fn = motion_reward,
        termination_fn = partial(motion_tracking_termination, temperature = 0.05),
        observation_noise_scale = UNITREE_G1_OBS_NOISE
    )

    sim_context = torch.randn(batch_size, 33) * 0.01
    goals = torch.randn(batch_size, horizon + 4, 71) * 0.1
    rollout = supermimic(
        state71[:, 0],
        sim_context = sim_context,
        actor_goals = goals,
        references = ref_dict,
        num_rollout_timesteps = horizon
    )
    assert rollout.rewards.shape == rollout.terminals.shape == (batch_size, horizon)

    critic_loss = supermimic.critic_loss(
        rollout.states,
        rollout.rewards,
        rollout.terminals,
        goals = goals,
        sim_context = sim_context
    )
    critic_loss.backward()
    assert has_grads(critic)
    critic.zero_grad()

    actor_loss = supermimic.actor_loss(
        rollout.rewards,
        rollout.terminals,
        final_state = rollout.states[:, -1],
        final_goal = goals[:, horizon],
        sim_context = sim_context
    )
    actor_loss.backward()
    assert has_nonzero_grads(actor) and not has_grads(world_model) and not has_grads(critic)
