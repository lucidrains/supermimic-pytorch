import pytest
import torch
from torch import nn

from einops import rearrange, repeat
from x_mlps_pytorch import MLP
from torch_einops_utils import tree_map_tensor

from supermimic_pytorch import (
    WorldModel,
    HumanoidActor,
    Critic,
    SuperMimic,
    HistoryWindow,
    action_smoothness_penalty
)

# constants

UNITREE_OBS_DIMS = dict(
    joint_pos = 29,
    joint_vel = 29,
    base_ang_vel = 3,
    projected_gravity = 3
)

UNITREE_OBS_NOISE = dict(
    joint_pos = 0.01,
    joint_vel = 0.5,
    base_ang_vel = 0.2,
    projected_gravity = 0.05
)

# helpers

def exists(v):
    return v is not None

def has_grads(module):
    return any(exists(p.grad) for p in module.parameters())

def has_nonzero_grads(module):
    return any(exists(p.grad) and p.grad.abs().sum() > 0 for p in module.parameters())

def make_unitree_obs(*leading_dims):
    return {name: torch.randn(*leading_dims, dim) for name, dim in UNITREE_OBS_DIMS.items()}

def make_unitree_state(*leading_dims):
    return dict(
        state = torch.randn(*leading_dims, 71),
        observation = make_unitree_obs(*leading_dims)
    )

# 1. core e2e flow

@pytest.mark.parametrize('dict_state', (False, True))
@pytest.mark.parametrize('use_critic, use_ema, normed_mlp', (
    (False, False, False),
    (True, False, False),
    (True, True, True)
))
@pytest.mark.parametrize('use_history', (False, True))
def test_e2e_supermimic(dict_state, use_critic, use_ema, normed_mlp, use_history):
    batch_size = 2
    horizon = 4
    dim_state = 22 if dict_state else 16
    dim_action = 6
    dim_sim_context = 8
    dim_goal = 5
    dim_cond = 4

    def make_state(*leading_dims):
        if dict_state:
            return dict(
                pos = torch.randn(*leading_dims, 10),
                vel = torch.randn(*leading_dims, 12)
            )
        return torch.randn(*leading_dims, dim_state)

    noise_scale = dict(pos = 0.05, vel = 0.2) if dict_state else 0.05

    world_model = WorldModel(
        dim_state = dim_state,
        dim_action = dim_action,
        dim_sim_context = dim_sim_context,
        dim_hiddens = (16, 16),
        normed_mlp = normed_mlp,
        pred_reward = True,
        pred_terminal = True
    )

    state = make_state(batch_size)
    sim_context = torch.randn(batch_size, dim_sim_context)

    # zero-init residual head starts as identity dynamics

    pred = world_model(state, torch.randn(batch_size, dim_action), sim_context)

    if dict_state:
        assert torch.allclose(pred.state['pos'], state['pos'])
    else:
        assert torch.allclose(pred.state, state)

    # autoregressive world model rollout fit

    wm_loss, preds, _ = world_model.rollout(
        state,
        torch.randn(batch_size, horizon, dim_action),
        make_state(batch_size, horizon),
        sim_context,
        next_terminals = torch.randint(0, 2, (batch_size, horizon)).float(),
        next_rewards = torch.randn(batch_size, horizon)
    )

    assert len(preds) == horizon

    wm_loss.backward()
    assert has_grads(world_model)
    world_model.zero_grad()

    # imagine rollout + critic + actor bptt

    obs_history = HistoryWindow(4, pad_mode = 'replicate') if use_history else None
    action_history = HistoryWindow(1, dim_action, lag = 1, pad_mode = 'learned') if use_history else None

    dim_actor_obs = (4 * dim_state + dim_action) if use_history else dim_state
    actor = MLP(dim_actor_obs + dim_goal + dim_cond, 32, dim_action, activation = nn.GELU())

    critic = None
    if use_critic:
        critic = Critic(
            dim_state = dim_state,
            dim_goal = dim_goal,
            dim_sim_context = dim_sim_context,
            dim_hiddens = (16,),
            normed_mlp = normed_mlp
        )

    supermimic = SuperMimic(
        actor = actor,
        world_model = world_model,
        critic = critic,
        use_ema_critic = use_ema,
        observation_noise_scale = noise_scale,
        obs_history = obs_history,
        action_history = action_history,
        action_smoothness_weight = 0.1 if use_history else 0.
    )

    goals = torch.randn(batch_size, horizon + 1, dim_goal)
    cond = torch.randn(batch_size, dim_cond)
    past_actions = torch.randn(batch_size, dim_action) if use_history else None

    rollout = supermimic(
        state,
        sim_context,
        actor_goals = goals,
        actor_cond = cond,
        past_actions = past_actions,
        num_rollout_timesteps = horizon
    )

    assert rollout.actions.shape == (batch_size, horizon, dim_action)
    assert rollout.rewards.shape == (batch_size, horizon)
    assert rollout.terminals.shape == (batch_size, horizon)

    if use_critic:
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
        supermimic.update_ema()

    final_state = tree_map_tensor(lambda t: t[:, -1], rollout.states) if use_critic else None
    final_goal = goals[:, -1] if use_critic else None

    actor_loss = supermimic.actor_loss(
        rollout.rewards,
        rollout.terminals,
        final_state = final_state,
        final_goal = final_goal,
        sim_context = sim_context,
        actions = rollout.actions,
        past_actions = past_actions
    )
    actor_loss.backward()

    assert has_nonzero_grads(actor)
    assert not has_grads(world_model)

    if use_critic:
        assert not has_grads(critic)
        assert not has_grads(supermimic.target_critic)

# 2. discrete & multi-discrete actions

@pytest.mark.parametrize('num_discrete_actions, split_actor_output', (
    (4, False),
    ((4, 3), False),
    ((4, 3), True)
))
def test_e2e_discrete_actions(num_discrete_actions, split_actor_output):
    action_dims = num_discrete_actions if isinstance(num_discrete_actions, tuple) else (num_discrete_actions,)
    total_dim = sum(action_dims)

    world_model = WorldModel(
        dim_state = 8,
        num_discrete_actions = num_discrete_actions,
        dim_sim_context = 0,
        dim_hiddens = (16,),
        pred_residual_state = False,
        pred_reward = True,
        pred_terminal = True
    )

    state = torch.randn(3, 8)
    per_head = [torch.randint(0, n, (3, 4)) for n in action_dims]
    action_indices = torch.stack(per_head, dim = -1) if len(action_dims) > 1 else per_head[0]

    wm_loss, _, _ = world_model.rollout(
        state,
        action_indices,
        torch.randn(3, 4, 8),
        next_terminals = torch.randint(0, 2, (3, 4)).float(),
        next_rewards = torch.randn(3, 4)
    )
    wm_loss.backward()

    assert has_grads(world_model)
    world_model.zero_grad()

    mlp = MLP(8 + total_dim, 16, total_dim, activation = nn.GELU())

    class SplitActor(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x).split(action_dims, dim = -1)

    actor = SplitActor(mlp) if split_actor_output else mlp

    critic = Critic(
        dim_state = 8,
        dim_goal = 0,
        dim_sim_context = 0,
        dim_hiddens = (16,)
    )

    supermimic = SuperMimic(
        actor = actor,
        world_model = world_model,
        critic = critic,
        use_ema_critic = True,
        action_history = 1
    )

    rollout = supermimic(
        state,
        past_actions = action_indices[:, 0],
        num_rollout_timesteps = 4
    )

    assert rollout.states.shape == (3, 5, 8)
    assert rollout.actions.shape == (3, 4, total_dim)

    actor_loss = supermimic.actor_loss(
        rollout.rewards,
        rollout.terminals,
        final_state = rollout.states[:, -1]
    )
    actor_loss.backward()

    assert has_grads(actor)
    assert not has_grads(world_model)
    assert not has_grads(critic)
    assert not has_grads(supermimic.target_critic)

# 3. Unitree G1 asymmetric POMDP

@pytest.mark.parametrize('mode', ('motion', 'velocity'))
@pytest.mark.parametrize('pad_mode', ('zero', 'replicate', 'learned'))
@pytest.mark.parametrize('action_window', (1, 10))
def test_e2e_unitree_humanoid(mode, pad_mode, action_window):
    batch_size = 2
    horizon = 5
    num_goal_frames = 10

    world_model = WorldModel(
        dim_state = 71 + 64,
        dim_action = 29,
        dim_sim_context = 33,
        dim_hiddens = (32,),
        pred_residual_state = False
    )

    actor = HumanoidActor(
        dim_obs = 64,
        num_obs_frames = 10,
        num_action_frames = action_window,
        num_goal_frames = num_goal_frames,
        dim_hiddens = (16,),
        dim_velocity_hiddens = (16,)
    )

    critic = Critic(
        dim_state = 71,
        dim_goal = 71 if mode == 'motion' else 0,
        dim_sim_context = 33,
        dim_hiddens = (16,)
    )

    obs_history = HistoryWindow(10, UNITREE_OBS_DIMS, pad_mode = pad_mode)
    action_history = HistoryWindow(action_window, 29, lag = 1, pad_mode = pad_mode)

    supermimic = SuperMimic(
        actor = actor,
        world_model = world_model,
        critic = critic,
        use_ema_critic = True,
        actor_obs_key = 'observation',
        critic_state_key = 'state',
        obs_history = obs_history,
        action_history = action_history,
        goal_window = num_goal_frames if mode == 'motion' else None,
        observation_noise_scale = UNITREE_OBS_NOISE,
        action_smoothness_weight = 0.1
    )

    sim_context = torch.randn(batch_size, 33)
    goals = torch.randn(batch_size, horizon + num_goal_frames, 71) if mode == 'motion' else None
    command = torch.randn(batch_size, 3) if mode == 'velocity' else None

    rollout = supermimic(
        make_unitree_state(batch_size),
        sim_context,
        actor_goals = goals,
        actor_cond = command,
        num_rollout_timesteps = horizon
    )

    assert rollout.states['state'].shape == (batch_size, horizon + 1, 71)
    assert rollout.actions.shape == (batch_size, horizon, 29)

    if mode == 'motion':
        rewards = -(rollout.states['state'][:, 1:] - goals[:, 1 : horizon + 1]).square().mean(dim = -1)
    else:
        rewards = -rollout.states['observation']['joint_vel'][:, 1:].square().mean(dim = -1)

    critic_loss = supermimic.critic_loss(
        rollout.states,
        rewards,
        rollout.terminals,
        goals = goals,
        sim_context = sim_context
    )
    critic_loss.backward()

    assert has_grads(critic)
    critic.zero_grad()
    supermimic.update_ema()

    final_state = tree_map_tensor(lambda t: t[:, -1], rollout.states)
    final_goal = goals[:, horizon] if exists(goals) else None

    actor_loss = supermimic.actor_loss(
        rewards,
        rollout.terminals,
        final_state = final_state,
        final_goal = final_goal,
        sim_context = sim_context,
        actions = rollout.actions
    )
    actor_loss.backward()

    assert has_nonzero_grads(actor)
    assert not has_grads(world_model)
    assert not has_grads(critic)
    assert not has_grads(supermimic.target_critic)

    if pad_mode == 'learned':
        assert has_nonzero_grads(obs_history)
        assert has_nonzero_grads(action_history)

def test_analytic_state_to_obs_matches_parallel_unfold():
    batch_size = 2
    horizon = 4

    world_model = WorldModel(
        dim_state = 71,
        dim_action = 29,
        dim_sim_context = 33,
        dim_hiddens = (16,),
        pred_residual_state = False
    )

    actor = HumanoidActor(
        dim_obs = 64,
        num_obs_frames = 10,
        num_action_frames = 1,
        num_goal_frames = 10,
        dim_hiddens = (16,)
    )

    def analytic_obs(state, sim_context):
        return dict(
            joint_pos = state[:, 7:36] - sim_context[:, 1:30],
            joint_vel = state[:, 42:71],
            base_ang_vel = state[:, 39:42],
            projected_gravity = state[:, 3:6]
        )

    supermimic = SuperMimic(
        actor = actor,
        world_model = world_model,
        state_to_obs = analytic_obs,
        obs_history = HistoryWindow(10, 64, pad_mode = 'learned'),
        action_history = HistoryWindow(1, 29, lag = 1, pad_mode = 'learned'),
        goal_window = 10,
        action_smoothness_weight = 0.1
    )

    sim_context = torch.randn(batch_size, 33)
    goals = torch.randn(batch_size, horizon + 10, 71)

    rollout = supermimic(
        torch.randn(batch_size, 71),
        sim_context,
        actor_goals = goals,
        num_rollout_timesteps = horizon
    )

    # verify streaming rollout matches parallel torch.unfold

    with torch.no_grad():
        flat_states = rearrange(rollout.states[:, :-1], 'b t d -> (b t) d')
        flat_context = repeat(sim_context, 'b c -> (b t) c', t = horizon)
        obs_seq = tree_map_tensor(
            lambda t: rearrange(t, '(b t) d -> b t d', b = batch_size),
            analytic_obs(flat_states, flat_context)
        )

        parallel_actions = actor(
            supermimic.obs_history(obs_seq),
            goal = supermimic.goal_window(goals)[:, :horizon],
            past_action = supermimic.action_history(rollout.actions)
        )

    assert torch.allclose(parallel_actions, rollout.actions, atol = 1e-5)

    rewards = -(rollout.states[:, 1:] - goals[:, 1 : horizon + 1]).square().mean(dim = -1)
    actor_loss = supermimic.actor_loss(rewards, rollout.terminals, actions = rollout.actions)
    actor_loss.backward()

    assert has_nonzero_grads(actor)
    assert has_nonzero_grads(supermimic.obs_history)
    assert has_nonzero_grads(supermimic.action_history)
    assert not has_grads(world_model)

# 4. HistoryWindow & action smoothness

@pytest.mark.parametrize('pad_mode, per_slot_pad', (
    ('zero', False),
    ('replicate', False),
    ('learned', False),
    ('learned', True)
))
def test_history_window_unfold_and_streaming(pad_mode, per_slot_pad):
    batch_size = 2
    seq_len = 7

    obs_window = HistoryWindow(4, UNITREE_OBS_DIMS, pad_mode = pad_mode, per_slot_pad = per_slot_pad)
    action_window = HistoryWindow(2, 29, lag = 1, pad_mode = pad_mode, per_slot_pad = per_slot_pad)

    if pad_mode == 'learned':
        with torch.no_grad():
            obs_window.pad_embed.normal_(std = 0.5)
            action_window.pad_embed.normal_(std = 0.5)

    obs_seq = make_unitree_obs(batch_size, seq_len)
    action_seq = torch.randn(batch_size, seq_len, 29)

    past_obs = make_unitree_obs(batch_size, 2)
    past_actions = torch.randn(batch_size, 2, 29)
    past_mask = torch.tensor([[True, True], [False, False]])

    episode_starts = torch.zeros(batch_size, seq_len, dtype = torch.bool)
    episode_starts[1, 3] = True

    batched_obs = obs_window(obs_seq, past = past_obs, past_mask = past_mask, episode_starts = episode_starts)
    batched_actions = action_window(action_seq, past = past_actions, past_mask = past_mask, episode_starts = episode_starts)

    assert batched_obs['joint_pos'].shape == (batch_size, seq_len, 4, 29)
    assert batched_actions.shape == (batch_size, seq_len, 2, 29)

    # stream step by step and assert exact equivalence

    _, obs_cache = obs_window.step(past_obs, mask = past_mask)
    _, action_cache = action_window.step(past_actions, mask = past_mask)

    streamed_obs = []
    streamed_actions = []

    for t in range(seq_len):
        done = episode_starts[:, t]
        step_obs = tree_map_tensor(lambda x: x[:, t], obs_seq)

        obs_win, obs_cache = obs_window.step(step_obs, obs_cache, reset = done)
        act_win, action_cache = action_window.step(cache = action_cache, reset = done)
        _, action_cache = action_window.step(action_seq[:, t], action_cache)

        streamed_obs.append(obs_win)
        streamed_actions.append(act_win)

    for key in UNITREE_OBS_DIMS:
        stacked = torch.stack([o[key] for o in streamed_obs], dim = 1)
        assert torch.allclose(batched_obs[key], stacked)

    assert torch.allclose(batched_actions, torch.stack(streamed_actions, dim = 1))

    if pad_mode == 'learned':
        loss = sum(o['joint_pos'].sum() for o in streamed_obs) + torch.stack(streamed_actions).sum()
        loss.backward()

        assert obs_window.pad_embed.grad.abs().sum() > 0
        assert action_window.pad_embed.grad.abs().sum() > 0

def test_future_window_and_action_smoothness():
    batch_size = 2
    seq_len = 6
    dim = 4

    goals = torch.randn(batch_size, seq_len + 2, dim)
    future_window, valid_mask = HistoryWindow(3, future = True)(goals, return_mask = True)

    future_window = future_window[:, :seq_len]
    valid_mask = valid_mask[:, :seq_len]

    assert future_window.shape == (batch_size, seq_len, 3, dim)
    assert valid_mask.all()
    assert torch.allclose(future_window[:, 2], goals[:, 2:5])

    # first and second order action smoothness

    actions = torch.randn(batch_size, seq_len, dim, requires_grad = True)
    past = torch.randn(batch_size, 2, dim)

    penalty_order1 = action_smoothness_penalty(actions, order = 1)
    assert torch.allclose(penalty_order1[:, 0], torch.zeros(batch_size))
    assert torch.allclose(penalty_order1[:, 1:], (actions[:, 1:] - actions[:, :-1]).square().sum(dim = -1))

    penalty_with_past = action_smoothness_penalty(actions, past_actions = past[:, -1], order = 1)
    assert torch.allclose(penalty_with_past[:, 0], (actions[:, 0] - past[:, -1]).square().sum(dim = -1))

    penalty_order2 = action_smoothness_penalty(actions, past_actions = past, order = 2)
    assert torch.allclose(penalty_order2[:, 0], (actions[:, 0] - 2 * past[:, 1] + past[:, 0]).square().sum(dim = -1))

    # episode boundaries zero out cross-episode penalty

    episode_starts = torch.zeros(batch_size, seq_len, dtype = torch.bool)
    episode_starts[0, 3] = True
    past_mask = torch.tensor([[True, True], [False, False]])

    penalty_reset = action_smoothness_penalty(
        actions,
        past_actions = past,
        past_mask = past_mask,
        episode_starts = episode_starts
    )

    assert penalty_reset[0, 0] > 0.
    assert penalty_reset[1, 0] == 0.
    assert penalty_reset[0, 3] == 0.
    assert penalty_reset[1, 3] > 0.

    penalty_with_past.sum().backward()
    assert exists(actions.grad)
    assert actions.grad.abs().sum() > 0

def test_single_state_single_action_and_compat():
    world_model = WorldModel(
        dim_state = 8,
        dim_action = 2,
        dim_sim_context = 0,
        dim_hiddens = (16,),
        pred_reward = True
    )

    for obs_window, action_window in ((None, None), (1, 0), (1, 1)):
        dim_in = 8 + 2 * (action_window or 0)
        actor = MLP(dim_in, 16, 2, activation = nn.GELU())

        supermimic = SuperMimic(
            actor = actor,
            world_model = world_model,
            obs_history = obs_window,
            action_history = action_window
        )

        rollout = supermimic(torch.randn(2, 8), num_rollout_timesteps = 4)

        assert rollout.states.shape == (2, 5, 8)
        assert rollout.actions.shape == (2, 4, 2)

        supermimic.actor_loss(rollout.rewards, rollout.terminals).backward()
        assert has_nonzero_grads(actor)

    # dict-of-tensors observation_noise_scale buffer + schema validation

    dict_world_model = WorldModel(
        dim_state = 10,
        dim_action = 2,
        dim_sim_context = 0,
        dim_hiddens = (16,),
        pred_reward = True
    )

    dict_supermimic = SuperMimic(
        actor = MLP(10, 16, 2, activation = nn.GELU()),
        world_model = dict_world_model,
        observation_noise_scale = dict(
            pos = torch.full((4,), 0.05),
            vel = torch.full((6,), 0.1)
        )
    )

    assert dict_supermimic.observation_noise_scale.shape == (10,)

    dict_state = dict(pos = torch.randn(2, 4), vel = torch.randn(2, 6))
    rollout = dict_supermimic(dict_state, num_rollout_timesteps = 3)
    assert rollout.states['pos'].shape == (2, 4, 4)
