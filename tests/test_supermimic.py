import torch
from torch import nn

import pytest
param = pytest.mark.parametrize

from x_mlps_pytorch import MLP

from supermimic_pytorch.supermimic import WorldModel, SuperMimic, Critic

def exists(v):
    return v is not None

def has_grads(module):
    return any(exists(p.grad) for p in module.parameters())

def test_humanoid_actor_forward():
    from x_mlps_pytorch import MimoMLP
    from supermimic_pytorch.supermimic import HumanoidActor

    actor = HumanoidActor(dim_hiddens = (16,))

    obs = torch.randn(2, 10, 93)
    goal = torch.randn(2, 10, 71)

    action = actor(obs, goal)

    assert isinstance(actor.net, MimoMLP)
    assert action.shape == (2, 29)

@param(
    'world_model_kwargs, use_sim_context, targets, check_identity',
    [
        (dict(), True, ('state',), False),
        (dict(dim_hiddens = (16,), pred_reward = True), True, ('state', 'terminal', 'reward'), False),
        (dict(dim_hiddens = (16, 16), normed_mlp = True, pred_reward = True), True, ('state', 'terminal', 'reward'), False),
        (dict(dim_sim_context = 0, dim_hiddens = (16,)), False, (), False),
        (dict(dim_hiddens = (16,)), True, (), True)
    ],
    ids = ['default', 'terminal-and-reward', 'normed-mlp', 'no-sim-context', 'residual-state-init']
)
def test_world_model(
    world_model_kwargs,
    use_sim_context,
    targets,
    check_identity
):
    world_model = WorldModel(**world_model_kwargs)

    state = torch.randn(3, 71)
    actions = torch.randn(3, 29)
    sim_context = torch.randn(3, 33) if use_sim_context else None

    # targets for the heads being trained

    next_state = torch.randn(3, 71) if 'state' in targets else None
    next_terminal = torch.randint(0, 2, (3,)).float() if 'terminal' in targets else None
    next_reward = torch.randn(3) if 'reward' in targets else None

    out = world_model(
        state,
        actions,
        sim_context,
        next_state = next_state,
        next_terminal = next_terminal,
        next_reward = next_reward
    )

    has_targets = any(exists(t) for t in (next_state, next_terminal, next_reward))

    if has_targets:
        loss, preds, losses = out
        loss.backward()

        # only active losses are returned

        assert exists(losses.state) == exists(next_state)
        assert exists(losses.terminal) == exists(next_terminal)
        assert exists(losses.reward) == exists(next_reward)

        # total loss sums the active losses (all weights are 1.)

        assert torch.allclose(loss, sum(component for component in losses if exists(component)))
    else:
        preds = out

    assert preds.state.shape == state.shape

    # zero init residual head -> prediction is exactly the input state

    if check_identity:
        assert torch.allclose(preds.state, state)

def test_world_model_rollout():
    world_model = WorldModel(
        dim_hiddens = (16,),
        pred_reward = True,
        pred_terminal = True,
        pred_residual_state = False
    )

    state = torch.randn(3, 71)
    actions = torch.randn(3, 8, 29)
    next_states = torch.randn(3, 8, 71)
    next_terminals = torch.randint(0, 2, (3, 8)).float()
    next_rewards = torch.randn(3, 8)
    sim_context = torch.randn(3, 33)

    total_loss, preds, losses = world_model.rollout(
        state, actions, next_states, sim_context, next_terminals, next_rewards
    )

    assert len(preds) == len(losses) == 8
    assert preds[0].state.shape == (3, 71)

    # total loss is the summed per step loss

    assert torch.allclose(total_loss, sum(sum(component for component in l if exists(component)) for l in losses))

    # autoregressive -> step 1 rolled from step 0's prediction

    first_pred, second_pred, *_ = preds
    _, second_action, *_ = actions.unbind(dim = 1)
    _, second_next_state, *_ = next_states.unbind(dim = 1)

    _, step_pred, _ = world_model(first_pred.state, second_action, sim_context, second_next_state)

    assert torch.allclose(step_pred.state, second_pred.state)

    total_loss.backward()
    assert has_grads(world_model)

@param(
    'use_critic, use_ema_critic, critic_normed_mlp',
    (
        (False, False, False),
        (True, False, False),
        (True, True, True)
    )
)
@param('actor_condition', (False, True))
@param('actor_goal', (False, True))
@param('observation_noise_scale', (None, 0.5))
def test_actor_rollout(
    use_critic,
    use_ema_critic,
    critic_normed_mlp,
    actor_condition,
    actor_goal,
    observation_noise_scale
):

    # maybe condition

    dim_cond = dim_goal = 0

    cond = None
    goals = None

    if actor_condition:
        dim_cond = 8
        cond = torch.randn(2, 8)

    if actor_goal:
        goals = torch.randn(2, 5, 3)
        dim_goal = goals.shape[-1]

    # networks

    actor = MLP(71 + dim_cond + dim_goal, 2048, 1024, 512, 256, 29, activation = nn.GELU())

    world_model = WorldModel(
        dim_hiddens = (1024, 1024),
        pred_reward = True,
        pred_terminal = True,
        pred_residual_state = False
    )

    critic = Critic(
        dim_state = 71,
        dim_goal = dim_goal,
        dim_sim_context = 33,
        dim_hiddens = (16,),
        normed_mlp = critic_normed_mlp
    ) if use_critic else None

    supermimic = SuperMimic(
        actor = actor,
        world_model = world_model,
        critic = critic,
        use_ema_critic = use_ema_critic,
        observation_noise_scale = observation_noise_scale
    )

    initial_state = torch.randn(2, 71)
    sim_context = torch.randn(2, 33)

    rollout = supermimic(initial_state, sim_context, actor_goals = goals, actor_cond = cond, num_rollout_timesteps = 4)

    assert rollout.states.shape == (2, 5, 71)
    assert rollout.actions.shape == (2, 4, 29)
    assert rollout.rewards.shape == (2, 4)
    assert rollout.terminals.shape == (2, 4)

    # trajectory starts at the given state

    assert torch.allclose(rollout.states[:, 0], initial_state)

    # actor loss bptt through frozen world model (critic bootstrap at s_H)

    rewards = rollout.states[:, 1:].sum(dim = -1)
    final_state = final_goal = None

    if use_critic:
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

        final_state = rollout.states[:, -1]
        final_goal = goals[:, -1] if exists(goals) else None

        # bootstrap flows into actor through frozen critic, even with zero rewards

        zero_rewards = torch.zeros_like(rewards)
        bootstrap_only_loss = supermimic.actor_loss(
            zero_rewards,
            rollout.terminals,
            final_state = final_state,
            final_goal = final_goal,
            sim_context = sim_context
        )
        bootstrap_only_loss.backward(retain_graph = True)

        assert any(exists(p.grad) and p.grad.abs().sum() > 0 for p in actor.parameters())
        assert not has_grads(critic)
        assert not has_grads(supermimic.target_critic)
        actor.zero_grad()

    actor_loss = supermimic.actor_loss(
        rewards,
        rollout.terminals,
        final_state = final_state,
        final_goal = final_goal,
        sim_context = sim_context
    )
    actor_loss.backward()
    assert has_grads(actor)

    # world model and critic stay frozen during actor backward

    assert not has_grads(world_model)

    if use_critic:
        assert not has_grads(critic)
        assert not has_grads(supermimic.target_critic)

@param('num_discrete_actions', (4, (4, 3)))
def test_world_model_discrete_actions(num_discrete_actions):
    world_model = WorldModel(
        dim_state = 8,
        num_discrete_actions = num_discrete_actions,
        dim_sim_context = 0,
        dim_hiddens = (16,),
        pred_residual_state = False,
        pred_reward = True,
        pred_terminal = True
    )

    action_dims = num_discrete_actions if isinstance(num_discrete_actions, tuple) else (num_discrete_actions,)
    is_multi = len(action_dims) > 1

    state = torch.randn(3, 8)
    per_head = [torch.randint(0, n, (3,)) for n in action_dims]
    indices = torch.stack(per_head, dim = -1) if is_multi else per_head[0]
    one_hot = torch.cat([torch.nn.functional.one_hot(a, n).float() for a, n in zip(per_head, action_dims)], dim = -1)

    # integer indices and one-hot produce identical predictions

    with torch.no_grad():
        pred_idx = world_model(state, indices)
        pred_oh = world_model(state, one_hot)

    assert torch.allclose(pred_idx.state, pred_oh.state)
    assert torch.allclose(pred_idx.reward, pred_oh.reward)
    assert torch.allclose(pred_idx.terminal, pred_oh.terminal)

    # rollout with integer action sequences

    seq_per_head = [torch.randint(0, n, (3, 5)) for n in action_dims]
    seq_indices = torch.stack(seq_per_head, dim = -1) if is_multi else seq_per_head[0]

    total_loss, preds, losses = world_model.rollout(
        state,
        seq_indices,
        torch.randn(3, 5, 8),
        next_terminals = torch.randint(0, 2, (3, 5)).float(),
        next_rewards = torch.randn(3, 5)
    )

    assert len(preds) == len(losses) == 5
    assert preds[0].state.shape == (3, 8)
    total_loss.backward()
    assert has_grads(world_model)

    with pytest.raises(AssertionError):
        WorldModel(dim_state = 8, dim_action = 2, num_discrete_actions = num_discrete_actions)

@param('num_discrete_actions, split_actor_output', ((4, False), ((4, 3), False), ((4, 3), True)))
def test_supermimic_discrete_bptt(num_discrete_actions, split_actor_output):
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

    critic = Critic(dim_state = 8, dim_goal = 0, dim_sim_context = 0, dim_hiddens = (16,))
    mlp = MLP(8, 16, total_dim, activation = nn.GELU())

    class TupleActor(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x).split(action_dims, dim = -1)

    actor = TupleActor(mlp) if split_actor_output else mlp

    supermimic = SuperMimic(
        actor = actor,
        world_model = world_model,
        critic = critic,
        use_ema_critic = True
    )

    rollout = supermimic(torch.randn(4, 8), num_rollout_timesteps = 4)

    assert rollout.states.shape == (4, 5, 8)
    assert rollout.actions.shape == (4, 4, total_dim)

    loss = supermimic.actor_loss(rollout.rewards, rollout.terminals, final_state = rollout.states[:, -1])
    loss.backward()

    assert has_grads(actor)
    assert not has_grads(world_model)
    assert not has_grads(critic)
    assert not has_grads(supermimic.target_critic)
