import torch
from torch import nn
import torch.nn.functional as F

import pytest
param = pytest.mark.parametrize

from x_mlps_pytorch import MLP

from supermimic_pytorch.supermimic import WorldModel, SuperMimic

def exists(v):
    return v is not None

def has_grads(module):
    return any(exists(p.grad) for p in module.parameters())

@param(
    'world_model_kwargs, use_sim_context, targets, check_identity',
    [
        (dict(), True, ('state',), False),
        (dict(dim_hiddens = (16,), pred_reward = True), True, ('state', 'terminal', 'reward'), False),
        (dict(dim_sim_context = 0, dim_hiddens = (16,)), False, (), False),
        (dict(dim_hiddens = (16,)), True, (), True)
    ],
    ids = ['default', 'terminal-and-reward', 'no-sim-context', 'residual-state-init']
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

@param('actor_condition', (False, True))
@param('actor_goal', (False, True))
@param('observation_noise_scale', (None, 0.5))
def test_actor_rollout(
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

    supermimic = SuperMimic(
        actor = actor,
        world_model = world_model,
        observation_noise_scale = observation_noise_scale
    )

    initial_state = torch.randn(2, 71)
    sim_context = torch.randn(2, 33)

    rollout = supermimic(initial_state, sim_context, actor_goals = goals, actor_cond = cond, num_rollout_steps = 4)

    assert rollout.states.shape == (2, 5, 71)
    assert rollout.actions.shape == (2, 4, 29)
    assert rollout.rewards.shape == (2, 4)
    assert rollout.terminals.shape == (2, 4)

    # trajectory starts at the given state

    assert torch.allclose(rollout.states[:, 0], initial_state)

    # bptt reaches the actor through the world model

    rollout.rewards.sum().backward()
    assert has_grads(actor)

    # but the world model weights stay frozen

    assert not has_grads(world_model)
