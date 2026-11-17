import torch

import pytest
param = pytest.mark.parametrize

from x_mlps_pytorch import MLP

from supermimic_pytorch.supermimic import WorldModel, SuperMimic

def exists(v):
    return v is not None

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

def test_rollout():
    policy = MLP(71, 16, 29)

    world_model = WorldModel(
        dim_hiddens = (16,),
        pred_reward = True,
        pred_terminal = True,
        pred_residual_state = False
    )

    supermimic = SuperMimic(policy = policy, world_model = world_model)

    initial_state = torch.randn(2, 71)
    sim_context = torch.randn(2, 33)

    rollout = supermimic(initial_state, sim_context, num_rollout_steps = 4)

    assert rollout.states.shape == (2, 5, 71)
    assert rollout.actions.shape == (2, 4, 29)
    assert rollout.rewards.shape == (2, 4)
    assert rollout.terminals.shape == (2, 4)

    # trajectory starts at the given state

    assert torch.allclose(rollout.states[:, 0], initial_state)

    # bptt reaches the policy through the world model

    rollout.rewards.sum().backward()
    assert any(exists(p.grad) for p in policy.parameters())

    # but the world model weights stay frozen

    assert not any(exists(p.grad) for p in world_model.parameters())
