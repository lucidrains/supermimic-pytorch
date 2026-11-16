import torch

import pytest
param = pytest.mark.parametrize

from supermimic_pytorch.supermimic import WorldModel

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
