import torch

import pytest
param = pytest.mark.parametrize

from supermimic_pytorch.supermimic import (
    WorldModel,
    SuperMimic
)

def test_supermimic(
):
    world_model = WorldModel()

    state = torch.randn(3, 71)
    actions = torch.randn(3, 29)
    sim_context = torch.randn(3, 33)

    next_state = torch.randn(3, 71)

    # train

    loss, *_ = world_model(state, actions, sim_context, next_state = next_state)
    loss.backward()

    # evaluate

    preds = world_model(state, actions, sim_context)

    assert state.shape == preds.state.shape
