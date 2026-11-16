import torch
import pytest

from supermimic_pytorch.supermimic import (
    WorldModel,
    SuperMimic
)

def test_supermimic():
    world_model = WorldModel()

    state = torch.randn(3, 71)
    actions = torch.randn(3, 29)
    sim_context = torch.randn(3, 33)

    next_state = world_model(state, actions, sim_context)

    assert state.shape == next_state.shape
