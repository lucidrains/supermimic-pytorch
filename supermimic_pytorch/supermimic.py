from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

# ein equations

from einops import rearrange
from torch_einops_utils.shape import size, shape

# mlps

from x_mlps_pytorch import MLP
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def compact(*arr):
    return tuple(filter(exists, arr))

# world model

class WorldModel(Module):
    def __init__(
        self,
        *,
        dim_state = 71,
        dim_action = 29,
        dim_sim_context = 33,
        dim_hiddens: tuple[int, ...] = (2048, 2048, 2048),
        normed_mlp = False,
        pred_residual_state = True
    ):
        super().__init__()

        mlp_klass = MLP if not normed_mlp else NormedMLP

        dim_in = dim_state + dim_action + dim_sim_context
        dim_out = dim_state

        self.transition_mlp = mlp_klass(dim_in, *dim_hiddens, dim_out)

        self.has_sim_context = dim_sim_context > 0
        self.pred_residual_state = pred_residual_state

    def forward(
        self,
        state,                                  # (b dim_state)
        actions,                                # (b dim_action)
        sim_context: Tensor | None = None       # (b dim_sim_context)
    ):
        assert self.has_sim_context or exists(sim_context)

        # world model is just a forward dynamics model f(s, a) -> s'
        # with simulation context (domain randomization presumably)

        pred = self.transition_mlp(compact(state, actions, sim_context))

        # whether predicting residual state or not

        if self.pred_residual_state:
            next_state = pred + state
        else:
            next_state = pred

        return next_state

# main class

class SuperMimic(Module):
    def __init__(
        self,
        *,
        policy: MLP | Module,
        world_model: WorldModel | Module
    ):
        super().__init__()

    def forward(
        self
    ):
        raise NotImplementedError
