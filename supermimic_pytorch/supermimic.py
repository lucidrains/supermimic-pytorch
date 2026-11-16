from __future__ import annotations

import torch
from torch.nn import Module
import torch.nn.functional as F

# ein equations

from einops import rearrange
from torch_einops_utils.shape import size, shape

# mlps

from x_mlps_pytorch import MLP, create_mlp

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# components

class WorldModel(Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        state
    ):
        raise NotImplementedError

# main class

class SuperMimic(Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(
        self
    ):
        raise NotImplementedError
