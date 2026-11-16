import torch
from torch.nn import Module
import torch.nn.functional as F

# ein equations

from einops import rearrange
from torch_einops_utils.shape import size,shape

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class SuperMimic(Module):
    def __init__(self):
        super().__init__()
