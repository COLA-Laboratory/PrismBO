
import numpy as np
import torch
from prismbo.benchmark.hpo.pde.base import PDE
from typing import Callable, Optional, Tuple, Dict, Literal

class Diffusion2D(PDE):
    def __init__(self, seed: int = 42):
        self.seed = seed

    @property
    def range(self):
        return torch.tensor([[-1., 1.], [0., 1.]])

    def pde(self, x, y):
        x.requires_grad_(True)
        dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        dy_t = dy[:, 1:2]
        dy_x = dy[:, 0:1]
        dy_xx = torch.autograd.grad(dy_x, x, torch.ones_like(dy_x), create_graph=True)[0][:, 0:1]
        source_term = torch.exp(-x[:, 1:2]) * (
                    torch.sin(np.pi * x[:, 0:1]) - (np.pi ** 2) * torch.sin(np.pi * x[:, 0:1]))
        return dy_t - dy_xx + source_term

    def bc(self, x, y):
        return torch.mean(y ** 2)
    
    def ic(self, x, y):
        return torch.mean(y ** 2)
    
    def analytic_func(self, x):
        spatial = torch.sin(np.pi * x[:, 0:1])
        temporal = torch.exp(-x[:, 1:2])
        return spatial * temporal

