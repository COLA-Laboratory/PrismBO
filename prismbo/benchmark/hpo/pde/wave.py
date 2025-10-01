import numpy as np
import torch
from prismbo.benchmark.hpo.pde.base import PDE
from typing import Callable, Optional, Tuple, Dict, Literal



class Wave2D(PDE):
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.beta = 3

    @property
    def range(self):
        return torch.tensor([[0., 1.], [0., 1.]])

    def pde(self, x, u):
        x.requires_grad_(True)
        du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        du_t = du[:, 1:2]
        du_x = du[:, 0:1]
        du_tt = torch.autograd.grad(du_t, x, torch.ones_like(du_t), create_graph=True)[0][:, 1:2]
        du_xx = torch.autograd.grad(du_x, x, torch.ones_like(du_x), create_graph=True)[0][:, 0:1]
        return du_tt - 4 * du_xx
    
    def ic(self, x, u):
        x.requires_grad_(True)
        du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        ic1 = du[:, 1:2]
        ic2 = torch.sin(torch.pi * x[:, 0:1]) + 0.5 * torch.sin(self.beta * torch.pi * x[:, 0:1]) - u
        return torch.mean((ic1) ** 2) + torch.mean((ic2) ** 2)

    def bc(self, x, u):
        return torch.mean(u ** 2)
    
    def analytic_func(self, x):
        x_coord = x[:, 0:1]
        t_coord = x[:, 1:2]
        pi = torch.tensor(np.pi)
        s_beta = np.sqrt(self.beta)
        term1 = torch.sin(pi * x_coord) * torch.cos(2 * pi * t_coord)
        term2 = 0.5 * torch.sin(self.beta * pi * x_coord) * torch.cos(self.beta * 2 * pi * t_coord)
        return term1 + term2



