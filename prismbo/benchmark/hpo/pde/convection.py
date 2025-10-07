
import numpy as np
import torch
from prismbo.benchmark.hpo.pde.base import PDE


class Convection2D(PDE):
    def __init__(self, beta: float = 30, seed: int = 42):
        self.beta = beta
        self.seed = seed

    @property
    def range(self):
        return torch.tensor([[0., 2 * torch.pi], [0., 1.]])

    @property
    def mu_range(self):
        return (0.005, 1.0)

    def pde(self, x, y, mu):
        x.requires_grad_(True)
        dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        dy_t = dy[:, 1:2]
        dy_x = dy[:, 0:1]
        return dy_t + self.beta * dy_x
    
    def ic(self, x, y, mu):
        return torch.mean(y ** 2)

    def bc(self, x, y, mu):
        return torch.mean(y ** 2)

    def analytic_func(self, x):
        return torch.sin(x[:, 0:1] - self.beta * x[:, 1:2])
