import torch
import torch.nn as nn
from prismbo.benchmark.hpo.pde.base import PDE



class Reaction2D(PDE):
    def __init__(self, rho: float = 5, seed: int = 42):
        self.rho = rho
        self.seed = seed
        
    @property
    def range(self):
        return torch.tensor([[0., 2 * torch.pi], [0., 1.]])

    def pde(self, x, y):
        x.requires_grad_(True)
        dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        dy_t = dy[:, 1:2]
        return dy_t - self.rho * y * (1 - y)
    
    def bc(self, x, y):
        return torch.mean(y ** 2)
    
    def ic(self, x, y):
        return torch.mean(y ** 2)
    
    def analytic_func(self, x):
        h = torch.exp(-(x[:, 0:1] - torch.pi) ** 2 / (2 * (torch.pi / 4) ** 2))
        return h * torch.exp(self.rho * x[:, 1:2]) / (h * torch.exp(self.rho * x[:, 1:2]) + 1 - h)

