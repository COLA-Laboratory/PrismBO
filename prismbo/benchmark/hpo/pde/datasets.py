from typing import Optional


import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from prismbo.benchmark.hpo.pde.sample import get_data_2d, get_data, get_data_with_mu
from prismbo.benchmark.hpo.pde import PROBLEMS, pde_classes

class Datasets:
    num_workers = 0

    @staticmethod
    def pde_dataset(pde_name: str, train: bool) -> Dataset:
        if pde_name[-2:] == "2D":
            num_x, num_y = 32, 32
            range_x = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
            points, bc_points, ic_points = get_data_2d(num_x, num_y, range_x)
            # 切割数据集，90%用于train，10%用于test
            num_points = points.shape[0]
            split_idx = int(num_points * 0.9)

            # 分别计算每个边界/初值的split
            num_bc_points = bc_points.shape[0]
            split_bc_points = int(num_bc_points * 0.9)
            num_ic_points = ic_points.shape[0]
            split_ic_points = int(num_ic_points * 0.9)
            

            if train:
                points = points[:split_idx]
                bc_points = bc_points[:split_bc_points]
                ic_points = ic_points[:split_ic_points]
            else:
                points = points[split_idx:]
                bc_points = bc_points[split_bc_points:]
                ic_points = ic_points[split_ic_points:]

        else:
            raise ValueError(f"Unsupported PDE name: {pde_name}")
        
        dataset_points = torch.utils.data.TensorDataset(points)
        dataset_bc_points = torch.utils.data.TensorDataset(bc_points)
        dataset_ic_points = torch.utils.data.TensorDataset(ic_points)
        
        return dataset_points, dataset_bc_points, dataset_ic_points

    @staticmethod
    def pde_parametric_dataset_with_mu(pde_name: str, mu: float, train: bool, batch_size: int) -> Dataset:

        if pde_name[-2:] == "2D":
            if pde_name in pde_classes:
                pde = pde_classes[pde_name]()
            else:
                raise ValueError(f"Unknown PDE name: {pde_name}")

            if train:
                num_x, num_y = 100, 100
                num_bc = 1000
                num_ic = 1000
            else:
                num_x, num_y = 50, 50
                num_bc = 2500
                num_ic = 2500

            range = pde.range

            X_f, mu_f, X_b, mu_b, X_i, mu_i = get_data_with_mu(
                mu=mu, num_x=num_x, num_y=num_y, num_bc=num_bc, num_ic=num_ic, range=range)

            def split_to_batches(tensor, batch_size):
                N, M = tensor.shape
                assert N % batch_size == 0, f"Total size {N} not divisible by batch_size {batch_size}"
                C = N // batch_size
                return tensor.view(batch_size, C, M)

            if X_f.shape[0] > 0:
                X_f = split_to_batches(X_f, batch_size)
                mu_f = split_to_batches(mu_f, batch_size)
            if X_b.shape[0] > 0:
                X_b = split_to_batches(X_b, batch_size)
                mu_b = split_to_batches(mu_b, batch_size)
            if X_i.shape[0] > 0:
                X_i = split_to_batches(X_i, batch_size)
                mu_i = split_to_batches(mu_i, batch_size)

            dataset_X_f = torch.utils.data.TensorDataset(X_f)
            dataset_mu_f = torch.utils.data.TensorDataset(mu_f)
            dataset_X_b = torch.utils.data.TensorDataset(X_b)
            dataset_mu_b = torch.utils.data.TensorDataset(mu_b)
            dataset_X_i = torch.utils.data.TensorDataset(X_i)
            dataset_mu_i = torch.utils.data.TensorDataset(mu_i)
            # dataset_nus = torch.utils.data.TensorDataset(nus)

            return dataset_X_f, dataset_mu_f, dataset_X_b, dataset_mu_b, dataset_X_i, dataset_mu_i
        
    @staticmethod
    def pde_parametric_dataset(pde_name: str, train: bool, batch_size: int) -> Dataset:

        if pde_name[-2:] == "2D":
            if pde_name in PROBLEMS:
                pde = pde_classes[pde_name]()
            else:
                raise ValueError(f"Unknown PDE name: {pde_name}")

            if train:
                num_x, num_y = 64, 64
                num_bc = 256
                num_ic = 256
            else:
                num_x, num_y = 32, 32
                num_bc = 64
                num_ic = 64

            range = pde.range

            X_f, mu_f, X_b, mu_b, X_i, mu_i, nus = get_data(
                num_mu=64, num_x=num_x, num_y=num_y, num_bc=num_bc, num_ic=num_ic, range=range, range_mu=pde.mu_range)

            dataset_X_f = torch.utils.data.TensorDataset(X_f)
            dataset_mu_f = torch.utils.data.TensorDataset(mu_f)
            dataset_X_b = torch.utils.data.TensorDataset(X_b)
            dataset_mu_b = torch.utils.data.TensorDataset(mu_b)
            dataset_X_i = torch.utils.data.TensorDataset(X_i)
            dataset_mu_i = torch.utils.data.TensorDataset(mu_i)
            dataset_nus = torch.utils.data.TensorDataset(nus)

            return dataset_X_f, dataset_mu_f, dataset_X_b, dataset_mu_b, dataset_X_i, dataset_mu_i, dataset_nus
    
    @classmethod
    def pde_dataloader(cls, pde_name: str, train: bool, batch_size: int, shuffle=True):
        dataset_X_f, dataset_mu_f, dataset_X_b, dataset_mu_b, dataset_X_i, dataset_mu_i, dataset_nus = Datasets.pde_parametric_dataset(pde_name, train, batch_size)
        combined_dataset = torch.utils.data.TensorDataset(
            dataset_X_f.tensors[0],
            dataset_mu_f.tensors[0],
            dataset_X_b.tensors[0],
            dataset_mu_b.tensors[0],
            dataset_X_i.tensors[0],
            dataset_mu_i.tensors[0],
            dataset_nus.tensors[0],
        )
        dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=1, shuffle=shuffle)
        
        return dataloader
    
    
    @classmethod
    def pde_dataloader_with_mu(cls, pde_name: str, mu: float, train: bool, batch_size: int, shuffle=True):
        dataset_X_f, dataset_mu_f, dataset_X_b, dataset_mu_b, dataset_X_i, dataset_mu_i = Datasets.pde_parametric_dataset_with_mu(pde_name, mu, train, batch_size)
        combined_dataset = torch.utils.data.TensorDataset(
            dataset_X_f.tensors[0],
            dataset_mu_f.tensors[0],
            dataset_X_b.tensors[0],
            dataset_mu_b.tensors[0],
            dataset_X_i.tensors[0],
            dataset_mu_i.tensors[0],
            # dataset_nus.tensors[0],
        )
        dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle)
        
        return dataloader
