
import collections
import os
import random
import time
import json
from typing import Dict, Union
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from prismbo.benchmark.hpo import datasets

import prismbo.benchmark.hpo.misc as misc
from prismbo.agent.registry import problem_registry
from prismbo.benchmark.hpo.fast_data_loader import (FastDataLoader,
                                                     InfiniteDataLoader)
from prismbo.benchmark.problem_base.non_tab_problem import NonTabularProblem
from prismbo.space.fidelity_space import FidelitySpace
from prismbo.space.search_space import SearchSpace
from prismbo.space.variable import *
from prismbo.benchmark.hpo import algorithms
from prismbo.benchmark.hpo.hparams_registry import get_hparam_space, get_subpolicy_num
from prismbo.benchmark.hpo.augmentation import SamplerPolicy
from torch.utils.data import TensorDataset
from prismbo.benchmark.hpo.hpo_base import HPO_base


@problem_registry.register("HPO_ResNet18")
class HPO_ResNet18(HPO_base):    
    def __init__(
        self, task_name, budget_type, budget, seed, workload, description, **kwargs
        ):            
        algorithm = kwargs.pop('algorithm', 'ERM')
        architecture = kwargs.pop('architecture', 'resnet')
        model_size = kwargs.pop('model_size', 18)
        optimizer = kwargs.pop('optimizer', 'random')
        base_dir = kwargs.pop('base_dir', os.path.expanduser('~'))
        
        super(HPO_ResNet18, self).__init__(
            task_name=task_name, 
            budget_type=budget_type, 
            budget=budget, 
            seed=seed, 
            workload=workload, 
            algorithm=algorithm, 
            architecture=architecture, 
            model_size=model_size,
            optimizer=optimizer,
            base_dir=base_dir,
            description=description,
            **kwargs
        )

@problem_registry.register("HPO_ResNet32")
class HPO_ResNet32(HPO_base):    
    def __init__(
        self, task_name, budget_type, budget, seed, workload, description, **kwargs
        ):            
        algorithm = kwargs.pop('algorithm', 'ERM')
        architecture = kwargs.pop('architecture', 'resnet')
        model_size = kwargs.pop('model_size', 32)
        optimizer = kwargs.pop('optimizer', 'random')
        base_dir = kwargs.pop('base_dir', os.path.expanduser('~'))
        
        super(HPO_ResNet32, self).__init__(
            task_name=task_name, 
            budget_type=budget_type, 
            budget=budget, 
            seed=seed, 
            workload=workload, 
            algorithm=algorithm, 
            architecture=architecture, 
            model_size=model_size,
            optimizer=optimizer,
            base_dir=base_dir,
            description=description,
            **kwargs
        )
