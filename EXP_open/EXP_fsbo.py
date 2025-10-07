import torch
from external.fsbo.fsbo_modules import FSBO
import numpy as np
import os
import json
import argparse
from functools import partial

import warnings
from datetime import datetime
from prismbo.agent.services import Services
from prismbo.analysis.analysisbase import AnalysisBase
from typing import List, Tuple, Callable, Dict, Hashable
import tqdm
from prismbo.benchmark.synthetic.singleobj import *
import matplotlib.pyplot as plt
import numpy as np
from external.fsbo.hpob_handler import HPOBHandler
from external.fsbo.fsbo_modules import DeepKernelGP
import os 
from GPy.kern import RBF

import argparse
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)
from prismbo.optimizer.initialization.random import RandomSampler
from external.transfergpbo.bo.run_bo import run_bo

import random

from external.transfergpbo.models import (
    TaskData,
    WrapperBase,
    MHGP,
    SHGP,
    BHGP,
)
from external.transfergpbo.bo.run_bo import run_bo
from external.transfergpbo import models, benchmarks
from external.transfergpbo.parameters import parameters as params

from emukit.core import ParameterSpace, ContinuousParameter

from prismbo.benchmark.hpo import * 
from prismbo.benchmark.csstuning.compiler import LLVMTuning, GCCTuning
from prismbo.benchmark.csstuning.dbms import MySQLTuning
from prismbo.benchmark.synthetic.singleobj import *

import os
import json
import time

task_class_dict = {
    'Ackley': Ackley,
    'Rastrigin': Rastrigin,
    'Rosenbrock': Rosenbrock,
    'XGBoost': XGBoostBenchmark,
    'HPO_PINN': HPO_PINN,
    'HPO_ResNet18': HPO_ResNet18,
    'HPO_ResNet32': HPO_ResNet32,
    'CSSTuning_GCC': GCCTuning,
    'CSSTuning_LLVM': LLVMTuning,
    'CSSTuning_MySQL': MySQLTuning,
}

CONFIG_FILE = os.path.join("config", "running_config.json")


def read_config():
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
        return {
            'tasks': config.get('tasks'),
            'optimizer': config.get('optimizer'),
            'seeds': config.get('seeds', '42'),
            'remote': config.get('remote', False),
            'server_url': config.get('server_url', ''),
            'experimentName': config.get('experimentName', ''),
        }
import json


def get_model(
    model_name: str, space: ParameterSpace) -> WrapperBase:
    """Create the model object."""
    model_class = getattr(models, model_name)
    if model_class == MHGP or model_class == SHGP or model_class == BHGP:
        model = model_class(space.dimensionality)
    else:
        kernel = RBF(space.dimensionality)
        model = model_class(kernel=kernel)
    model = WrapperBase(model)
    return model

def make_objective(task):
    def objective(config, seed: int = 0):
        cfg = config.get_dictionary() if hasattr(config, "get_dictionary") else dict(config)
        configuration = [cfg[v] for v in task.configuration_space.variables_order]
        configuration = task.configuration_space.map_to_design_space(configuration)
        result = task.objective_function(configuration=configuration)
        return result['f1']
    return objective

# Define the configuration space
def get_configspace(task):
    original_ranges = task.configuration_space.original_ranges
    hyperparameters = [cs.UniformHyperparameter(param_name, lower=param_range[0], upper=param_range[1]) for param_name, param_range in original_ranges.items() ]
    space = cs.ConfigurationSpace(hyperparameters)
    
    return space


class objective_function4GPBO():
    def __init__(self, search_space, task) -> None:
        self.search_space = search_space
        self.task = task
    
    
    def f(self, X, **kwargs):
        Y = []
        query_datasets = [self.search_space.map_to_design_space(sample) for sample in X]
        Y = np.array([[self.task.objective_function(data)['f1'] for data in query_datasets]])
        return Y


class task_wrapper():
    def __init__(self, search_space, task, task_name, workload, seed, output_file) -> None:
        self.search_space = search_space
        self.task = task
        self.task_name = task_name
        self.workload = workload
        self.seed = seed
        self.output_file = output_file
    
    def get_configuration_space(self):
        return self.search_space

    def objective_function(self, X, **kwargs):
        # Map each sample to design space and evaluate
        query_datasets = [self.search_space.map_to_design_space(sample) for sample in X]
        Y = np.array([[self.task.objective_function(data)['f1'] for data in query_datasets]])
        
        # Read existing trajectory from file or create new one
        try:
            with open(self.output_file, 'r') as f:
                data = json.load(f)
                trajectory = data['result']['history']
        except:
            trajectory = []
            
        # Add each new evaluation to trajectory
        for i in range(len(X)):
            trajectory.append({
                'iteration': len(trajectory),
                'params': X[i].tolist(),
                'loss': float(Y[0][i])
            })
        
        # Update result with best seen so far
        best_idx = np.argmax([t['loss'] for t in trajectory])
        result = {
            'best_params': trajectory[best_idx]['params'],
            'best_value': trajectory[best_idx]['loss'], 
            'history': trajectory
        }
        
        # Save updated result to file
        with open(self.output_file, 'w') as f:
            json.dump({
                'task_name': self.task_name,
                'workload': self.workload,
                'seed': self.seed,
                'result': result
            }, f, indent=2)
            
        return Y


def meta_train(rootdir, epochs, task_name, train_data, valid_data):
    np.random.seed(123)
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    checkpoint_path = os.path.join(rootdir,"checkpoints.pt")
    fsbo_model = FSBO(train_data = train_data, valid_data = valid_data, checkpoint_path = checkpoint_path)
    fsbo_model.meta_train(epochs)
    fsbo_model.save_checkpoint(checkpoint_path)




def run_optimization(task_name, task, space, output_file, seed, workload, initial_budget, num_function_evals, rootdir):
    n_trials = num_function_evals
    load_model = True  #if load_Model == False the DeepKernel is randomly initialized 
    verbose = True
    discrete_spaces = False

    torch_seed = random.randint(0,100000)
    search_space = task.get_configuration_space()
    sampler = RandomSampler(config = {})
    ini_X = sampler.sample(search_space, n_points=initial_budget)
    # query_datasets = [search_space.map_to_design_space(sample) for sample in initial_samples]
    ini_Y = task.objective_function(ini_X)
    
    # ini_X = np.array([v for k,v in search_space.variables_order.items()])
    # ini_Y = np.array([v for k,v in search_space.variables_order.items()])

    dim = len(space.variables_order)

    hpob_hdlr = HPOBHandler(root_dir=rootdir, mode="v3-test" , surrogates_dir=rootdir+"/saved-surrogates/", ini_X=ini_X, ini_Y=ini_Y)


    #loads pretrained model from the checkpoint "FSBO",
    checkpoint = os.path.join(rootdir,"checkpoints","FSBO2", f"{task_name}")

    #define the DeepKernelGP as HPO method
    method = DeepKernelGP(dim, torch_seed, epochs= 100, load_model = load_model, 
                                            checkpoint = checkpoint, verbose = verbose)

    acc = hpob_hdlr.evaluate_continuous(task, method, seed = seed, n_trials = n_trials )




if __name__ == "__main__":
   # Experiment settings
    config = read_config()
    seeds = [int(s.strip()) for s in config['seeds'].split(',')]

    # Create results directory
    results_dir = f"results/fsbo_{config['experimentName']}"
    os.makedirs(results_dir, exist_ok=True)

    rootdir     = f'./external/fsbo'
    # checkpoint = os.path.join(rootdir,"checkpoints","FSBO2", '7609')
    load_model = True
    checkpoint_dir     = f'./results/fsbo/checkpoints/FSBO2/'
    os.makedirs(checkpoint_dir, exist_ok=True)


    # Run experiments
    all_results = {}
    source_data = {}
    set_id = 0
    for task_info in config['tasks']:
        task_name = task_info['name']
        task_results = {}
        workloads = [int(w.strip()) for w in task_info['workloads'].split(',')]

        for workload in workloads:
            workload_results = []

            for seed in seeds:
                print(f"Running {task_name} with workload {workload}, seed {seed}")
                task = task_class_dict[task_name](
                    task_name=task_name,
                    budget_type=task_info['budget_type'],
                    budget=task_info['budget'],
                    seed=seed,
                    workload=workload,
                    description=task_info['description'],
                    params={'input_dim': int(task_info['num_vars'])}
                )
                
                if len(source_data) > 0:
                    # 合并所有source_data中的X和y，然后划分为train和valid
                    all_X = []
                    all_y = []
                    for d in source_data.values():
                        all_X.append(np.array(d['X']))
                        all_y.append(np.array(d['y']))
                    all_X = np.vstack(all_X)
                    all_y = np.vstack(all_y)
                    # 打乱数据
                    idx = np.arange(all_X.shape[0])
                    np.random.shuffle(idx)
                    all_X = all_X[idx]
                    all_y = all_y[idx]
                    # 80% train, 20% valid
                    split = int(0.8 * all_X.shape[0])
                    train_data = {'train': {'X': all_X[:split], 'y': all_y[:split]}}
                    valid_data = {'valid': {'X': all_X[split:], 'y': all_y[split:]}}
                    meta_train(checkpoint_dir, 2000, task_name, train_data, valid_data)
                    search_space = task.get_configuration_space()
                    sampler = RandomSampler(init_num=20, config = {})
                    ini_X = sampler.sample(search_space)
                    query_datasets = [search_space.map_to_design_space(sample) for sample in ini_X]
                    ini_Y = np.array([task.f(query)['f1'] for query in query_datasets]).reshape(-1, 1)
                    #loads pretrained model from the checkpoint "FSBO",
                    hpob_hdlr = HPOBHandler(root_dir=rootdir, mode="v3-test" , surrogates_dir=rootdir+"/saved-surrogates/", ini_X=ini_X, ini_Y=ini_Y)
                    #define the DeepKernelGP as HPO method
                    method = DeepKernelGP(int(task_info['num_vars']), seed, epochs= 100, load_model = load_model, checkpoint = checkpoint_dir, verbose = True)

                    acc = hpob_hdlr.evaluate_continuous(task, method, seed = seed, n_trials = int(task_info['budget']) )
                else:
                    obj = objective_function4GPBO(search_space=task.configuration_space, task=task)

                    space = ParameterSpace([ContinuousParameter(k, var[0], var[1]) for k, var in task.configuration_space.original_ranges.items()])
                    model = get_model('GPBO', space)
                    start_time = time.time()
                    
                    regret, X, Y = run_bo(
                        experiment_fun=partial(obj.f, output_noise=0.1),
                        model=model,
                        space=space,
                        num_iter=int(task_info['budget']),
                    )
                    end_time = time.time()
                    optimization_time = end_time - start_time
                    source_data[set_id] = {
                        'X': X,
                        'y': Y
                    }
                    set_id += 1
                
    #             if len(source_data) > 0:
    #                 start_time = time.time()
    #                 model.meta_fit(source_data)

    #                 # Run BO and return the regret, X, Y
    #                 regret, X, Y = run_bo(
    #                     experiment_fun=partial(obj.f, output_noise=0.1),
    #                     model=model,
    #                     space=space,
    #                     num_iter=int(task_info['budget']),
    #                 )
    #                 end_time = time.time()
    #                 optimization_time = end_time - start_time
    #             else:
    #                 model = get_model('GPBO', space)
    #                 start_time = time.time()
                    
    #                 regret, X, Y = run_bo(
    #                     experiment_fun=partial(obj.f, output_noise=0.1),
    #                     model=model,
    #                     space=space,
    #                     num_iter=int(task_info['budget']),
    #                 )
    #                 end_time = time.time()
    #                 optimization_time = end_time - start_time
    
    
    
    
    
    
    # for task_name, task_class, input_dim in tqdm.tqdm(tasks, desc="Processing tasks"):
    #     source_data = get_source_data(task_name, services)
    #     train_data = {}
    #     valid_data = {}
    #     set_id = 0
    #     train_ratio = 0.8  # 80% for training, 20% for validation
        
    #     for name, data in source_data.items():
    #         n_samples = len(data['X'])
    #         n_train = int(n_samples * train_ratio)
            
    #         # Split into train and validation
    #         train_data[set_id] = {
    #             'X': data['X'][:n_train],
    #             'y': data['y'][:n_train].reshape(-1, 1)
    #         }
    #         valid_data[set_id] = {
    #             'X': data['X'][n_train:],
    #             'y': data['y'][n_train:].reshape(-1, 1)
    #         }
    #         set_id += 1
    #     meta_train(epochs, task_name, train_data, valid_data)
    #     for workload in workloads:
    #         for seed in seeds:
    #             output_file = f'{output_dir}/{task_name}_workload{workload}_seed{seed}.json'
    #             task, space = get_benchmark(task_name, seed, workload, output_file)

    #             run_optimization(task_name, task, space, output_file, seed, workload, initial_num, budget, output_dir)
