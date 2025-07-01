import torch
from external.fsbo.fsbo_modules import FSBO
import numpy as np
import os
import json
import argparse
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
import argparse
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)
from prismbo.optimizer.initialization.random import RandomSampler

import random

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


def meta_train(epochs, task_name, train_data, valid_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rootdir     = f'./results/FSBO_{timestamp}'
    np.random.seed(123)
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    os.makedirs(os.path.join(rootdir,"checkpoints"), exist_ok=True)
    checkpoint_path = os.path.join(rootdir,"checkpoints","FSBO2", f"{task_name}")
    fsbo_model = FSBO(train_data = train_data, valid_data = valid_data, checkpoint_path = checkpoint_path)
    fsbo_model.meta_train(epochs)


def get_source_data(task_name, services):
    task_source = {'Sphere': 'Sphere_source', 'Rastrigin': 'Rastrigin_source', 'Schwefel': 'Schwefel_source', 'Ackley': 'Ackley_source', 'Griewank': 'Griewank_source', 'Rosenbrock': 'Rosenbrock_source'}
    source_name = task_source[task_name]

    datasets = {}
    with open('Results/datasets.txt', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith("Experiment:"):
                experiment_name = lines[i].strip().split(":")[1].strip()
                dataset_list = []
                i += 1
                while i < len(lines) and not lines[i].startswith("-----"):
                    dataset_list.append(lines[i].strip())
                    i += 1
                datasets[experiment_name] = dataset_list
    Exper_folder = 'Results'
    ab = AnalysisBase(Exper_folder, datasets, services.data_manager)
    ab.read_data_from_db()
    
    train_data = {}
    datasets = ab._all_data[source_name]
    metadata_info = ab._data_infos[source_name]
    sub_dataset_id = 0
    for dataset_name, data in datasets.items():
        objectives = metadata_info[dataset_name]["objectives"]
        obj = objectives[0]["name"]

        obj_data = [d[obj] for d in data]
        var_data = [[d[var["name"]] for var in metadata_info[dataset_name]["variables"]] for d in data]
        input_size = metadata_info[dataset_name]['num_variables']
        X = np.array(var_data)
        Y = np.array(obj_data)
        train_data[sub_dataset_id] = {'X': X, 'y': Y}
        sub_dataset_id += 1
    return train_data


def get_benchmark(
    benchmark_name: str,
    seed: int,
    workload: int,
    output_file: str,
    output_noise: float = 0.0,
    params_source: List[Dict[str, float]] = None,
    params_target: Dict[str, float] = None,
):
    """Create the benchmark object."""
    target_task = task_class(
            task_name=benchmark_name,
            budget_type='FEs',
            budget=budget,
            seed=seed,
            workload=workload,
            params={'input_dim': input_dim}
        )

    task = task_wrapper(output_file=output_file, task_name=benchmark_name, workload=workload, seed=seed, search_space=target_task.configuration_space, task=target_task)
    
    space = target_task.configuration_space
    


    return task, space



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
    input_dim = 10
    tasks = [
        # ('Rastrigin', Rastrigin, input_dim),
        ('Schwefel', Schwefel, input_dim),
        ('Ackley', Ackley, input_dim),
        ('Griewank', Griewank, input_dim),
        ('Rosenbrock', Rosenbrock, input_dim),
    ]
    budget = input_dim * 11
    initial_num = input_dim * 11
    dtype = torch.float
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    services = Services(None, None, None)
    services._initialize_modules()

    configurations = services.configer.get_configuration()

    workloads = [0, 1, 2, 3, 4, 5]  # Different workloads
    seeds = [0, 1, 2, 3, 4, 5]  
    epochs = 10

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'results/FSBO_{timestamp}'
    os.mkdir(output_dir)
    
    for task_name, task_class, input_dim in tqdm.tqdm(tasks, desc="Processing tasks"):
        source_data = get_source_data(task_name, services)
        train_data = {}
        valid_data = {}
        set_id = 0
        train_ratio = 0.8  # 80% for training, 20% for validation
        
        for name, data in source_data.items():
            n_samples = len(data['X'])
            n_train = int(n_samples * train_ratio)
            
            # Split into train and validation
            train_data[set_id] = {
                'X': data['X'][:n_train],
                'y': data['y'][:n_train].reshape(-1, 1)
            }
            valid_data[set_id] = {
                'X': data['X'][n_train:],
                'y': data['y'][n_train:].reshape(-1, 1)
            }
            set_id += 1
        meta_train(epochs, task_name, train_data, valid_data)
        for workload in workloads:
            for seed in seeds:
                output_file = f'{output_dir}/{task_name}_workload{workload}_seed{seed}.json'
                task, space = get_benchmark(task_name, seed, workload, output_file)

                run_optimization(task_name, task, space, output_file, seed, workload, initial_num, budget, output_dir)
