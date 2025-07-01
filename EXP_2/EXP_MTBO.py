# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import List, Tuple, Callable, Dict, Hashable
from functools import partial
import random

from emukit.core import ParameterSpace
from GPy.kern import RBF
import torch
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


from torch import Tensor
from typing import List, Dict, Tuple, Callable, Hashable
import json

from prismbo.agent.services import Services
from prismbo.analysis.analysisbase import AnalysisBase

from prismbo.benchmark.synthetic.singleobj import *

from datetime import datetime
import os
import tqdm
# suppress GPyTorch warnings about adding jitter
import warnings
warnings.filterwarnings("ignore", "^.*jitter.*", category=RuntimeWarning)




def get_model(
    model_name: str, space: ParameterSpace, source_data: Dict[Hashable, TaskData]
) -> WrapperBase:
    """Create the model object."""
    model_class = getattr(models, model_name)
    if model_class == MHGP or model_class == SHGP or model_class == BHGP:
        model = model_class(space.dimensionality)
    else:
        kernel = RBF(space.dimensionality)
        model = model_class(kernel=kernel)
    model = WrapperBase(model)
    model.meta_fit(source_data)
    return model


def run_experiment(num_source_points, benchmark_name, num_steps, output_noise=0.1, params_source=None, params_target=None) -> List[float]:
    """The actual experiment code."""
    num_source_points = num_source_points
    technique = "MTGP"
    benchmark_name = benchmark_name
    num_steps = num_steps
    output_noise = output_noise
    params_source = params_source
    params_target = params_target

    # Initialize the benchmark and model
    f_target, source_data, space = get_benchmark(
        benchmark_name, num_source_points, output_noise, params_source, params_target
    )
    model = get_model(technique, space, source_data)

    # Run BO and return the regret
    return run_bo(
        experiment_fun=partial(f_target, output_noise=output_noise),
        model=model,
        space=space,
        num_iter=num_steps,
    )

class objective_function():
    def __init__(self, search_space, task) -> None:
        self.search_space = search_space
        self.task = task
    
    
    def f(self, X, **kwargs):
        Y = []
        query_datasets = [self.search_space.map_to_design_space(sample) for sample in X]
        Y = np.array([[self.task.objective_function(data)['f1'] for data in query_datasets]])
        return Y

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
    # datasets = ab._all_data[source_name]
    # metadata_info = ab._data_infos[source_name]
    
    dataset_key = random.choice(list(ab._all_data[source_name].keys()))
    datasets = {dataset_key: ab._all_data[source_name][dataset_key]}
    
    metadata_info = {dataset_key: ab._data_infos[source_name][dataset_key]}
    sub_dataset_id = 0
    for dataset_name, data in datasets.items():
        objectives = metadata_info[dataset_name]["objectives"]
        obj = objectives[0]["name"]

        obj_data = [d[obj] for d in data]
        var_data = [[d[var["name"]] for var in metadata_info[dataset_name]["variables"]] for d in data]
        input_size = metadata_info[dataset_name]['num_variables']
        X = np.array(var_data)
        Y = np.array(obj_data)
        train_data[sub_dataset_id] = {'X': X, 'Y': Y}
        sub_dataset_id += 1
    return train_data



if __name__ == "__main__":
    input_dim = 10
    tasks = [
        ('Rastrigin', Rastrigin, input_dim),
        ('Schwefel', Schwefel, input_dim),
        ('Ackley', Ackley, input_dim),
        ('Griewank', Griewank, input_dim),
        ('Rosenbrock', Rosenbrock, input_dim),
    ]
    budget = input_dim * 22
    initial_num = input_dim * 11
    dtype = torch.float
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    
    services = Services(None, None, None)
    services._initialize_modules()
    
    configurations = services.configer.get_configuration()

    workloads = [0, 1, 2, 3, 4, 5]  # Different workloads
    seeds = [0, 1, 2, 3, 4, 5]  
    
    
    def get_benchmark(
        benchmark_name: str,
        num_source_points: List[int],
        seed: int,
        workload: int,
        output_noise: float = 0.0,
        params_source: List[Dict[str, float]] = None,
        params_target: Dict[str, float] = None,
    ) -> Tuple[Callable, Dict[Hashable, TaskData], ParameterSpace]:
        """Create the benchmark object."""
        num_source_functions = len(num_source_points)

        target_task = task_class(
                task_name=benchmark_name,
                budget_type='FEs',
                budget=budget,
                seed=seed,
                workload=workload,
                params={'input_dim': input_dim}
            )
                    
        obj = objective_function(search_space=target_task.configuration_space, task=target_task)

        # Get parameter space
        space = ParameterSpace([ContinuousParameter(k, var[0], var[1]) for k, var in target_task.configuration_space.original_ranges.items()])
        

        datasets = get_source_data(task_name, services)
        source_data = {}
        set_id = 0
        for name, data in datasets.items():
            source_data[set_id] = TaskData(
                X=data['X'], Y=data['Y'].reshape(-1, 1)
            )
            set_id += 1

        return obj.f, source_data, space
    
    # Multiple seeds for repetition
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/MTBO_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    for task_name, task_class, input_dim in tqdm.tqdm(tasks, desc="Processing tasks"):
        print(f"\nStarting task: {task_name}")
        task_results = {}
        # Get source data and pretrain GP
    
        for workload in workloads:
            workload_results = []
            for seed in seeds:
                print(f"Running {task_name} with workload {workload}, seed {seed}")
                

                regret, X, Y = run_experiment(num_source_points=[120], benchmark_name=task_name, num_steps=budget, output_noise=0.0, params_source=None, params_target=None)
                

                # Collect results
                print("Collecting final results...")
                history = []
                for i in range(len(X)):
                    history.append({
                        'iteration': i,
                        'params': X[i].tolist(),
                        'loss': float(Y[i])
                    })

                result = {
                    'best_params': X[np.argmax(Y)].tolist(),
                    'best_value': float(np.max(Y)),
                    'history': history
                }

                # Save individual result file
                result_filename = f"{task_name}_workload{workload}_seed{seed}.json"
                result_path = os.path.join(results_dir, result_filename)
                with open(result_path, 'w') as f:
                    json.dump({
                        'task_name': task_name,
                        'workload': workload,
                        'seed': seed,
                        'result': result
                    }, f, indent=2)

                workload_results.append({
                    'seed': seed,
                    'result': result
                })
            task_results[f'workload_{workload}'] = workload_results
        all_results[task_name] = task_results

    print("\nSaving final results...")
    # Save summary results 
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\nFinal Summary:")
    for task_name in all_results:
        print(f"\nResults for {task_name}:")
        for workload in workloads:
            workload_results = all_results[task_name][f'workload_{workload}']
            best_values = [r['result']['best_value'] for r in workload_results]
            print(f"Workload {workload}:")
            print(f"  Mean best value: {np.mean(best_values):.4f}")
            print(f"  Std best value: {np.std(best_values):.4f}")
