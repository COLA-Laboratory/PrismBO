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
import json
from emukit.core import ParameterSpace
from GPy.kern import RBF
import torch
from external.transfergpbo.models import (
    TaskData,
    WrapperBase,
    MHGP,
    SHGP,
    BHGP,
    RGPE,
)
from external.transfergpbo.bo.run_bo import run_bo
from external.transfergpbo import models, benchmarks
from external.transfergpbo.parameters import parameters as params

from emukit.core import ParameterSpace, ContinuousParameter

from prismbo.benchmark.hpo import * 
from prismbo.benchmark.csstuning.compiler import LLVMTuning, GCCTuning
from prismbo.benchmark.csstuning.dbms import MySQLTuning
from prismbo.benchmark.synthetic.singleobj import *

from prismbo.agent.services import Services
from prismbo.analysis.analysisbase import AnalysisBase
import time

from datetime import datetime
import os
# suppress GPyTorch warnings about adding jitter
import warnings
warnings.filterwarnings("ignore", "^.*jitter.*", category=RuntimeWarning)

task_class_dict = {
    'Ackley': Ackley,
    'Rastrigin': Rastrigin,
    'Rosenbrock': Rosenbrock,
    'XGB': XGBoostBenchmark,
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


class objective_function():
    def __init__(self, search_space, task) -> None:
        self.search_space = search_space
        self.task = task
    
    
    def f(self, X, **kwargs):
        Y = []
        query_datasets = [self.search_space.map_to_design_space(sample) for sample in X]
        Y = np.array([[self.task.objective_function(data)['f1'] for data in query_datasets]])
        return Y



if __name__ == "__main__":
    # Experiment settings
    config = read_config()
    seeds = [int(s.strip()) for s in config['seeds'].split(',')]  # Multiple seeds for repetition

    # Create results directory
    results_dir = f"results/rgpe_{config['experimentName']}"
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}
    source_data = {}
    set_id = 0
    for task_info in config['tasks']:
        task_results = {}
        task_class = task_class_dict[task_info['name']]
        workloads = [int(w.strip()) for w in task_info['workloads'].split(',')]
        for workload in workloads:
            workload_results = []
            for seed in seeds:
                print(f"Running {task_info['name']} with workload {workload}, seed {seed}")

                task = task_class(
                    task_name=task_info['name'],
                    budget_type=task_info['budget_type'],
                    budget=task_info['budget'],
                    seed=seed,
                    workload=workload,
                    description=task_info['description'],
                    params={'input_dim': int(task_info['num_vars'])}
                )

                # Create search space
                space = ParameterSpace([ContinuousParameter(k, task.configuration_space.original_ranges[k][0], task.configuration_space.original_ranges[k][1]) for k in task.configuration_space.variables_order])

                # Build model
                model = get_model('RGPE', space)
                obj = objective_function(search_space=task.configuration_space, task=task)

                if len(source_data) > 0:
                    start_time = time.time()
                    model.meta_fit(source_data)

                    # Run BO and return the regret, X, Y
                    regret, X, Y = run_bo(
                        experiment_fun=partial(obj.f, output_noise=0.1),
                        model=model,
                        space=space,
                        num_iter=int(task_info['budget']),
                    )
                    end_time = time.time()
                    optimization_time = end_time - start_time
                else:
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

                # Collect optimization history
                history = []
                for i in range(X.shape[0]):
                    history.append({
                        'iteration': i,
                        'params': X[i].tolist(),
                        'loss': float(Y[i])
                    })

                result = {
                    'best_params': X[np.argmin(Y)].tolist(),
                    'best_value': float(np.min(Y)),
                    'history': history,
                    'optimization_time': optimization_time  # You can add timing if needed
                }

                # Save result immediately after each task completion
                task_dir = os.path.join(results_dir, task_info['name'])
                os.makedirs(task_dir, exist_ok=True)
                workload_dir = os.path.join(task_dir, f"workload_{workload}")
                os.makedirs(workload_dir, exist_ok=True)

                filename = f"{task_info['name']}_workload_{workload}_seed_{seed}.json"
                filepath = os.path.join(workload_dir, filename)

                with open(filepath, 'w') as f:
                    json.dump({
                        'task_name': task_info['name'],
                        'workload': workload,
                        'seed': seed,
                        'result': result
                    }, f, indent=2)

                print(f"Saved result to {filepath}")

                workload_results.append({
                    'seed': seed,
                    'result': result,
                })
                source_data = {1:TaskData(X, Y)}
            
                set_id += 1
            task_results[f'workload_{workload}'] = workload_results
        all_results[task_info['name']] = task_results

    # Save all results
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    for task_name in all_results:
        print(f"\nResults for {task_name}:")
        for workload in workloads:
            workload_results = all_results[task_name][f'workload_{workload}']
            best_values = [r['result']['best_value'] for r in workload_results]
            print(f"Workload {workload}:")
            print(f"  Mean best value: {np.mean(best_values):.4f}")
            print(f"  Std best value: {np.std(best_values):.4f}")

    
    
    # def get_benchmark(
    #     benchmark_name: str,
    #     num_source_points: List[int],
    #     seed: int,
    #     workload: int,
    #     output_noise: float = 0.0,
    #     params_source: List[Dict[str, float]] = None,
    #     params_target: Dict[str, float] = None,
    # ) -> Tuple[Callable, Dict[Hashable, TaskData], ParameterSpace]:
    #     """Create the benchmark object."""
    #     num_source_functions = len(num_source_points)

    #     target_task = task_class(
    #             task_name=benchmark_name,
    #             budget_type='FEs',
    #             budget=budget,
    #             seed=seed,
    #             workload=workload,
    #             params={'input_dim': input_dim}
    #         )

    #     obj = objective_function(search_space=target_task.configuration_space, task=target_task)

    #     # Get parameter space
    #     space = ParameterSpace([ContinuousParameter(k, var[0], var[1]) for k, var in target_task.configuration_space.original_ranges.items()])
        

    #     datasets = get_source_data(task_name, services)
    #     source_data = {}
    #     set_id = 0
    #     for name, data in datasets.items():
    #         source_data[set_id] = TaskData(
    #             X=data['X'], Y=data['Y'].reshape(-1, 1)
    #         )
    #         set_id += 1

    #     return obj.f, source_data, space
    
    # Multiple seeds for repetition
    # Create results directory
    