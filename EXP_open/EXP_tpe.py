import ConfigSpace as cs
import time
import numpy as np
from typing import Any, Dict, List, Optional, Protocol, Tuple

from tpe.optimizer import TPEOptimizer

from tpe.optimizer.base_optimizer import BaseOptimizer, ObjectiveFunc

from prismbo.benchmark.hpo import * 
from prismbo.benchmark.csstuning.compiler import LLVMTuning, GCCTuning
from prismbo.benchmark.csstuning.dbms import MySQLTuning
from prismbo.benchmark.synthetic.singleobj import *

import os
import json


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

# Define the configuration space
def get_configspace(task):
    hyperparameters = [cs.UniformFloatHyperparameter(param_name, lower=task.configuration_space.original_ranges[param_name][0], upper=task.configuration_space.original_ranges[param_name][1]) for param_name in task.configuration_space.variables_order]
    space = cs.ConfigurationSpace(hyperparameters)
    
    return space


class formal_obj(ObjectiveFunc):
    def __init__(self, f, task):
        self.task = task
        self.f = f
    
    def __call__(self, eval_config: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
        configuration = [eval_config[v] for v in self.task.configuration_space.variables_order]
        configuration = self.task.configuration_space.map_to_design_space(configuration)
        start = time.time()
        results = self.f(configuration)
        return {'loss': results['f1']}, time.time() - start




if __name__ == "__main__":
    # Experiment settings
    config = read_config()
    seeds = [int(s.strip()) for s in config['seeds'].split(',')]  # Multiple seeds for repetition

    # Create results directory
    results_dir = f"results/tpe_{config['experimentName']}"
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

                obj_fun = formal_obj(task.f, task)
                config_space = get_configspace(task)
                # Initialize TPE Optimizer
                opt = TPEOptimizer(obj_func=obj_fun, config_space=config_space, n_init=10, max_evals=int(task_info['budget']), resultfile='tpe_results')

                # Run optimization
                start_time = time.time()
                best_config, best_value = opt.optimize()
                end_time = time.time()
                observations = opt.fetch_observations()

                orders = task.configuration_space.variables_order

                X = np.zeros((len(observations[orders[0]]), len(orders)))
                Y = np.zeros((len(observations[orders[0]]), 1))
                for i, k in enumerate(orders):
                    X[:, i] = observations[k]
                Y[:, 0] = observations['loss']
                # 找到最小值best及其对应的bestX和bestY
                best_idx = np.argmin(Y[:, 0])
                bestX = X[best_idx, :]
                bestY = Y[best_idx, 0]

                # Convert numpy arrays to lists for JSON serialization
                bestX_list = bestX.tolist() if isinstance(bestX, np.ndarray) else bestX
                bestY_item = float(bestY) if isinstance(bestY, (np.floating, np.float32, np.float64, np.ndarray)) else bestY
                
                history = []
                for i in range(X.shape[0]):
                    history.append({
                        'iteration': i,
                        'params': X[i].tolist(),
                        'loss': float(Y[i])
                    })

                result = {
                    'seed': seed,
                    'result': {
                        'best_params': bestX_list,
                        'best_value': bestY_item
                    },
                    'optimization_time': end_time - start_time,
                    'history': history,
                }

                # 记录单次
                current_result = {
                    'task_name': task_info['name'],
                    'workload': workload,
                    'seed': seed,
                    'result': result,
                }

                single_result_file = (
                    f"{results_dir}/{task_info['name']}/workload_{workload}/"
                    f"{task_info['name']}_workload_{workload}_seed_{seed}.json"
                )
                os.makedirs(os.path.dirname(single_result_file), exist_ok=True)
                with open(single_result_file, 'w') as f:
                    json.dump(current_result, f, indent=2)
                print(f"Saved result to {single_result_file}")

                workload_results.append(result)

            task_results[f'workload_{workload}'] = workload_results

        all_results[task_info['name']] = task_results

        # 每完成一个任务就保存累积的结果
        with open(f"{results_dir}/results_{task_info['name']}.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved cumulative results for {task_info['name']}")

    with open(f"{results_dir}/results_complete.json", 'w') as f:
        json.dump(all_results, f, indent=2)

