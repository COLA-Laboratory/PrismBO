# -*- coding: utf-8 -*-
from ConfigSpace import ConfigurationSpace
import ConfigSpace as cs
import numpy as np
import json
import os
from datetime import datetime
import time
from prismbo.benchmark.synthetic.singleobj import *
from prismbo.benchmark.hpo import *
from prismbo.benchmark.csstuning.compiler import LLVMTuning, GCCTuning
from prismbo.benchmark.csstuning.dbms import MySQLTuning

from smac import HyperparameterOptimizationFacade, Scenario

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


def make_objective(task):
    def objective(config, seed: int = 0):
        cfg = config.get_dictionary() if hasattr(config, "get_dictionary") else dict(config)
        configuration = [cfg[v] for v in task.configuration_space.variables_order]
        configuration = task.configuration_space.map_to_design_space(configuration)
        result = task.objective_function(configuration=configuration)
        return result['f1']
    return objective


def get_configspace(task):
    space = ConfigurationSpace()
    original_ranges = task.configuration_space.original_ranges
    for param_name, (lo, hi) in original_ranges.items():
        space.add_hyperparameter(
            cs.UniformFloatHyperparameter(param_name, lower=lo, upper=hi)
        )
    return space


if __name__ == "__main__":
    # Experiment settings
    config = read_config()
    seeds = [int(s.strip()) for s in config['seeds'].split(',')]

    # Create results directory
    results_dir = f"results/smac_{config['experimentName']}"
    os.makedirs(results_dir, exist_ok=True)

    # Run experiments
    all_results = {}

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

                config_space = get_configspace(task)

                run_name = f"{config['experimentName']}_{task_name}_wl{workload}_seed{seed}"
                scenario = Scenario(
                    config_space,
                    name=run_name,
                    deterministic=True,             # 与旧接口 run_obj=quality 对应的常见设置
                    n_trials=int(task_info['budget']),
                    output_directory=os.path.join(results_dir, "smac3_output"),
                    seed=seed
                )

                objective_fn = make_objective(task)

                smac = HyperparameterOptimizationFacade(scenario, objective_fn)
                start_time = time.time()
                incumbent = smac.optimize()
                end_time = time.time()

                incumbent_dict = incumbent.get_dictionary() if hasattr(incumbent, "get_dictionary") else dict(incumbent)
                best_value = objective_fn(incumbent, seed=seed)

                result = {
                    'seed': seed,
                    'result': { 
                        'best_params': incumbent_dict,
                        'best_value': best_value
                    },
                    'optimization_time': end_time - start_time,
                }

                # 记录单次
                current_result = {
                    'task_name': task_name,
                    'workload': workload,
                    'seed': seed,
                    'result': result,
                }

                single_result_file = (
                    f"{results_dir}/{task_name}/workload_{workload}/"
                    f"{task_name}_workload_{workload}_seed_{seed}.json"
                )
                os.makedirs(os.path.dirname(single_result_file), exist_ok=True)
                with open(single_result_file, 'w') as f:
                    json.dump(current_result, f, indent=2)
                print(f"Saved result to {single_result_file}")

                workload_results.append(result)

            task_results[f'workload_{workload}'] = workload_results

        all_results[task_name] = task_results

        # 每完成一个任务就保存累积的结果
        with open(f"{results_dir}/results_{task_name}.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved cumulative results for {task_name}")

    with open(f"{results_dir}/results_complete.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    for task_name, task_results in all_results.items():
        print(f"\nResults for {task_name}:")
        for workload_key, workload_results in task_results.items():
            best_values = [r['result']['best_value'] for r in workload_results] if workload_results else []
            print(f"  {workload_key}:")
            if best_values:
                print(f"    Mean best value: {np.mean(best_values):.4f}")
                print(f"    Std  best value: {np.std(best_values):.4f}")
            else:
                print("    (no runs)")