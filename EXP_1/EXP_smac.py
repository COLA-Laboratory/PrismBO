from ConfigSpace import ConfigurationSpace
import ConfigSpace as cs
import numpy as np
import json
import os
from datetime import datetime
from prismbo.benchmark.synthetic.singleobj import *


from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario

from prismbo.benchmark.hpo import * 
from prismbo.benchmark.csstuning.compiler import LLVMTuning, GCCTuning
from prismbo.benchmark.csstuning.dbms import MySQLTuning

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




def objective(configuration, seed: int = 0):
    # config_list = np.array([configuration.get(name) for name in task.configuration_space.variables_order])
    result = task.objective_function(configuration=configuration)
    return 1 - result['f1']  # SMAC minimizes, so we return 1 - accuracy

def get_configspace():
    space = ConfigurationSpace()
    original_ranges = task.configuration_space.original_ranges
    for param_name, param_range in original_ranges.items():
        space.add_hyperparameter(cs.UniformFloatHyperparameter(param_name, lower=param_range[0], upper=param_range[1]))
    return space


if __name__ == "__main__":
    # Experiment settings
    config =read_config()
    seeds = [int(s.strip()) for s in config['seeds'].split(',')]  # Multiple seeds for repetition
    
    # Create results directory
    results_dir = f"results/hyperopt_{config['experimentName']}"
    os.makedirs(results_dir, exist_ok=True)
    
    
    
    # Run experiments
    all_results = {}
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
                
                config_space = get_configspace()
                # Provide meta data for the optimization
                scenario = Scenario({
                    "run_obj": "quality",  # Optimize quality (alternatively runtime)
                    "runcount-limit": task_info['budget'],  # Max number of function evaluations (the more the better)
                    "cs": config_space,
                })
                smac = SMAC4BB(scenario=scenario, tae_runner=objective)
                incumbent = smac.optimize()
                
                result={
                    'seed': seed,
                    'result':{
                    'best_params': {name: incumbent[name] for name in incumbent},
                    'best_value': objective({name: incumbent[name] for name in incumbent})}
                }
                
                current_result = {
                    'task_name': task_info['name'],
                    'workload': workload,
                    'seed': seed,
                    'result': result,
                }
                
                single_result_file = f"{results_dir}/{task_info['name']}/workload_{workload}/{task_info['name']}_workload_{workload}_seed_{seed}.json"
                os.makedirs(os.path.dirname(single_result_file), exist_ok=True)
                with open(single_result_file, 'w') as f:
                    json.dump(current_result, f, indent=2)
                
                print(f"Saved result to {single_result_file}")
                
            task_results[f'workload_{workload}'] = workload_results
        all_results[task_info['name']] = task_results
        
        # 每完成一个任务就保存累积的结果
        with open(f"{results_dir}/results_{task_info['name']}.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved cumulative results for {task_info['name']}")
    
    # 保存最终完整结果
    with open(f"{results_dir}/results_complete.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    for task_info['name'] in all_results:
        print(f"\nResults for {task_info['name']}:")
        for workload in workloads:
            workload_results = all_results[task_info['name']][f'workload_{workload}']
            best_values = [r['result']['best_value'] for r in workload_results]
            print(f"Workload {workload}:")
            print(f"  Mean best value: {np.mean(best_values):.4f}")
            print(f"  Std best value: {np.std(best_values):.4f}")
