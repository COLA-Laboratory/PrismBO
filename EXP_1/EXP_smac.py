from ConfigSpace import ConfigurationSpace
import ConfigSpace as cs
import numpy as np
import json
import os
from datetime import datetime
from prismbo.benchmark.synthetic.singleobj import *


from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario


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

def run_experiment(task_name, task_class, input_dim, workload, seed, n_iterations=220):
    global task
    task = task_class(
        task_name=task_name,
        budget_type='FEs',
        budget=220,
        seed=seed,
        workload=workload,
        params={'input_dim': input_dim}
    )
    
    config_space = get_configspace()
        # Provide meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": n_iterations,  # Max number of function evaluations (the more the better)
        "cs": config_space,
    })
    smac = SMAC4BB(scenario=scenario, tae_runner=objective)
    incumbent = smac.optimize()
    
    return {
        'best_params': {name: incumbent[name] for name in incumbent},
        'best_value': objective({name: incumbent[name] for name in incumbent})
    }

if __name__ == "__main__":
    tasks = [
        ('Rastrigin', Rastrigin, 10),
        ('Schwefel', Schwefel, 10),
        ('Ackley', Ackley, 10),
        ('Griewank', Griewank, 10),
        ('Rosenbrock', Rosenbrock, 10),
    ]
    workloads = [0,1,2,3,4,5]  # [0-19]
    seeds = [0,1,2,3,4,5]  # [0-19]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/smac_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    for task_name, task_class, input_dim in tasks:
        task_results = {}
        for workload in workloads:
            workload_results = []
            for seed in seeds:
                print(f"Running {task_name} with workload {workload}, seed {seed}")
                result = run_experiment(task_name, task_class, input_dim, workload, seed)
                workload_results.append({
                    'seed': seed,
                    'result': result
                })
                
                # 立即保存当前任务的结果
                current_result = {
                    'task_name': task_name,
                    'workload': workload,
                    'seed': seed,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
                
                # 保存单个任务结果
                single_result_file = f"{results_dir}/{task_name}/workload_{workload}/{task_name}_workload_{workload}_seed_{seed}.json"
                os.makedirs(os.path.dirname(single_result_file), exist_ok=True)
                with open(single_result_file, 'w') as f:
                    json.dump(current_result, f, indent=2)
                
                print(f"Saved result to {single_result_file}")
                
            task_results[f'workload_{workload}'] = workload_results
        all_results[task_name] = task_results
        
        # 每完成一个任务就保存累积的结果
        with open(f"{results_dir}/results_{task_name}.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved cumulative results for {task_name}")
    
    # 保存最终完整结果
    with open(f"{results_dir}/results_complete.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    for task_name in all_results:
        print(f"\nResults for {task_name}:")
        for workload in workloads:
            workload_results = all_results[task_name][f'workload_{workload}']
            best_values = [r['result']['best_value'] for r in workload_results]
            print(f"Workload {workload}:")
            print(f"  Mean best value: {np.mean(best_values):.4f}")
            print(f"  Std best value: {np.std(best_values):.4f}")
