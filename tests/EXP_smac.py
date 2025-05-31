from ConfigSpace import ConfigurationSpace
import ConfigSpace as cs
import numpy as np
import json
import os
from datetime import datetime
from smac import HyperparameterOptimizationFacade, Scenario
from prismbo.benchmark.synthetic.singleobj import *

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
    scenario = Scenario(config_space, deterministic=True, n_trials=n_iterations)
    smac = HyperparameterOptimizationFacade(scenario, objective)
    incumbent = smac.optimize()
    
    return {
        'best_params': {name: incumbent[name] for name in incumbent},
        'best_value': objective({name: incumbent[name] for name in incumbent})
    }

if __name__ == "__main__":
    tasks = [
        ('Sphere', Sphere, 10),
        ('Rastrigin', Rastrigin, 10),
        ('Schwefel', Schwefel, 10),
        ('Ackley', Ackley, 10),
        ('Griewank', Griewank, 10),
        ('Rosenbrock', Rosenbrock, 10),
    ]
    workloads = [0]  # [0-19]
    seeds = [0]  # [0-19]
    
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
            task_results[f'workload_{workload}'] = workload_results
        all_results[task_name] = task_results
    
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    for task_name in all_results:
        print(f"\nResults for {task_name}:")
        for workload in workloads:
            workload_results = all_results[task_name][f'workload_{workload}']
            best_values = [r['result']['best_value'] for r in workload_results]
            print(f"Workload {workload}:")
            print(f"  Mean best value: {np.mean(best_values):.4f}")
            print(f"  Std best value: {np.std(best_values):.4f}")
