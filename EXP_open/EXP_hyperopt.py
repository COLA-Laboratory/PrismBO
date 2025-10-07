from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import json
import os
from datetime import datetime
import time
from prismbo.benchmark.hpo import * 
from prismbo.benchmark.csstuning.compiler import LLVMTuning, GCCTuning
from prismbo.benchmark.csstuning.dbms import MySQLTuning
from prismbo.benchmark.synthetic.singleobj import *

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
    


def objective(params):
    configuration = [params[v] for v in task.configuration_space.variables_order]
    configuration = task.configuration_space.map_to_design_space(configuration)
    result = task.objective_function(configuration=configuration)
    return {'loss': result['f1'], 'status': STATUS_OK}

def get_hyperopt_space(task):
    original_ranges = task.configuration_space.original_ranges
    variables_orders = task.configuration_space.variables_order
    space = {}
    for param_name in variables_orders:
        space[param_name] = hp.uniform(param_name, original_ranges[param_name][0], original_ranges[param_name][1])
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

                # Create search space
                search_space = get_hyperopt_space(task)
                
                # Run optimization
                trials = Trials()
                np.random.seed(seed)
                
                start_time = time.time()
                best = fmin(fn=objective,
                            space=search_space,
                            algo=tpe.suggest,
                            max_evals=int(task_info['budget']),
                            trials=trials,
                            rstate=np.random.default_rng(seed))
                end_time = time.time()
                optimization_time = end_time - start_time

                # Collect optimization history
                history = []
                for trial in trials.trials:
                    history.append({
                        'iteration': trial['tid'],
                        'params': trial['misc']['vals'],
                        'loss': trial['result']['loss']
                    })
                
                result =  {
                    'best_params': best,
                    'best_value': float(1 - min(trials.losses())),
                    'history': history,
                    'optimization_time': optimization_time
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
                    'result': result
                })
            task_results[f'workload_{workload}'] = workload_results
        all_results[task_info['name']] = task_results
    
    # Save results
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
