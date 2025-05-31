from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prismbo.benchmark.synthetic.singleobj import *
import numpy as np
import json
import os
from datetime import datetime

def objective(params):
    result = task.objective_function(configuration=params)
    return {'loss': result['f1'], 'status': STATUS_OK}

def get_hyperopt_space():
    original_ranges = task.configuration_space.original_ranges
    space = {}
    for param_name, param_range in original_ranges.items():
        space[param_name] = hp.uniform(param_name, param_range[0], param_range[1])
    return space

def run_experiment(task_name, task_class, input_dim, workload, seed, n_iterations=200):
    # Create task instance
    task = task_class(
        task_name=task_name,
        budget_type='FEs',
        budget=220,
        seed=seed,
        workload=workload,
        params={'input_dim': input_dim}
    )
    
    # Create search space
    search_space = get_hyperopt_space()
    
    # Run optimization
    trials = Trials()
    np.random.seed(seed)
    
    best = fmin(fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=n_iterations,
                trials=trials,
                rstate=np.random.default_rng(seed))
    
    # Collect optimization history
    history = []
    for trial in trials.trials:
        history.append({
            'iteration': trial['tid'],
            'params': trial['misc']['vals'],
            'loss': trial['result']['loss']
        })
    
    return {
        'best_params': best,
        'best_value': float(1 - min(trials.losses())),
        'history': history
    }

if __name__ == "__main__":
    # Experiment settings
    tasks = [
        ('Sphere', Sphere, 10),
        ('Rastrigin', Rastrigin, 10),
        ('Schwefel', Schwefel, 10),
        ('Ackley', Ackley, 10),
        ('Griewank', Griewank, 10),
        ('Rosenbrock', Rosenbrock, 10),
    ]
    workloads = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # Different workloads
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # Multiple seeds for repetition
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/hyperopt_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run experiments
    all_results = {}
    for task_name, task_class, input_dim in tasks:
        task_results = {}
        for workload in workloads:
            workload_results = []
            for seed in seeds:
                print(f"Running {task_name} with workload {workload}, seed {seed}")
                
                task = task_class(
                    task_name=task_name,
                    budget_type='FEs',
                    budget=220,
                    seed=seed,
                    workload=workload,
                    params={'input_dim': input_dim}
                )

                # Create search space
                search_space = get_hyperopt_space()
                
                # Run optimization
                trials = Trials()
                np.random.seed(seed)
                
                best = fmin(fn=objective,
                            space=search_space,
                            algo=tpe.suggest,
                            max_evals=220,
                            trials=trials,
                            rstate=np.random.default_rng(seed))
                
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
                    'history': history
                }
                
                
                # result = run_experiment(task_name, task_class, input_dim, workload, seed)
                workload_results.append({
                    'seed': seed,
                    'result': result
                })
            task_results[f'workload_{workload}'] = workload_results
        all_results[task_name] = task_results
    
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
