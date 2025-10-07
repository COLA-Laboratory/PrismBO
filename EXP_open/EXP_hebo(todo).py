import numpy as np
# print(hasattr(np.dtypes, 'StringDType'))

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from prismbo.benchmark.synthetic.singleobj import *
import numpy as np
import json
import os
from datetime import datetime

def objective(params):
    result = task.objective_function(configuration=params)
    return {'loss': result['f1'], 'status': 'ok'}

def get_design_space():
    original_ranges = task.configuration_space.original_ranges
    space = DesignSpace().parse([
        {'name': param_name, 'type': 'num', 'lb': param_range[0], 'ub': param_range[1]}
        for param_name, param_range in original_ranges.items()
    ])
    return space

def run_experiment(task_name, task_class, input_dim, workload, seed, n_iterations=200, results_dir=None):
    # Create task instance
    global task
    task = task_class(
        task_name=task_name,
        budget_type='FEs',
        budget=220,
        seed=seed,
        workload=workload,
        params={'input_dim': input_dim}
    )
    
    # Create design space
    design_space = get_design_space()
    
    # Initialize HEBO
    opt = HEBO(design_space)
    
    # Run optimization
    for i in range(n_iterations):
        rec = opt.suggest(n_suggestions=1)
        config = rec.to_dict(orient='records')[0]
        result = objective(config)
        y = np.array([[result['loss']]])
        opt.observe(rec, y)
    
    # Collect optimization historypip
    history = []
    for i in range(len(opt.y)):
        history.append({
            'iteration': i,
            'params': opt.X[i].tolist(),
            'loss': float(opt.y[i][0])
        })
    
    result_data = {
        'task_name': task_name,
        'workload': workload,
        'seed': seed,
        'best_params': opt.X[opt.y.argmin()].tolist(),
        'best_value': float(1 - opt.y.min()),
        'history': history
    }
    
    # Save result to JSON file
    if results_dir:
        filename = f"{results_dir}/{task_name}/workload_{workload}/{task_name}_workload_{workload}_seed_{seed}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=2)
    
    return {
        'best_params': opt.X[opt.y.argmin()].tolist(),
        'best_value': float(1 - opt.y.min()),
        'history': history
    }

if __name__ == "__main__":
    # Experiment settings
    tasks = [
        ('Rastrigin', Rastrigin, 10),
        ('Schwefel', Schwefel, 10),
        ('Ackley', Ackley, 10),
        ('Griewank', Griewank, 10),
        ('Rosenbrock', Rosenbrock, 10),
    ]
    workloads = [0, 1, 2, 3, 4, 5]  # Different workloads
    seeds = [0, 1, 2, 3, 4, 5]  # Multiple seeds for repetition
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/hebo_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run experiments
    all_results = {}
    for task_name, task_class, input_dim in tasks:
        task_results = {}
        for workload in workloads:
            workload_results = []
            for seed in seeds:
                print(f"Running {task_name} with workload {workload}, seed {seed}")
                result = run_experiment(task_name, task_class, input_dim, workload, seed, results_dir=results_dir)
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
