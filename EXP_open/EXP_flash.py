import pandas as pd
import sys
from os import listdir
from random import shuffle
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import json
import os
import time
from prismbo.benchmark.csstuning.compiler import LLVMTuning, GCCTuning
from prismbo.benchmark.csstuning.dbms import MySQLTuning
from prismbo.benchmark.csstuning.vdbms.vdbms import VDBMSTuning
from prismbo.benchmark.synthetic.singleobj import *
from prismbo.optimizer.initialization.random import RandomSampler

task_class_dict = {
    'Ackley': Ackley,
    'Rastrigin': Rastrigin,
    'Rosenbrock': Rosenbrock,
    # 'XGB': XGBoostBenchmark,
    # 'HPO_PINN': HPO_PINN,
    # 'HPO_ResNet18': HPO_ResNet18,
    # 'HPO_ResNet32': HPO_ResNet32,
    'CSSTuning_GCC': GCCTuning,
    'CSSTuning_LLVM': LLVMTuning,
    'CSSTuning_MySQL': MySQLTuning,
    'CSSTuning_VDBMS': VDBMSTuning,
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


def policy1(scores, lives=3):
    """
    No improvement in last 3 runs
    """
    temp_lives = lives
    last = scores[0]
    for i, score in enumerate(scores):
        if i > 0:
            if temp_lives == 0:
                return i
            elif score >= last:
                temp_lives -= 1
                last = score
            else:
                temp_lives = lives
                last = score
    return -1

def policy2(scores, lives=3):
    """
    No improvement in last 3 runs
    """
    temp_lives = lives
    last = scores[0]
    for i, score in enumerate(scores):
        if i > 0:
            if temp_lives == 0:
                return i
            elif score >= last:
                temp_lives -= 1
                last = score
            else:
                last = score
    return -1

class solution_holder:
    def __init__(self, id, decisions, objective, rank):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.rank = rank


def get_data(filename, initial_size):
    """
    :param filename:
    :param Initial training size
    :return: Training and Testing
    """
    pdcontent = pd.read_csv(filename)
    indepcolumns = [col for col in pdcontent.columns if "$<" not in col]
    depcolumns = [col for col in pdcontent.columns if "$<" in col]
    sortpdcontent = pdcontent.sort_values(by=depcolumns[-1])
    ranks = {}
    for i, item in enumerate(sorted(set(sortpdcontent[depcolumns[-1]].tolist()))):
        ranks[item] = i

    content = []
    for c in range(len(sortpdcontent)):
        content.append(solution_holder(
            c,
            sortpdcontent.iloc[c][indepcolumns].tolist(),
            sortpdcontent.iloc[c][depcolumns].tolist(),
            ranks[sortpdcontent.iloc[c][depcolumns].tolist()[-1]]
        ))

    shuffle(content)
    indexes = list(range(len(content)))
    train_indexes, test_indexes = indexes[:initial_size], indexes[initial_size:]
    assert (len(train_indexes) + len(test_indexes) == len(indexes)), "Something is wrong"
    train_set = [content[i] for i in train_indexes]
    test_set = [content[i] for i in test_indexes]

    return [train_set, test_set]


def get_best_configuration_id(train, test):
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective for t in train]

    test_independent = [t.decision for t in test]

    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test_independent)
    predicted_id = [[t.id, p] for t, p in zip(test, predicted)]
    predicted_sorted = sorted(predicted_id, key=lambda x: x[-1])
    # Find index of the best predicted configuration
    best_index = predicted_sorted[0][0]
    return best_index


def run_active_learning(filename, initial_size, max_lives=10):
    steps = 0
    lives = max_lives
    training_set, testing_set = get_data(filename, initial_size)
    dataset_size = len(training_set) + len(testing_set)
    while (initial_size + steps) < dataset_size - 1:
        best_id = get_best_configuration_id(training_set, testing_set)
        best_solution = [t for t in testing_set if t.id == best_id][-1]

        list_of_all_solutions = [t.objective[-1] for t in training_set]
        if best_solution.objective[-1] < min(list_of_all_solutions):
            lives = max_lives
        else:
            lives -= 1
        training_set.append(best_solution)
        # find index of the best_index
        best_index = [i for i in range(len(testing_set)) if testing_set[i].id == best_id]
        assert (len(best_index) == 1), "Something is wrong"
        best_index = best_index[-1]
        del testing_set[best_index]
        assert (len(training_set) + len(testing_set) == dataset_size), "Something is wrong"
        if lives == 0:
            break
        steps += 1

    return training_set, testing_set


def wrapper_run_active_learning(filename, initial_size):
    training_set, testing_set = run_active_learning(filename, initial_size)
    global_min = min([t.objective[-1] for t in training_set + testing_set])
    best_training_solution = [tt.rank for tt in training_set if min([t.objective[-1] for t in training_set]) == tt.objective[-1]]
    best_solution = [tt.rank for tt in training_set + testing_set if tt.objective[-1] == global_min]
    print(f"{min(best_training_solution) - min(best_solution)}, {len(training_set)} | ", end="")
    return (min(best_training_solution) - min(best_solution)), len(training_set)

if __name__ == "__main__":
     # Experiment settings
    config = read_config()
    seeds = [int(s.strip()) for s in config['seeds'].split(',')]

    # Create results directory
    results_dir = f"results/flash_{config['experimentName']}"
    os.makedirs(results_dir, exist_ok=True)

    # Run experiments
    all_results = {}
    ini_num = 3
    max_lives = 110
    inf = 1e8

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

                time_start = time.time()
                steps = 0
                lives = max_lives
                search_space = task.get_configuration_space()
                sampler = RandomSampler(init_num=ini_num, config = {})
                ini_X = sampler.sample(search_space)
                query_datasets = [search_space.map_to_design_space(sample) for sample in ini_X]
                ini_Y = np.array([task.f(query)['f1'] for query in query_datasets]).reshape(-1, 1)
                
                test_sampler = RandomSampler(init_num=2000, config = {})
                test_X = test_sampler.sample(search_space)
                # test_query_datasets = [search_space.map_to_design_space(sample) for sample in test_X]

                sorted_indices = np.argsort(ini_Y.flatten())
                ranks_array = np.empty_like(sorted_indices)
                ranks_array[sorted_indices] = np.arange(len(ini_Y))

                content = []
                for c, x in enumerate(ini_X):
                    content.append(solution_holder(
                        c,
                        x.tolist(),
                        ini_Y[c],
                        ranks_array[c]
                    ))
                    
                for c, x in enumerate(test_X):
                    content.append(solution_holder(
                        c + ini_num,
                        x.tolist(),
                        inf,
                        inf
                    ))

                indexes = list(range(len(content)))
                train_indexes, test_indexes = indexes[:ini_num], indexes[ini_num:]

                training_set = [content[i] for i in train_indexes]
                testing_set = [content[i] for i in test_indexes]

                
                while (ini_num + steps) < int(task_info['budget']) - 1:
                    best_id = get_best_configuration_id(training_set, testing_set)
                    best_solution = [t for t in testing_set if t.id == best_id][-1]
                    best_solution.objective = np.array([task.f(search_space.map_to_design_space(best_solution.decision))['f1']])
                    

                    list_of_all_solutions = [t.objective for t in training_set]
                    if best_solution.objective < min(list_of_all_solutions):
                        lives = max_lives
                    else:
                        lives -= 1
                    training_set.append(best_solution)
                    # find index of the best_index
                    best_index = [i for i in range(len(testing_set)) if testing_set[i].id == best_id]
                    best_index = best_index[-1]
                    del testing_set[best_index]
                    if lives == 0:
                        break
                    steps += 1
                
                best = training_set[np.argmin([t.objective for t in training_set])].decision
                history = []
                for i, trial in enumerate(training_set):
                    history.append({
                        'iteration': i,
                        'params': trial.decision,
                        'loss': trial.objective.tolist()[0]
                    })
                
                result =  {
                    'best_params': best,
                    'best_value': float(min([t.objective for t in training_set])),
                    'history': history,
                    'optimization_time': time.time() - time_start
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
    
    