import os, sys
import time
import json
import random
import numpy as np
import copy
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import math


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


random.seed(456)
iters = 60
begin2end = 5
md = int(os.environ.get('MODEL', 1))
fnum = int(os.environ.get('FNUM', 3))
decay = float(os.environ.get('DECAY', 0.5))
scale = float(os.environ.get('SCALE', 10))
offset = float(os.environ.get('OFFSET', 20))


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



def generate_opts(independent):
    result = []
    for k, s in enumerate(independent):
        if s == 1:
            result.append(options[k])
    independent = result

    return independent


def generate_conf(x):
    comb = bin(x).replace('0b', '')
    comb = '0' * (len(options) - len(comb)) + comb
    conf = []
    for k, s in enumerate(comb):
        if s == '1':
            conf.append(1)
        else:
            conf.append(0)
    return conf

class get_exchange(object):
    def __init__(self, incumbent):
        self.incumbent = incumbent

    def to_next(self, feature_id):
        ans = [0] * len(options)
        for f in feature_id:
            ans[f] = 1
        for f in self.incumbent:
            ans[f[0]] = f[1] 
        return ans

def do_search(train_indep, model, eta, rnum):
    features = model.feature_importances_
    print('features')
    print(features)
    
    b = time.time()
    feature_sort = [[i, x] for i, x in enumerate(features)]
    feature_selected = sorted(feature_sort, key=lambda x: x[1], reverse=True)[:fnum]
    feature_ids = [x[0] for x in feature_sort]
    neighborhood_iterators = []    
    for i in range(2 ** fnum):
        comb = bin(i).replace('0b', '')
        comb = '0' * (fnum - len(comb)) + comb
        inc = []
        for k, s in enumerate(comb):
            if s == '1':
                inc.append((feature_selected[k][0], 1))
            else:
                inc.append((feature_selected[k][0], 0))
        neighborhood_iterators.append(get_exchange(inc))
    print('time1:' + str(time.time() - b))

    s = time.time()
    neighbors = []
    r = 0
    print('rnum:' + str(rnum))
    for i, inc in enumerate(neighborhood_iterators):
        for j in range(1 + int(rnum)):
            selected_feature_ids = random.sample(feature_ids, random.randint(0, len(feature_ids)))
            n = neighborhood_iterators[i].to_next(selected_feature_ids)
            neighbors.append(n)
    print('neighbrslen:'+str(len(neighbors)))
    print('time2:' + str(time.time()-s))
    
    pred = []
    estimators = model.estimators_
    s = time.time()
    for e in estimators:
        pred.append(e.predict(np.array(neighbors)))
    acq_val_incumbent = get_ei(pred, eta)
    print('time3:' + str(time.time()-s))
   
    return [[i, a] for a, i in zip(acq_val_incumbent, neighbors)]

def get_ei(pred, eta):
    pred = np.array(pred).transpose(1, 0)
    m = np.mean(pred, axis=1)
    s = np.std(pred, axis=1)

    def calculate_f():
        z = (eta - m) / s
        return (eta - m) * norm.cdf(z) + s * norm.pdf(z)
    
    if np.any(s == 0.0):
        s_copy = np.copy(s)
        s[s_copy == 0.0] = 1.0
        f = calculate_f()
        f[s_copy == 0.0] = 0.0
    else:
        f = calculate_f()

    return f

def get_nd_solutions(train_indep, training_dep, eta, rnum):
    predicted_objectives = []
    model = RandomForestRegressor()
    
    model.fit(np.array(train_indep), np.array(training_dep).reshape(-1))
    estimators = model.estimators_

    pred = []
    for e in estimators:
        pred.append(e.predict(train_indep))
    train_ei = get_ei(pred, eta)

    #get_initial_points
    configs_previous_runs = [(x, train_ei[i]) for i, x in enumerate(train_indep)]
    configs_previous_runs_sorted = sorted(configs_previous_runs, key=lambda x: x[1], reverse=True)

    # do search
    begin = time.time()
    merged_predicted_objectives = do_search(train_indep, model, eta, rnum)
    merged_predicted_objectives = sorted(merged_predicted_objectives, key=lambda x: x[1], reverse=True)
    end = time.time()
    print('search time:' + str(begin - end)) 

    begin = time.time()
    for x in merged_predicted_objectives:
        if x[0] not in train_indep:
            print('no repete time:' + str(time.time() - begin))
            return x[0], x[1]

def get_training_sequence(training_indep, training_dep, testing_indep, rnum):
    return_nd_independent, predicted_objectives = get_nd_solutions(training_indep, training_dep, testing_indep, rnum)
    return return_nd_independent, predicted_objectives




if __name__ == "__main__":
    # Experiment settings
    config = read_config()
    seeds = [int(s.strip()) for s in config['seeds'].split(',')]

    # Create results directory
    results_dir = f"results/boca_{config['experimentName']}"
    os.makedirs(results_dir, exist_ok=True)

    # Run experiments
    all_results = {}
    ini_num = 3
    rnum0 = int(os.environ.get('RNUM', 2 ** 8))

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
                b = time.time()

                training_indep = []
                training_dep = []
                ts = []
                result = 1e8
                steps = 0
                search_space = task.get_configuration_space()
                sampler = RandomSampler(init_num=ini_num, config = {})
                ini_X = sampler.sample(search_space)
                query_datasets = [search_space.map_to_design_space(sample) for sample in ini_X]
                ini_Y = np.array([task.f(query)['f1'] for query in query_datasets]).reshape(-1, 1)
                sigma = -scale ** 2 / (2 * math.log(decay))
                options = search_space.variables_order
                
                training_indep = [i for i in ini_X]
                training_dep = [i for i in ini_Y]

                while ini_num + steps < int(task_info['budget']):
                    steps += 1
                    rnum = rnum0 * math.exp(-max(0, len(ini_X) - offset) ** 2 / (2 * sigma ** 2))
                    best_solution, return_nd_independent = get_training_sequence(ini_X, ini_Y, result, rnum)
                    print('best_solution')
                    print(best_solution)
                    training_indep.append(np.array(best_solution))
                    ts.append(time.time() - b)
                    query = search_space.map_to_design_space(np.array(best_solution))
                    best_result = task.f(query)['f1']
                    training_dep.append(np.array(best_result))

                    if best_result < result:
                        result = best_result
                        best = best_solution
                
                history = []
                for i, trial in enumerate(training_indep):
                    history.append({
                        'iteration': i,
                        'params': trial.tolist(),
                        'loss': training_dep[i].tolist()
                    })
                
                result =  {
                    'best_params': best,
                    'best_value': float(min(training_dep)),
                    'history': history,
                    'optimization_time': sum(ts)
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