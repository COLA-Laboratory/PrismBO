# Copyright 2023 HyperBO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import time
import numpy as np
import os
import json
from datetime import datetime

import jaxopt
import tqdm

from external.hyperbo.basics import definitions as defs
from external.hyperbo.basics import params_utils
from external.hyperbo.gp_utils import gp
from external.hyperbo.gp_utils import kernel
from external.hyperbo.gp_utils import mean
from external.hyperbo.gp_utils import utils
from external.hyperbo.bo_utils import acfun


from absl.testing import absltest
from absl.testing import parameterized

from external.hyperbo.bo_utils import bayesopt
from external.hyperbo.bo_utils import const
from external.hyperbo.bo_utils import data
from external.hyperbo.gp_utils import mean

from prismbo.optimizer.initialization.random import RandomSampler

from typing import Callable, Optional, Sequence, Tuple, Union, Any



import jax
import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt


from prismbo.agent.services import Services
from prismbo.analysis.analysisbase import AnalysisBase
from external.hyperbo.bo_utils import bayesopt

from prismbo.benchmark.synthetic.singleobj import *




DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
SubDataset = defs.SubDataset



def get_source_data(task_name):
    task_source = {'Sphere': 'Sphere_source', 'Rastrigin': 'Rastrigin_source', 'Schwefel': 'Schwefel_source', 'Ackley': 'Ackley_source', 'Griewank': 'Griewank_source', 'Rosenbrock': 'Rosenbrock_source'}
    source_name = task_source[task_name]

    datasets = {}
    with open('Results/datasets.txt', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith("Experiment:"):
                experiment_name = lines[i].strip().split(":")[1].strip()
                dataset_list = []
                i += 1
                while i < len(lines) and not lines[i].startswith("-----"):
                    dataset_list.append(lines[i].strip())
                    i += 1
                datasets[experiment_name] = dataset_list
    Exper_folder = 'Results'
    ab = AnalysisBase(Exper_folder, datasets, services.data_manager)
    ab.read_data_from_db()
    
    train_data = {}
    datasets = ab._all_data[source_name]
    metadata_info = ab._data_infos[source_name]
    sub_dataset_id = 0
    for dataset_name, data in datasets.items():
        objectives = metadata_info[dataset_name]["objectives"]
        obj = objectives[0]["name"]

        obj_data = [d[obj] for d in data]
        var_data = [[d[var["name"]] for var in metadata_info[dataset_name]["variables"]] for d in data]
        input_size = metadata_info[dataset_name]['num_variables']
        X = jax.numpy.array(var_data)
        Y = jax.numpy.array(obj_data)[:, np.newaxis]
        train_data[str(sub_dataset_id)] = SubDataset(X, Y)
        sub_dataset_id += 1
    return train_data
        
def pretrain_GP(train_data, optimization_method, loss_function, max_training_step):
    #pretrain GP
    DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
    GPParams = defs.GPParams
    SubDataset = defs.SubDataset
    key = jax.random.PRNGKey(1)
    params = GPParams(
        model={
            'lengthscale': jnp.array([.0]),
            'signal_variance': 0.0,
            'noise_variance': -6.,
        },
        config={
            'mlp_features': (8, 8),
            'method': optimization_method,
            'max_training_step': max_training_step,
            'batch_size': 100,
            'objective': loss_function if loss_function == 'nll' else 'kl',
            'learning_rate': 1e-3,
        },
    )
    mean_func = mean.linear_mlp
    cov_func = kernel.squared_exponential_mlp
    warp_func = DEFAULT_WARP_FUNC


    model = gp.GP(
        dataset=train_data,
        params=params,
        mean_func=mean_func,
        cov_func=cov_func,
        warp_func=warp_func,
    )

    key, subkey = jax.random.split(key, 2)
    model.initialize_params(subkey)
    
    _ = model.stats()
    start = time.time()
    print('Pre-training..')
    trained_params = model.train()
    print(f'Pre-training time (s): {time.time() - start}')
    print('After pre-training.')
    _ = model.stats()
    return model
    
    

def retrain_model(model: gp.GP,
                  sub_dataset_key: Union[int, str],
                  random_key: Optional[jax.Array] = None,
                  get_params_path: Optional[Callable[[Any], Any]] = None,
                  callback: Optional[Callable[[Any], Any]] = None):
  """Retrain the model with more observations on sub_dataset.

  Args:
    model: gp.GP.
    sub_dataset_key: key of the sub_dataset for testing in dataset.
    random_key: random state for jax.random, to be used for training.
    get_params_path: optional function handle that returns params path.
    callback: optional callback function for loggin of training steps.
  """
  retrain_condition = 'retrain' in model.params.config and model.params.config[
      'retrain'] > 0 and model.dataset[sub_dataset_key].x.shape[0] > 0
  if not retrain_condition:
    return
  if model.params.config['objective'] in [obj.regkl, obj.regeuc]:
    raise ValueError('Objective must include NLL to retrain.')
  max_training_step = model.params.config['retrain']
  logging.info(
      msg=('Retraining with max_training_step = '
           f'{max_training_step}.'))
  model.params.config['max_training_step'] = max_training_step
  model.train(
      random_key, get_params_path=get_params_path, callback=callback)

if __name__ == '__main__':    
    tasks = [
        # ('Rastrigin', Rastrigin, 10),
        ('Schwefel', Schwefel, 10),
        ('Ackley', Ackley, 10),
        ('Griewank', Griewank, 10),
        ('Rosenbrock', Rosenbrock, 10),
    ]
    budget = 220
    # @title Initialize a GP model to be pre-trained
    optimization_method = 'lbfgs'  # @param ['lbfgs', 'adam']
    loss_function = 'ekl'  # @param ['nll', 'ekl']
    max_training_step = 1000  #@param{type: "number", isTemplate: true}
    
    services = Services(None, None, None)
    services._initialize_modules()
    
    configurations = services.configer.get_configuration()

    workloads = [0, 1, 2, 3, 4, 5]  # Different workloads
    seeds = [0, 1, 2, 3, 4, 5]  # Multiple seeds for repetition
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/HyperBO_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}
    for task_name, task_class, input_dim in tqdm.tqdm(tasks, desc="Processing tasks"):
        print(f"\nStarting task: {task_name}")
        task_results = {}
        # Get source data and pretrain GP
        train_data = get_source_data(task_name)
        model = pretrain_GP(train_data, optimization_method, loss_function, max_training_step)
        
        for workload in workloads:
            workload_results = []
            for seed in seeds:
                print(f"Running {task_name} with workload {workload}, seed {seed}")
                key = jax.random.PRNGKey(seed)
                
                task = task_class(
                    task_name=task_name,
                    budget_type='FEs',
                    budget=budget,
                    seed=seed,
                    workload=workload,
                    params={'input_dim': input_dim}
                )
                
                #get serchspace of the task
                original_ranges = task.configuration_space.original_ranges


                # Initialize dataset for HyperBO
                search_space = task.configuration_space
                initial_num = model.input_dim * 11
                sampler = RandomSampler(config = {})
                initial_samples = sampler.sample(search_space, n_points=initial_num)
                query_datasets = [search_space.map_to_design_space(sample) for sample in initial_samples]

                print("Collecting initial observations...")
                observations = [task.objective_function(data) for data in tqdm.tqdm(query_datasets, desc="Initial sampling")]
                Y = np.array([obs['f1'] for obs in observations]).reshape(-1, 1)
                
                
                sub_dataset_key = str(len(model.dataset))
                queried_sub_dataset = SubDataset(
                    x=jnp.array(initial_samples),
                    y=jnp.array(Y)
                )
                model.update_sub_dataset(
                    queried_sub_dataset, sub_dataset_key=sub_dataset_key, is_append=False)

                acf = acfun.ei

                input_dim = model.input_dim
                for i in tqdm.tqdm(range(budget-initial_num), desc="Optimization iterations"):
                    start_time = time.time()
                    retrain_model(model, sub_dataset_key=sub_dataset_key)
                    key, subkey = jax.random.split(key)
                    x_samples = sampler.sample(search_space, n_points=1000)

                    evals = acf(
                        model=model, sub_dataset_key=sub_dataset_key, x_queries=x_samples)
                    select_idx = evals.argmax()
                    x_init = x_samples[select_idx]
                    
                    def f(x):
                        return -acf(
                            model=model,
                            sub_dataset_key=sub_dataset_key,
                            x_queries=jnp.array([x])).flatten()[0]

                    
                    opt = jaxopt.ScipyBoundedMinimize(method='L-BFGS-B', fun=f)
                    
                    opt_ret = opt.run(
                        x_init, bounds=[jnp.array([original_ranges[k][0] for k,v in original_ranges.items()]),
                                      jnp.array([original_ranges[k][1] for k,v in original_ranges.items()])])
                    task_params = search_space.map_to_design_space(np.array(opt_ret.params))
                    obs = task.objective_function(task_params)
                    eval_datapoint = opt_ret.params, jax.numpy.array([obs['f1']]).reshape(-1, 1)
                    logging.info(msg=f'{i}-th iter, x_init={x_init}, '
                                f'eval_datapoint={eval_datapoint}, '
                                f'elpased_time={time.time() - start_time}')
                    model.update_sub_dataset(
                        eval_datapoint, sub_dataset_key=sub_dataset_key, is_append=True)

                observations = model.dataset.get(sub_dataset_key)
                # Collect results
                print("Collecting final results...")
                history = []
                for i, (x, y) in enumerate(zip(observations.x, observations.y)):
                    history.append({
                        'iteration': i,
                        'params': x.tolist(),
                        'loss': float(y[0])
                    })

                result = {
                    'best_params': observations.x[np.argmin(observations.y)].tolist(),
                    'best_value': float(np.min(observations.y)),
                    'history': history
                }

                # Save individual result file
                result_filename = f"{task_name}_workload{workload}_seed{seed}.json"
                result_path = os.path.join(results_dir, result_filename)
                with open(result_path, 'w') as f:
                    json.dump({
                        'task_name': task_name,
                        'workload': workload,
                        'seed': seed,
                        'result': result
                    }, f, indent=2)

                workload_results.append({
                    'seed': seed,
                    'result': result
                })
            task_results[f'workload_{workload}'] = workload_results
        all_results[task_name] = task_results

    print("\nSaving final results...")
    # Save summary results 
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\nFinal Summary:")
    for task_name in all_results:
        print(f"\nResults for {task_name}:")
        for workload in workloads:
            workload_results = all_results[task_name][f'workload_{workload}']
            best_values = [r['result']['best_value'] for r in workload_results]
            print(f"Workload {workload}:")
            print(f"  Mean best value: {np.mean(best_values):.4f}")
            print(f"  Std best value: {np.std(best_values):.4f}")
    

    
    

    