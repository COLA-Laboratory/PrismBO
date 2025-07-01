import argparse
import copy
import json
import os


from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
import numpy as np
import pandas as pd
import tqdm
from smac.optimizer.acquisition import EI
from smac.optimizer.ei_optimization import FixedSet
from smac.scenario.scenario import Scenario
from smac.facade.smac_bo_facade import SMAC4BO
from smac.initial_design.latin_hypercube_design import LHDesign
import ConfigSpace as cs

from prismbo.agent.services import Services
from prismbo.analysis.analysisbase import AnalysisBase
from prismbo.benchmark.synthetic.singleobj import *
from typing import List, Tuple, Callable, Dict, Hashable
from datetime import datetime


def run_experiment(output_file, seed, benchmark_name, workload, initial_budget, num_function_evals, method_name, empirical_meta_configs, grid_meta_configs, learned_initial_design, search_space_pruning, percent_meta_tasks, percent_meta_data):

    # with open(output_file, 'r') as fh:
    #     json.load(fh)
    # print('Output file %s exists - shutting down.' % output_file)

    if empirical_meta_configs and grid_meta_configs:
        raise ValueError('Only one allowed at a time!')
    # Use the same seed for each method benchmarked!
    rng_initial_design = np.random.RandomState(seed)

    target_task, data_by_task, space = get_benchmark(benchmark_name, seed, workload, output_file)
    # benchmark = get_benchmark(benchmark_name, seed, workload)
    # data_by_task = benchmark.get_meta_data(fixed_grid=grid_meta_configs)
    acquisition_function_maximizer = None
    acquisition_function_maximizer_kwargs = None
    initial_design = LHDesign
    initial_design_kwargs = {'init_budget': initial_budget, 'rng': rng_initial_design}
    initial_configurations = None


    # Do custom changes to the number of function evaluations.
    # The multiplier is used for random 2x, random 4x, etc.
    # However, some benchmarks don't provide enough recorded configurations
    # for this, so we cap the number of function evaluations here.


    def wrapper(config: Configuration, **kwargs) -> float:
        return target_task.objective_function(config)['function_value']

    # Disable SMAC using the pynisher to limit memory and time usage of subprocesses. If you're using
    # this code for some real benchmarks, make sure to enable this again!!!
    tae_kwargs = {'use_pynisher': False}

    # Now load data from previous runs if possible
    # if empirical_meta_configs is True:
    #     data_by_task_new = {}
    #     meta_config_files_dir, _ = os.path.split(output_file)
    #     for task_id in data_by_task:
    #         if benchmark_name in ['openml-glmnet', 'openml-svm', 'openml-xgb']:
    #             data_by_task_new[task_id] = {}
    #         # TODO change this number if changing the number of seed!

    #         meta_config_file = os.path.join(meta_config_files_dir, '..', 'gpmap-10',
    #                                         '%d_50_%d.configs' % (seed, task_id))
    #         with open(meta_config_file) as fh:
    #             metadata = json.load(fh)
    #         configurations = []
    #         targets = []
    #         for config, target in metadata:
    #             configurations.append(Configuration(
    #                 configuration_space=benchmark.get_configuration_space(), values=config)
    #             )
    #             targets.append(target)
    #         targets = np.array(targets)
    #         data_by_task_new[task_id] = {'configurations': configurations, 'y': targets}
    #     data_by_task = data_by_task_new
    #     print('Metadata available for tasks:', {key: len(data_by_task[key]['y']) for key in data_by_task})

    # subsample data and/or number of meta-tasks
    dropping_rng = np.random.RandomState(seed + 13475)
    # percent_meta_tasks = args.percent_meta_tasks
    # if percent_meta_tasks == 'rand':
    #     percent_meta_tasks = dropping_rng.uniform(0.1, 1.0)
    # else:
    #     percent_meta_tasks = float(percent_meta_tasks)
    # if percent_meta_tasks < 1:
    #     actual_num_base_tasks = len(data_by_task)
    #     keep_num_base_tasks = int(np.ceil(actual_num_base_tasks * percent_meta_tasks))
    #     print('Percent meta tasks', percent_meta_tasks, 'keeping only', keep_num_base_tasks, 'tasks')
    #     if keep_num_base_tasks < actual_num_base_tasks:
    #         base_tasks_to_drop = dropping_rng.choice(
    #             list(data_by_task.keys()),
    #             replace=False,
    #             size=actual_num_base_tasks - keep_num_base_tasks,
    #         )
    #         for base_task_to_drop in base_tasks_to_drop:
    #             del data_by_task[base_task_to_drop]
    # if percent_meta_data == 'rand' or float(percent_meta_data) < 1:
    #     for task_id in data_by_task:
    #         if percent_meta_data == 'rand':
    #             percent_meta_data_ = dropping_rng.uniform(0.1, 1.0)
    #         else:
    #             percent_meta_data_ = float(percent_meta_data)
    #         num_configurations = len(data_by_task[task_id]['configurations'])
    #         keep_num_configurations = int(np.ceil(num_configurations * percent_meta_data_))
    #         print('Percent meta data', percent_meta_data, 'keeping only', keep_num_configurations,
    #             'configurations for task', task_id)
    #         if keep_num_configurations < num_configurations:
    #             keep_data_mask = dropping_rng.choice(
    #                 num_configurations, replace=False, size=keep_num_configurations,
    #             )
    #             data_by_task[task_id] = {
    #                 'configurations': [
    #                     config
    #                     for i, config in enumerate(data_by_task[task_id]['configurations'])
    #                     if i in keep_data_mask
    #                 ],
    #                 'y': np.array([
    #                     y
    #                     for i, y in enumerate(data_by_task[task_id]['y'])
    #                     if i in keep_data_mask
    #                 ]),
    #             }

    # Conduct search psace pruning
    if search_space_pruning != 'None':

        full_search_space = benchmark.get_configuration_space()
        to_optimize = [True if isinstance(hp, (UniformIntegerHyperparameter,
                                            UniformFloatHyperparameter)) else False
                    for hp in full_search_space.get_hyperparameters()]

        # Section 4 of Perrone et al., 2019
        if search_space_pruning == 'complete':
            minima_by_dimension = [hp.upper if to_optimize[i] else None
                                for i, hp in enumerate(full_search_space.get_hyperparameters())]
            maxima_by_dimension = [hp.lower if to_optimize[i] else None
                                for i, hp in enumerate(full_search_space.get_hyperparameters())]
            print(minima_by_dimension, maxima_by_dimension, to_optimize)
            for task_id, metadata in data_by_task.items():
                argmin = np.argmin(metadata['y'])
                best_config = metadata['configurations'][argmin]
                for i, hp in enumerate(full_search_space.get_hyperparameters()):
                    if to_optimize[i]:
                        value = best_config[hp.name]
                        if value is None:
                            continue
                        if value < minima_by_dimension[i]:
                            minima_by_dimension[i] = value
                        if value > maxima_by_dimension[i]:
                            maxima_by_dimension[i] = value

        # Section 5 of Perrone et al., 2019
        elif search_space_pruning == 'half':

            num_hyperparameters = len(full_search_space.get_hyperparameters())
            num_tasks = len(data_by_task)
            bounds = [(0, 1)] * (num_hyperparameters * 2)

            optima = []
            for task_id, metadata in data_by_task.items():
                argmin = np.argmin(metadata['y'])
                best_config = metadata['configurations'][argmin]
                optima.append(best_config.get_array())
                bounds.append((0, 1))
                bounds.append((0, 1))
            import scipy.optimize
            optima = np.array(optima)

            def _optimizee(x, lambda_, return_n_violations=False):
                x = np.round(x, 12)
                l = x[: num_hyperparameters]
                u = x[num_hyperparameters: 2 * num_hyperparameters]
                slack_minus = x[num_hyperparameters * 2: num_hyperparameters * 2 + num_tasks]
                slack_plus = x[num_hyperparameters * 2 + num_tasks:]

                n_violations = 0
                for t in range(num_tasks):
                    for i in range(num_hyperparameters):
                        if not to_optimize[i]:
                            continue
                        if np.isfinite(optima[t][i]) and (optima[t][i] < l[i] or optima[t][i] > u[i]):
                            n_violations += 1
                            break
                if return_n_violations:
                    return n_violations

                rval = (
                    (lambda_ / 2 * np.power(np.linalg.norm(u - l, 2), 2))
                    + (1 / (2 * num_tasks) * np.sum(slack_minus) + np.sum(slack_plus))
                )
                return rval

            minima_by_dimension = [0 for i in
                                range(len(full_search_space.get_hyperparameters()))]
            maxima_by_dimension = [1 for i in
                                range(len(full_search_space.get_hyperparameters()))]
            # The paper isn't specific about the values to use for lambda...
            for lambda_ in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
                init = []
                while len(init) < 1:
                    cand = [0] * num_hyperparameters
                    cand.extend([1] * num_hyperparameters)
                    cand.extend([0] * (2 * num_tasks))
                    cand = np.array(cand)
                    if _optimizee(cand, lambda_) < 10e7:
                        init.append(cand)
                init = np.array(init)

                constraints = []
                class LowerConstraint:
                    def __init__(self, i, t):
                        self.i = i
                        self.t = t

                    def __call__(self, x):
                        rval = x[self.i] - x[num_hyperparameters * 2 + self.t] - optima[self.t, self.i]
                        return rval if np.isfinite(rval) else 0

                class UpperConstraint:
                    def __init__(self, i, t):
                        self.i = i
                        self.t = t

                    def __call__(self, x):
                        rval = optima[self.t, self.i] - x[num_hyperparameters * 2 + num_tasks + self.t] - x[num_hyperparameters + self.i]
                        return rval if np.isfinite(rval) else 0

                for t in range(num_tasks):
                    for i in range(num_hyperparameters):

                        if not to_optimize[i]:
                            continue

                        constraints.append(scipy.optimize.NonlinearConstraint(
                            LowerConstraint(i, t), -np.inf, 0)
                        )
                        constraints.append(scipy.optimize.NonlinearConstraint(
                            UpperConstraint(i, t), -np.inf, 0)
                        )

                res = scipy.optimize.minimize(
                    _optimizee, bounds=bounds, args=(lambda_, ),
                    x0=init[0],
                    tol=1e-12,
                    constraints=constraints,
                )
                print(res)
                n_violations = _optimizee(res.x, lambda_, return_n_violations=True)
                print('Number of violations', n_violations)
                if n_violations > 25:
                    continue
                else:
                    result = np.round(res.x, 12)
                    minima_by_dimension = [result[i] for i in
                                        range(len(full_search_space.get_hyperparameters()))]
                    maxima_by_dimension = [result[num_hyperparameters + i] for i in
                                        range(len(full_search_space.get_hyperparameters()))]
                    break

        else:
            raise ValueError(search_space_pruning)

        print('Original configuration space')
        print(benchmark.get_configuration_space())
        print('Pruned configuration space')
        configuration_space = ConfigurationSpace()
        for i, hp in enumerate(full_search_space.get_hyperparameters()):
            if to_optimize[i]:
                if search_space_pruning == 'half':
                    tmp_config = full_search_space.get_default_configuration()
                    vector = tmp_config.get_array()
                    vector[i] = minima_by_dimension[i]
                    tmp_config = Configuration(full_search_space, vector=vector)
                    new_lower = tmp_config[hp.name]
                    vector[i] = maxima_by_dimension[i]
                    tmp_config = Configuration(full_search_space, vector=vector)
                    new_upper = tmp_config[hp.name]
                else:
                    new_lower = minima_by_dimension[i]
                    new_upper = maxima_by_dimension[i]
                if isinstance(hp, UniformFloatHyperparameter):
                    new_hp = UniformFloatHyperparameter(
                        name=hp.name,
                        lower=new_lower,
                        upper=new_upper,
                        log=hp.log,
                    )
                elif isinstance(hp, UniformIntegerHyperparameter):
                    new_hp = UniformIntegerHyperparameter(
                        name=hp.name,
                        lower=new_lower,
                        upper=new_upper,
                        log=hp.log,
                    )
                else:
                    raise ValueError(type(hp))
            else:
                new_hp = copy.deepcopy(hp)
            configuration_space.add_hyperparameter(new_hp)

        for condition in full_search_space.get_conditions():
            hp1 = configuration_space.get_hyperparameter(condition.child.name)
            hp2 = configuration_space.get_hyperparameter(condition.parent.name)
            configuration_space.add_condition(type(condition)(hp1, hp2, condition.value))

        print(configuration_space)
        reduced_configuration_space = configuration_space

        if benchmark_name in ['adaboost', 'svm', 'nn']:
            fixed_set_configurations = []
            for config in benchmark.data[task].keys():
                try:
                    Configuration(reduced_configuration_space, config.get_dictionary())
                    fixed_set_configurations.append(config)
                except:
                    continue
            acquisition_function_maximizer_kwargs['configurations'] = fixed_set_configurations
            print('Using only %d configurations' %
                len(acquisition_function_maximizer_kwargs['configurations']))
            configuration_space = benchmark.get_configuration_space()

    else:
        configuration_space = space
        reduced_configuration_space = configuration_space


    scenario = Scenario({
        'run_obj': 'quality',
        'runcount_limit': num_function_evals,
        'cs': reduced_configuration_space,
        'deterministic': True,
        'output_dir': None,
    })

   
    # Now learn an initial design
    if learned_initial_design in ['scaled', 'unscaled', 'copula']:
        from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
        from smac.runhistory.runhistory import RunHistory
        from smac.tae.execute_ta_run import StatusType
        from external.rgpe.utils import get_gaussian_process
        from smac.epm.util_funcs import get_types

        if learned_initial_design != 'copula':
            print('Learned init with scaled/unscaled')
            rh2epm = RunHistory2EPM4Cost(
                scenario=scenario,
                num_params=len(scenario.cs.get_hyperparameter_names()),
                success_states=[StatusType.SUCCESS],
            )
        else:
            print('Learned init with copula transform')
            from external.rgpe.utils import copula_transform
            class CopulaRH2EPM(RunHistory2EPM4Cost):

                def transform_response_values(self, values: np.ndarray) -> np.ndarray:
                    return copula_transform(values)
            rh2epm = CopulaRH2EPM(
                scenario=scenario,
                num_params=len(scenario.cs.get_hyperparameter_names()),
                success_states=[StatusType.SUCCESS],
            )

        new_initial_design = []

        minima = {}
        maxima = {}
        candidate_configurations = list()
        candidate_set = set()
        benchmark_class = benchmark.__class__
        meta_models = {}
        meta_models_rng = np.random.RandomState(seed)
        for task_id, metadata in data_by_task.items():
            if benchmark_name in ['openml-svm', 'openml-xgb', 'openml-glmnet']:
                dataset_id = task_to_dataset_mapping[task_id]
                meta_benchmark = benchmark_class(rng=seed, dataset_id=dataset_id)
            else:
                meta_benchmark = benchmark_class(task=task_id, rng=seed, load_all=False)
            if learned_initial_design == 'scaled':
                minima[task_id] = meta_benchmark.get_empirical_f_opt()
                if hasattr(meta_benchmark, 'get_empirical_f_worst'):
                    maxima[task_id] = meta_benchmark.get_empirical_f_worst()
                elif hasattr(meta_benchmark, 'get_empirical_f_max'):
                    maxima[task_id] = meta_benchmark.get_empirical_f_max()
                else:
                    raise NotImplementedError()
            rh = RunHistory()
            for config, target in zip(metadata['configurations'], metadata['y']):
                rh.add(config=config, cost=target, time=0, status=StatusType.SUCCESS)

            types, bounds = get_types(benchmark.get_configuration_space(), None)
            gp = get_gaussian_process(meta_benchmark.get_configuration_space(), rng=meta_models_rng,
                                    bounds=bounds, types=types, kernel=None)
            X, y = rh2epm.transform(rh)
            gp.train(X, y)
            meta_models[task_id] = gp
            for config in metadata['configurations']:
                if config not in candidate_set:
                    if benchmark_name in ['adaboost', 'svm', 'nn']:
                        if config not in acquisition_function_maximizer_kwargs['configurations']:
                            continue
                    else:
                        try:
                            Configuration(reduced_configuration_space, config.get_dictionary())
                        except Exception as e:
                            continue
                    candidate_configurations.append(config)
                    candidate_set.add(config)

        print('Using %d candidates for the initial design' % len(candidate_configurations))
        predicted_losses_cache = dict()
        def target_function(config, previous_losses):
            losses = []
            for i, (task_id, meta_benchmark) in enumerate(data_by_task.items()):
                meta_model = meta_models[task_id]

                key = (config, task_id)
                if key in predicted_losses_cache:
                    loss_cfg = predicted_losses_cache[key]
                else:
                    loss_cfg, _ = meta_model.predict(config.get_array().reshape((1, -1)))
                    if learned_initial_design == 'scaled':
                        minimum = minima[task_id]
                        diff = maxima[task_id] - minimum
                        diff = diff if diff > 0 else 1
                        loss_cfg = (loss_cfg - minimum) / diff
                    predicted_losses_cache[key] = loss_cfg
                if loss_cfg < previous_losses[i]:
                    tmp_loss = loss_cfg
                else:
                    tmp_loss = previous_losses[i]
                losses.append(tmp_loss)

            return np.mean(losses), losses

        current_loss_cache = [np.inf] * len(data_by_task)
        for i in range(args.n_init):
            losses = []
            loss_cache = []
            for j, candidate_config in enumerate(candidate_configurations):
                loss, loss_cache_tmp = target_function(candidate_config, current_loss_cache)
                losses.append(loss)
                loss_cache.append(loss_cache_tmp)
            min_loss = np.min(losses)
            min_losses_indices = np.where(losses == min_loss)[0]
            argmin = meta_models_rng.choice(min_losses_indices)
            print(argmin, losses[argmin], len(losses), losses)
            new_initial_design.append(candidate_configurations[argmin])
            current_loss_cache = loss_cache[argmin]
            del candidate_configurations[argmin]

        initial_configurations = copy.deepcopy(new_initial_design)
        initial_design = None
        initial_design_kwargs = None

        del meta_models

        print('Learned initial design')
        print(initial_configurations)


    from external.rgpe.methods.rgpe import RGPE
    from external.rgpe.utils import EI as EI4RGPE
    acquisition_function = EI4RGPE
    # acquisition_function = EI
    acquisition_function_kwargs = {}

    num_posterior_samples = 1000
    sampling_mode = 'bootstrap'
    variance_mode = 'target'
    normalization = 'None'

    weight_dilution_strategy = '95'
    print(num_posterior_samples, acquisition_function, sampling_mode, variance_mode,
        weight_dilution_strategy, normalization)
    method = SMAC4BO(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        tae_runner=wrapper,
        tae_runner_kwargs=tae_kwargs,
        initial_design=initial_design,
        initial_design_kwargs=initial_design_kwargs,
        initial_configurations=initial_configurations,
        model=RGPE,
        model_kwargs={
            'training_data': data_by_task,
            'weight_dilution_strategy': weight_dilution_strategy,
            'number_of_function_evaluations': num_function_evals,
            'variance_mode': variance_mode,
            'num_posterior_samples': num_posterior_samples,
            'sampling_mode': sampling_mode,
            'normalization': normalization,
        },
        acquisition_function_kwargs=acquisition_function_kwargs,
        acquisition_function=acquisition_function,
        acquisition_function_optimizer=acquisition_function_maximizer,
        acquisition_function_optimizer_kwargs=acquisition_function_maximizer_kwargs,
    )

    # Disable random configurations form SMAC
    method.solver.epm_chooser.random_configuration_chooser = None

    # And now run the optimizer
    method.optimize()

    if hasattr(method.solver.epm_chooser.model, 'weights_over_time'):
        weight_file = output_file
        weight_file = weight_file.replace('.json', '.weights')
        weights_over_time = method.solver.epm_chooser.model.weights_over_time
        weights_over_time = [
            [float(weight) for weight in weights]
            for weights in weights_over_time
        ]
        with open(weight_file, 'w') as fh:
            json.dump(weights_over_time, fh, indent=4)
    if hasattr(method.solver.epm_chooser.model, 'p_drop_over_time'):
        p_drop_file = output_file
        p_drop_file = p_drop_file.replace('.json', '.pdrop')
        p_drop_over_time = method.solver.epm_chooser.model.p_drop_over_time
        p_drop_over_time = [
            [float(p_drop) for p_drop in drops]
            for drops in p_drop_over_time
        ]
        with open(p_drop_file, 'w') as fh:
            json.dump(p_drop_over_time, fh, indent=4)

    # Dump the evaluated configurations as meta-data for later runs
    print(method_name, method_name == 'gpmap')
    if method_name == 'gpmap':
        rh = method.get_runhistory()
        evaluated_configurations = []
        for config in rh.config_ids:
            cost = rh.get_cost(config)
            print(cost)
            evaluated_configurations.append([config.get_dictionary(), cost])
        print(evaluated_configurations)
        evaluated_configs_file = output_file
        evaluated_configs_file = evaluated_configs_file.replace('.json', '.configs')
        print(evaluated_configs_file)
        with open(evaluated_configs_file, 'w') as fh:
            json.dump(evaluated_configurations, fh)
class task_wrapper():
    def __init__(self, search_space, task, task_name, workload, seed, output_file) -> None:
        self.search_space = search_space
        self.task = task
        self.task_name = task_name
        self.workload = workload
        self.seed = seed
        self.output_file = output_file

    def objective_function(self, X, **kwargs):
        sample = np.array([v for k,v in X.items()])
        query_data = self.search_space.map_to_design_space(sample)
        Y = self.task.objective_function(query_data)['f1']
        
        # Read existing trajectory from file or create new one
        try:
            with open(self.output_file, 'r') as f:
                data = json.load(f)
                trajectory = data['result']['history']
        except:
            trajectory = []
            
        # Add new evaluation to trajectory
        trajectory.append({
            'iteration': len(trajectory),
            'params': sample.tolist(), 
            'loss': float(Y)
        })
        
        # Update result with best seen so far
        best_idx = np.argmax([t['loss'] for t in trajectory])
        result = {
            'best_params': trajectory[best_idx]['params'],
            'best_value': trajectory[best_idx]['loss'],
            'history': trajectory
        }
        
        # Save updated result to file
        with open(self.output_file, 'w') as f:
            json.dump({
                'task_name': self.task_name,
                'workload': self.workload, 
                'seed': self.seed,
                'result': result
            }, f, indent=2)
            
        return {'function_value': Y}



def get_source_data(task_name, services):
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
        X = np.array(var_data)
        Y = np.array(obj_data)
        train_data[sub_dataset_id] = {'X': X, 'Y': Y}
        sub_dataset_id += 1
    return train_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--iteration-multiplier', type=int, default=1)
    parser.add_argument('--empirical-meta-configs', action='store_true')
    parser.add_argument('--grid-meta-configs', action='store_true')
    parser.add_argument('--learned-initial-design', choices=['None', 'unscaled', 'scaled', 'copula'],
                        default='None')
    parser.add_argument('--search-space-pruning', choices=['None', 'complete', 'half'], default='None')
    parser.add_argument('--percent-meta-tasks', default=1.0)
    parser.add_argument('--percent-meta-data', default=1.0)
    args, unknown = parser.parse_known_args()
    
    seed = 0
    input_dim = 10
    tasks = [
            ('Rastrigin', Rastrigin, input_dim),
            ('Schwefel', Schwefel, input_dim),
            ('Ackley', Ackley, input_dim),
            ('Griewank', Griewank, input_dim),
            ('Rosenbrock', Rosenbrock, input_dim),
        ]
    
    n_init = 5
    budget = input_dim * 1
    services = Services(None, None, None)
    services._initialize_modules()
    
    configurations = services.configer.get_configuration()

    workloads = [0, 1, 2, 3, 4, 5]  # Different workloads
    seeds = [0, 1, 2, 3, 4, 5]
    
    
    def get_benchmark(
        benchmark_name: str,
        seed: int,
        workload: int,
        output_file: str,
        output_noise: float = 0.0,
        params_source: List[Dict[str, float]] = None,
        params_target: Dict[str, float] = None,
    ):
        """Create the benchmark object."""
        target_task = task_class(
                task_name=benchmark_name,
                budget_type='FEs',
                budget=budget,
                seed=seed,
                workload=workload,
                params={'input_dim': input_dim}
            )

        task = task_wrapper(output_file=output_file, task_name=benchmark_name, workload=workload, seed=seed, search_space=target_task.configuration_space, task=target_task)

        # Get parameter space
        def get_configspace():
            space = ConfigurationSpace()
            original_ranges = target_task.configuration_space.original_ranges
            for param_name, param_range in original_ranges.items():
                space.add_hyperparameter(cs.UniformFloatHyperparameter(param_name, lower=param_range[0], upper=param_range[1]))
            return space
        space = get_configspace()
        

        datasets = get_source_data(task_name, services)
        source_data = {}
        set_id = 0
        for name, data in datasets.items():
            source_data[set_id] = {'configurations': data['X'], 'y': data['Y'].reshape(-1, 1)}
            set_id += 1

        return task, source_data, space
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'results/RGPE_{timestamp}'
    os.mkdir(output_dir)
    for task_name, task_class, input_dim in tqdm.tqdm(tasks, desc="Processing tasks"):
        for workload in workloads:
            for seed in seeds:
                output_file = f'{output_dir}/{task_name}_workload{workload}_seed{seed}.json'
                run_experiment(output_file, seed, task_name, workload, initial_budget=n_init, num_function_evals=budget, method_name='rgpe', empirical_meta_configs=True, grid_meta_configs=False, learned_initial_design='None', search_space_pruning='None', percent_meta_tasks = 1.0, percent_meta_data=1.0)