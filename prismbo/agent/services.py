import os
import signal
import time
from multiprocessing import Manager, Process

import numpy as np


from prismbo.agent.chat.openai_chat import OpenAIChat
from prismbo.agent.config import ChatbotConfig, Configer
from prismbo.agent.registry import *
from prismbo.analysis.sensitivity import sensitivity_analysis
from prismbo.benchmark.instantiate_problems import InstantiateProblems
from prismbo.datamanager.manager import Database, DataManager
from prismbo.optimizer.construct_optimizer import (ConstructOptimizer,
                                                    ConstructSelector)
from prismbo.datamanager.bnf import parse_task_from_string
from prismbo.utils.log import logger
from prismbo.analysis.mds import FootPrint

class Services:
    def __init__(self, task_queue, result_queue, lock):
        self.chatbotconfig = ChatbotConfig()
        self.configer = Configer()
        
        # DataManager for general tasks, not specific optimization tasks
        self.data_manager = DataManager()
        self.tasks_info = []

        self.openai_chat = OpenAIChat(
            api_key=self.chatbotconfig.OPENAI_API_KEY,
            model="gpt-3.5-turbo",
            base_url=self.chatbotconfig.OPENAI_URL,
            data_manager= self.data_manager
        )

        self._initialize_modules()
        self.process_info = Manager().dict()
        self.lock = Manager().Lock()
        self.multi_run = False


    def chat(self, user_input):
        response_content = self.openai_chat.get_response(user_input)
        return response_content

    def _initialize_modules(self):
        import prismbo.benchmark.rnainversedesign
        # import prismbo.benchmark.hpo.HPOB
        # import prismbo.benchmark.hpo.HPOOOD
        import prismbo.benchmark.hpo
        import prismbo.benchmark.synthetic
        import prismbo.benchmark.gym
        try:
            import prismbo.benchmark.csstuning
        except:
            logger.warning("CSSTuning module not found. Please install the CSSTuning package to use this functionality.")
        import prismbo.optimizer.acquisition_function
        import prismbo.optimizer.model
        import prismbo.optimizer.normalizer
        import prismbo.optimizer.pretrain
        import prismbo.optimizer.refiner
        import prismbo.optimizer.initialization
        import prismbo.optimizer.selector
        
        
    def get_modules(self):
        basic_info = {}
        tasks_info = []
        selector_info = []
        model_info = []
        sampler_info = []
        acf_info = []
        pretrain_info = [{'name':'None'}]
        refiner_info = [{'name':'None'}]
        normalizer_info = [{'name':'None'}]

        # tasks information
        task_names = problem_registry.list_names()
        for name in task_names:
            if problem_registry[name].problem_type == "synthetic":
                num_obj = problem_registry[name].num_objectives
                num_var = problem_registry[name].num_variables
                task_info = {
                    "name": name,
                    "problem_type": "synthetic",
                    "anyDim": "True",
                    'num_vars': [],
                    "num_objs": [1],
                    "workloads": [],
                    "fidelity": [],
                }
            else:
                num_obj = problem_registry[name].num_objectives
                num_var = problem_registry[name].num_variables
                fidelity = problem_registry[name].fidelity
                workloads = problem_registry[name].workloads
                problem_type = problem_registry[name].problem_type
                task_info = {
                    "name": name,
                    "problem_type": problem_type,
                    "anyDim": False,
                    "num_vars": [num_var],
                    "num_objs": [num_obj],
                    "workloads": workloads,
                    "fidelity": [fidelity],
                }
            tasks_info.append(task_info)
        basic_info["TasksData"] = tasks_info

        sampler_names = sampler_registry.list_names()
        for name in sampler_names:
            sampler_info.append({"name": name})
        basic_info["Sampler"] = sampler_info

        refiner_names = space_refiner_registry.list_names()
        for name in refiner_names:
            refiner_info.append({"name": name})
        basic_info["SpaceRefiner"] = refiner_info

        pretrain_names = pretrain_registry.list_names()
        for name in pretrain_names:
            pretrain_info.append({"name": name})
        basic_info["Pretrain"] = pretrain_info

        model_names = model_registry.list_names()
        for name in model_names:
            model_info.append({"name": name})
        basic_info["Model"] = model_info

        acf_names = acf_registry.list_names()
        for name in acf_names:
            acf_info.append({"name": name})
        basic_info["ACF"] = acf_info

        selector_names = selector_registry.list_names()
        for name in selector_names:
            selector_info.append({"name": name})
        
        basic_info["DataSelector"] = selector_info
        
        normalizer_names = normalizer_registry.list_names()
        for name in normalizer_names:
            normalizer_info.append({"name": name})
        basic_info["Normalizer"] = normalizer_info

        return basic_info
    
    
    def get_comparision_modules(self):
        module_info = {}
        model_info = []
        sampler_info = []
        acf_info = []
        pretrain_info = ['None']
        refiner_info = ['None']
        normalizer_info = ['None']

        sampler_names = sampler_registry.list_names()
        for name in sampler_names:
            sampler_info.append(name)
        module_info["Sampler"] = sampler_info

        refiner_names = space_refiner_registry.list_names()
        for name in refiner_names:
            refiner_info.append(name)
        module_info["Refiner"] = refiner_info

        pretrain_names = pretrain_registry.list_names()
        for name in pretrain_names:
            pretrain_info.append(name)
        module_info["Pretrain"] = pretrain_info

        model_names = model_registry.list_names()
        for name in model_names:
            model_info.append(name)
        module_info["Model"] = model_info

        acf_names = acf_registry.list_names()
        for name in acf_names:
            acf_info.append(name)
        module_info["ACF"] = acf_info
        
        normalizer_names = normalizer_registry.list_names()
        for name in normalizer_names:
            normalizer_info.append(name)
        module_info["Normalizer"] = normalizer_info

        return module_info

    def search_dataset(self, search_method, dataset_name, dataset_info):
        if search_method == 'Fuzzy':
            datasets_list = {"isExact": False, 
                             "datasets": list(self.data_manager.search_datasets_by_name(dataset_name))}
        elif search_method == 'Hash':
            dataset_detail_info = self.data_manager.get_dataset_info(dataset_name)
            if dataset_detail_info:
                datasets_list = {"isExact": True, "datasets": dataset_detail_info['additional_config']}
            else:
                raise ValueError("Dataset not found")
        elif search_method == 'LSH':
            datasets_list = {"isExact": False, 
                             "datasets":list(self.data_manager.search_similar_datasets(dataset_name, dataset_info))}
            
        else:
            raise ValueError("Invalid search method")

        return datasets_list
   
    def convert_metadata(self, conditions):
        type_map = {
            "NumVars": int,
            "NumObjs": int,
            "Workload": int,
            "Seed": int,
            # Add other fields as necessary
        }
        converted_conditions = {}
        for key, value in conditions.items():
            if key in type_map:
                try:
                    # Convert the value according to its expected type
                    if type_map[key] == int:
                        converted_conditions[key] = int(value)
                    elif type_map[key] == float:
                        converted_conditions[key] = float(value)
                    elif type_map[key] == bool:
                        converted_conditions[key] = value.lower() in ['true', '1', 't', 'yes', 'y']
                    else:
                        converted_conditions[key] = value  # Assume string or no conversion needed
                except ValueError:
                    raise ValueError(f"Invalid value for {key}: {value}")
            else:
                # If no specific type is expected, assume string
                converted_conditions[key] = value

        return converted_conditions
 
    def comparision_search(self, conditions):
        conditions = {k: v for k, v in conditions.items() if v}
        conditions = self.convert_metadata(conditions)
        
        key_map = {
            "TaskName": "problem_name",
            "NumVars": "dimensions",
            "NumObjs": "objectives",
            "Fidelity": "fidelities",
            "Workload": "workloads",
            "Seed": "seeds",
            "Refiner": "space_refiner",
            "Sampler": "sampler",
            "Pretrain": "pretrain",
            "Model": "model",
            "ACF": "acf",
            "Normalizer": "normalizer"
        }
        
        # change key in conditions to match the key in database
        conditions = {key_map[k]: v for k, v in conditions.items() if k in key_map}
        
        return self.data_manager.db.search_tables_by_metadata(conditions)
    


    def receive_configuration(self, config_info):
        self.configer.set_configuration(config_info)
        return

    def get_all_datasets(self):
        all_tables = self.data_manager.db.get_table_list()
        return [self.data_manager.db.query_dataset_info(table) for table in all_tables]
    
    def get_experiment_datasets(self):
        experiment_tables = self.data_manager.db.get_table_list()
        return [(experiment_tables[table_id],self.data_manager.db.query_dataset_info(table)) for table_id, table in enumerate(experiment_tables)] 
   
    def construct_dataset_info(self, task_set, config, seed):
        dataset_info = {}
        dataset_info["variables"] = [
            {"name": var.name, "type": var.type, "range": var.range}
            for var_name, var in task_set.get_cur_searchspace_info().items()
        ]
        dataset_info["objectives"] = [
            {"name": name, "type": type}
            for name, type in task_set.get_curobj_info().items()
        ]
        dataset_info["fidelities"] = [
            {"name": var.name, "type": var.type, "range": var.range}
            for var_name, var in task_set.get_cur_fidelity_info().items()
        ]
        
        meta_features = parse_task_from_string(config.get('experimentDescription', ''))
        # Simplify dataset name construction
        timestamp = int(time.time())
        dataset_name = f"{task_set.get_curname()}_w{task_set.get_cur_workload()}_s{seed}_{timestamp}"

        dataset_info['additional_config'] = {
            "experimentName": config.get('experimentName', ''),
            "metafeatures": meta_features,
            "problem_name": task_set.get_curname(),
            "dim": len(dataset_info["variables"]),
            "obj": len(dataset_info["objectives"]),
            "fidelity": ', '.join([d['name'] for d in dataset_info["fidelities"] if 'name' in d]) if dataset_info["fidelities"] else '',
            "workloads": task_set.get_cur_workload(),
            "budget_type": task_set.get_cur_budgettype(),
            "initial_number": config['optimizer']['Initialization']['InitNum'],
            "budget": task_set.get_cur_budget(),
            "seeds": seed,
            "SearchSpace": config['optimizer']['SearchSpace']['type'],
            "Initialization": config['optimizer']['Initialization']['type'],
            "Pretrain": config['optimizer']['Pretrain']['type'],
            "Model": config['optimizer']['Model']['type'],
            "AcquisitionFunction": config['optimizer']['AcquisitionFunction']['type'],
            "Normalizer": config['optimizer']['Normalizer']['type'],
            "AutoSelect": {'SearchSpace':config['optimizer']['SearchSpace']['autoSelect'],
                          'Initialization':config['optimizer']['Initialization']['autoSelect'],
                          'AcquisitionFunction':config['optimizer']['AcquisitionFunction']['autoSelect'],
                          'Pretrain':config['optimizer']['Pretrain']['autoSelect'],
                          'Model':config['optimizer']['Model']['autoSelect'],
                          'Normalizer':config['optimizer']['Normalizer']['autoSelect']},
            "auxiliaryData": {'SearchSpace':config['optimizer']['SearchSpace']['auxiliaryData'],
                              'Initialization':config['optimizer']['Initialization']['auxiliaryData'],
                              'AcquisitionFunction':config['optimizer']['AcquisitionFunction']['auxiliaryData'],
                              'Pretrain':config['optimizer']['Pretrain']['auxiliaryData'],
                              'Model':config['optimizer']['Model']['auxiliaryData'],
                              'Normalizer':config['optimizer']['Normalizer']['auxiliaryData']},
        }

        return dataset_info, dataset_name
 
    def get_metadata(self, module_name):
        configuration = self.configer.get_configuration()
        metadata_names = configuration['optimizer'][module_name]['auxiliaryData']
        if len(metadata_names):
            metadata = {}
            metadata_info = {}
            for dataset_name in metadata_names:
                metadata[dataset_name] = self.data_manager.db.select_data(dataset_name)
                metadata_info[dataset_name] = self.data_manager.db.query_dataset_info(dataset_name)
            return metadata, metadata_info
        else:
            return {}, {}
    
    def Ifautoselect(self, module_name):
        configuration = self.configer.get_configuration()
        autoselect = configuration['optimizer'][module_name]['autoSelect']
        return autoselect
    
    def auto_get_data(self, task_name, task_info):
        datasets_list = list(self.data_manager.search_similar_datasets(task_info))
        if len(datasets_list):
            metadata = {}
            metadata_info = {}
            for dataset_name in datasets_list:
                if dataset_name == task_name:
                    continue
                data = self.data_manager.db.select_data(dataset_name)
                if len(data) > 0:
                    metadata[dataset_name] = data
                    metadata_info[dataset_name] = self.data_manager.db.query_dataset_info(dataset_name)
        return metadata, metadata_info
    
    def save_data(self, dataset_name, parameters, observations, iteration):
        data = [{} for i in range(len(parameters))]
        [data[i].update(parameters[i]) for i in range(len(parameters))]
        [data[i].update(observations[i]) for i in range(len(parameters))]
        [data[i].update({'batch':iteration}) for i in range(len(parameters))]
        self.data_manager.db.insert_data(dataset_name, data)
    
    def remove_dataset(self, dataset_name):
        if isinstance(dataset_name, str):
            self.data_manager.db.remove_table(dataset_name)
        elif isinstance(dataset_name, list):
            for name in dataset_name:
                self.data_manager.db.remove_table(name)
        else:
            raise ValueError("Invalid dataset name")

    def set_multi_run(self, data):
        self.multi_run = data['is_multi']
        

    def run_optimize(self):
        configurations = self.configer.get_configuration()
        seeds = configurations['seeds'].split(',')
        seeds = [int(seed) for seed in seeds]
        
        # Create a separate process for each seed
        if self.multi_run:
            # Multi-process execution
            process_list = []
            for seed in seeds:
                p = Process(target=self._run_optimize_process, args=(int(seed), configurations))
                process_list.append(p)
                p.start()
            
            for p in process_list:
                p.join()
        else:
            # Single process execution
            for seed in seeds:
                self._run_optimize_process(int(seed), configurations)
        
    
    def _run_optimize_process(self, seed, configurations):
        # Each process constructs its own DataManager
        try:
            import os
            pid = os.getpid()
            self.process_info[pid] = {'status': 'running', 'seed': [seed], 'budget': [], 'task': [], 'iteration': 0, 'dataset_name': [], 'progress':0}
            logger.info(f"Start process #{pid}")

            # Instantiate problems and optimizer
            task_set = InstantiateProblems(configurations['tasks'], seed)
            optimizer = ConstructOptimizer(configurations['optimizer'], seed)

            while (task_set.get_unsolved_num()):
                search_space = task_set.get_cur_searchspace()
                task_info, task_name = self.construct_dataset_info(task_set, configurations, seed=seed)
                
                self.data_manager.create_dataset(task_name, task_info, overwrite=True)
                self.update_process_info(pid, {'dataset_name': task_name, 'task': task_set.get_curname(), 'budget': task_set.get_cur_budget()})
                
                optimizer.link_task(task_name=task_set.get_curname(), search_space=search_space)

                #step1: initialization
                metadata, metadata_info = self.get_metadata('Initialization')
                autoselect = self.Ifautoselect('Initialization')
                if autoselect:
                    metadata, metadata_info = self.auto_get_data(task_name, task_info)
                samples = optimizer.sample_initial_set(metadata, metadata_info)
                
                parameters = [search_space.map_to_design_space(sample) for sample in samples]

                observations = task_set.f(parameters)

                self.save_data(task_name, parameters, observations, self.process_info[pid]['iteration'])
                optimizer.observe(samples, observations)

                #step2: search space prune
                metadata, metadata_info = self.get_metadata('SearchSpace')
                autoselect = self.Ifautoselect('SearchSpace')
                if autoselect:
                    metadata, metadata_info = self.auto_get_data(task_name, task_info)
                optimizer.search_space_refine(optimizer.search_space, metadata, metadata_info)
                
                #step3: pretrain
                metadata, metadata_info = self.get_metadata('Pretrain')
                autoselect = self.Ifautoselect('Pretrain')
                if autoselect:
                    metadata, metadata_info = self.auto_get_data(task_name, task_info)
                optimizer.pretrain(metadata, metadata_info)
                
                #step4: meta-fit
                metadata, metadata_info = self.get_metadata('Model')
                autoselect = self.Ifautoselect('Model')
                if autoselect:
                    metadata, metadata_info = self.auto_get_data(task_name, task_info)
                optimizer.meta_fit(metadata, metadata_info)
                
                cur_iter = 0
                self.update_process_info(pid, {'iteration': cur_iter + 1})
                
                #step5: Acquisition Function
                metadata, metadata_info = self.get_metadata('AcquisitionFunction')
                autoselect = self.Ifautoselect('AcquisitionFunction')
                if autoselect:
                    metadata, metadata_info = self.auto_get_data(task_name, task_info)
                optimizer.ACF_meta(metadata, metadata_info)
                
                while (task_set.get_rest_budget()):
                    optimizer.fit()
                    suggested_samples = optimizer.suggest()
                    parameters = [optimizer.search_space.map_to_design_space(sample) for sample in suggested_samples]
                    observations = task_set.f(parameters)
                    if observations is None:
                        break
                    self.save_data(task_name, parameters, observations, self.process_info[pid]['iteration'])
                    optimizer.observe(suggested_samples, observations)
                    
                    cur_iter = self.process_info[pid]['iteration']
                    self.update_process_info(pid, {'iteration': cur_iter + 1})
                    self.update_process_info(pid, {'progress': 100 * (task_set.get_cur_budget() - task_set.get_rest_budget()) / task_set.get_cur_budget()})
                    logger.info(f"PID {pid}: Seed {seed}, Task {task_set.get_curname()}, Iteration {self.process_info[pid]['iteration']}")
                task_set.roll()
                # optimizer.meta_observe({'X':optimizer._X, 'Y':optimizer._Y}, optimizer.search_space)
                optimizer.reload_model(model_registry[configurations['optimizer']['Model']['type']](config = configurations['optimizer']['Model']['Parameters']))
                optimizer.reload_acf(acf_registry[configurations['optimizer']['AcquisitionFunction']['type']](config = configurations['optimizer']['AcquisitionFunction']['Parameters']))
                
        except Exception as e:
            logger.error(f"Error in process {pid}: {str(e)}")
            raise e
        finally:
            self.update_process_info(pid, {'status': 'completed'})
        
   
    def terminate_task(self, pid):
        with self.lock:
            if pid in self.process_info:
                dataset_name = self.process_info[pid].get('dataset_name')
                try:
                    os.kill(pid, signal.SIGTERM)
                    logger.info(f"Process {pid} has been terminated.")
                except Exception as e:
                    logger.error(f"Failed to terminate process {pid}: {str(e)}")
                if dataset_name:
                    try:
                        self.data_manager.remove_dataset(dataset_name)
                        logger.info(f"Dataset {dataset_name} associated with process {pid} has been deleted.")
                    except Exception as e:
                        logger.error(f"Failed to delete dataset {dataset_name}: {str(e)}")
                del self.process_info[pid]
            else:
                logger.warning(f"No such process {pid} found in process info.")
    
    def update_process_info(self, pid, updates):
        with self.lock:
            temp_info = self.process_info[pid].copy()
            list_keys = {'seed', 'budget', 'task', 'dataset_name'}
            
            for key, value in updates.items():
                if key in list_keys:
                    if key not in temp_info:
                        temp_info[key] = []
                    if not isinstance(temp_info[key], list):
                        temp_info[key] = [temp_info[key]]
                    temp_info[key].append(value)
                else:
                    temp_info[key] = value
            
            self.process_info[pid] = temp_info
        
    def get_all_process_info(self):
        return dict(self.process_info)
    
    def get_box_plot_data(self, task_names):
        all_data = {}
        for group_id, group in enumerate(task_names):
            all_data[str(group_id)] = []
            for task_name in group:
                data = self.data_manager.db.select_data(task_name)
                table_info = self.data_manager.db.query_dataset_info(task_name)
                objectives = table_info["objectives"]
                obj = objectives[0]["name"]
                try:
                    best_obj = min([d[obj] for d in data])
                except:
                    pass
                all_data[str(group_id)].append(best_obj)

        return all_data
    
    
    
    def get_report_charts(self, task_name):
        all_data = self.data_manager.db.select_data(task_name)

        table_info = self.data_manager.db.query_dataset_info(task_name)
        objectives = table_info["objectives"]
        ranges = [tuple(var['range']) for var in table_info["variables"]]
        initial_number = table_info["additional_config"]["initial_number"]
        obj = objectives[0]["name"]
        obj_type = objectives[0]["type"]

        obj_data = [data[obj] for data in all_data]
        var_data = [[data[var["name"]] for var in table_info["variables"]] for data in all_data]
        variables = [var["name"] for var in table_info["variables"]]
        ret = {}
        ret.update(self.construct_footprint_data(task_name, var_data, ranges, initial_number))
        ret.update(self.construct_trajectory_data(task_name, obj_data, obj_type))
        ret.update(self.construct_importance_data(task_name, var_data, obj_data, variables))

        return ret


    def get_report_traj(self, task_name):
        all_data = self.data_manager.db.select_data(task_name)

        table_info = self.data_manager.db.query_dataset_info(task_name)
        objectives = table_info["objectives"]

        obj = objectives[0]["name"]
        obj_type = objectives[0]["type"]

        obj_data = [data[obj] for data in all_data]
        ret = {}
        ret.update(self.construct_trajectory_data(task_name, obj_data, obj_type))

        return ret

    def construct_footprint_data(self, name, var_data, ranges, initial_number):
        # Initialize the list to store trajectory data and the best value seen so far
        fp = FootPrint(var_data, ranges)
        fp.calculate_distances()
        fp.get_mds()
        scatter_data = {'Initial vectors': fp._reduced_data[:initial_number], 'Decision vectors': fp._reduced_data[initial_number:len(fp.X)], 'Boundary vectors': fp._reduced_data[len(fp.X):]}
        # scatter_data = {}
        return {"ScatterData": scatter_data}
    
    def construct_statistic_trajectory_data(self, task_names):
        all_data = []
        for group_id, group in enumerate(task_names):
            min_data = {'name': f'Algorithm{group_id + 1}', 'average': [], 'uncertainty': []}
            res = []
            max_length = 0
            for task_name in group:
                data = self.data_manager.db.select_data(task_name)
                table_info = self.data_manager.db.query_dataset_info(task_name)
                objectives = table_info["objectives"]
                obj = objectives[0]["name"]
                obj_data = [d[obj] for d in data]
                acc_obj_data = np.minimum.accumulate(obj_data).flatten().tolist()
                res.append(acc_obj_data)
                if len(acc_obj_data) > max_length:
                    max_length = len(acc_obj_data)

            # 计算每个点的中位数和标准差
            for i in range(max_length):
                current_data = [r[i] for r in res if i < len(r)]
                median = np.median(current_data)
                std = np.std(current_data)
                min_data['average'].append({'FEs': i + 1, 'y': median})
                min_data['uncertainty'].append({'FEs': i + 1, 'y': [median - std, median + std]})

            all_data.append(min_data)

        return all_data
        
    
    def construct_trajectory_data(self, name, obj_data, obj_type="minimize"):
        # Initialize the list to store trajectory data and the best value seen so far
        trajectory = []
        best_value = float("inf") if obj_type == "minimize" else -float("inf")
        best_values_so_far = []

        # Loop through each function evaluation
        for index, current_value in enumerate(obj_data, start=1):
            # Update the best value based on the objective type
            if obj_type == "minimize":
                if current_value < best_value:
                    best_value = current_value
            else:  # maximize
                if current_value > best_value:
                    best_value = current_value

            # Append the best value observed so far to the list
            best_values_so_far.append(best_value)
            trajectory.append({"FEs": index, "y": best_value})

        uncertainty = []
        for data_point in trajectory:
            base_value = data_point["y"]
            uncertainty_range = [base_value, base_value]
            uncertainty.append({"FEs": data_point["FEs"], "y": uncertainty_range})

        trajectory_data = {
            "name": name,
            "average": trajectory,
            "uncertainty": uncertainty,
        }

        return {"TrajectoryData": [trajectory_data]}

    def construct_importance_data(self, name, var_data, obj_data, variables):
        importance_data = sensitivity_analysis(np.array(var_data), np.array(obj_data), variables)
        return {"ImportanceData": importance_data}

    def get_configuration(self):
        return self.configer.get_configuration()
