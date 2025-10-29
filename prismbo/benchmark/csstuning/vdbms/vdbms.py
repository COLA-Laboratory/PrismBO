from ast import main
import json
import yaml
import numpy as np
from prismbo.agent.registry import problem_registry
from prismbo.benchmark.problem_base.non_tab_problem import NonTabularProblem
from prismbo.space.fidelity_space import FidelitySpace
from prismbo.space.search_space import SearchSpace
from prismbo.space.variable import *



import sys 
sys.path.append("..") 

import joblib
from scipy.stats import qmc
import json
import numpy as np
import time
import subprocess as sp
import random
import os
# from vdb import *


env = os.environ.copy()
for k in ["http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"]:
    env.pop(k, None)

KNOB_PATH = r'./prismbo/benchmark/csstuning/vdbms/whole_param.json'
INDEX_PARAM_PATH = r'./prismbo/benchmark/csstuning/vdbms/index_param.json'
RUN_ENGINE_PATH = r'./external/vector-db-benchmark-master/run_engine.sh'
VDBMS_DATA_INFO_PATH = r'./prismbo/benchmark/csstuning/vdbms/datasets.json'
CONF_PATH = r'./external/vector-db-benchmark-master/experiments/configurations/milvus-single-node.json'
ORIGIN_PATH = r'./external/vector-db-benchmark-master/engine/servers/milvus-single-node/milvus.yaml.backup'
ADJUST_PATH = r'./external/vector-db-benchmark-master/engine/servers/milvus-single-node/milvus.yaml'


with open(INDEX_PARAM_PATH, 'r') as f:
    INDEX_PARAM_DICT = json.load(f)


def get_datesets_info():
    with open(VDBMS_DATA_INFO_PATH, 'r') as f:
        data_info = json.load(f)
    return data_info

def filter_index_rule(conf):
    for item in INDEX_PARAM_DICT.keys():
        if item not in list(conf.keys()):
            conf[item] = INDEX_PARAM_DICT[item]['default']

    # print(conf)
    conf['nprobe'] = int(conf['nlist'] * conf['nprobe'] / 100)
    conf['nprobe'] = max(1, conf['nprobe'])
    # if conf['nprobe'] > conf['nlist']:
    #     conf['nprobe'] = conf['nlist']

    if conf['index_type'] in ['AUTOINDEX', 'FLAT']:
        building_params = {}
        searching_params = {}
    elif conf['index_type'] in ['IVF_FLAT', 'IVF_SQ8']:
        building_params = {'nlist': conf['nlist']}
        searching_params = {'nprobe': conf['nprobe']}
    elif conf['index_type'] in ['IVF_PQ']:
        building_params = {'nlist': conf['nlist'], 'm': conf['m'], 'nbits': conf['nbits']}
        searching_params = {'nprobe': conf['nprobe']}
    elif conf['index_type'] in ['HNSW']:
        building_params = {'M': conf['M'], 'efConstruction': conf['efConstruction']}
        searching_params = {'ef': conf['ef']}
    elif conf['index_type'] in ['SCANN']:
        building_params = {'nlist': conf['nlist']}
        searching_params = {'nprobe': conf['nprobe'], 'reorder_k': conf['reorder_k']}

    return conf['index_type'], building_params, searching_params

def configure_index(index_type, building_params, searching_params):
    conf_path = CONF_PATH
    with open(conf_path, 'r') as f:
        conf = json.load(f)
    conf[0]['upload_params']['index_type'] = index_type
    conf[0]['upload_params']['index_params'] = building_params
    conf[0]['search_params'][0]['params'] = searching_params
    with open(conf_path, 'w') as f:
        f.write(json.dumps(conf, indent=2))


def filter_system_rule(conf):
    for k in conf.keys():
        if k in ['dataCoord*segment*sealProportion']:
            conf[k] = conf[k] / 100
    return conf


def configure_system(params):
    origin_path = ORIGIN_PATH
    adjust_path = ADJUST_PATH
    with open(origin_path, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    for k,v in params.items():
        pos = k.split('*')
        if len(pos) == 4:
            conf[pos[0]][pos[1]][pos[2]][pos[3]] = v
        elif len(pos) == 3:
            conf[pos[0]][pos[1]][pos[2]] = v
        elif len(pos) == 2:
            conf[pos[0]][pos[1]] = v
    with open(adjust_path, 'w') as f:
        yaml.dump(conf, f)



def LHS_sample(dimension, num_points, seed=0):
    sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    latin_samples = sampler.random(n=num_points)

    return latin_samples

class KnobStand:
    def __init__(self, path) -> None:
        self.path = path
        with open(path, 'r') as f:
            self.knobs_detail = json.load(f)

    def scale_back(self, knob_name, zero_one_val):
        knob = self.knobs_detail[knob_name]
        if knob['type'] == 'integer':
            real_val = zero_one_val * (knob['max'] - knob['min']) + knob['min']
            return int(real_val), int(real_val)

        elif knob['type'] == 'enum':
            enum_size = len(knob['enum_values'])
            enum_index = int(enum_size * zero_one_val)
            enum_index = min(enum_size - 1, enum_index)
            real_val = knob['enum_values'][enum_index]
            return enum_index, real_val
    
    def scale_forward(self, knob_name, real_val):
        knob = self.knobs_detail[knob_name]
        if knob['type'] == 'integer':
            zero_one_val = (real_val - knob['min']) / (knob['max'] - knob['min'])
            return zero_one_val

        elif knob['type'] == 'enum':
            enum_size = len(knob['enum_values'])
            zero_one_val = knob['enum_values'].index(real_val) / enum_size
            return zero_one_val

class StaticEnv:
    def __init__(self, model_path=['XGBoost_20knob_thro.model', 'XGBoost_20knob_prec.model'], knob_path=r'milvus_important_params.json') -> None:
        self.model_path = model_path
        self.get_surrogate(model_path)
        self.knob_stand = KnobStand(knob_path)
        self.names = list(self.knob_stand.knobs_detail.keys())
        self.t1 = time.time()
        self.sampled_times = 0

        self.X_record = []
        self.Y1_record = []
        self.Y2_record = []
        self.Y_record = []

    def get_surrogate(self, surrogate_path):
        # surrogate1, surrogate2 = joblib.load(surrogate_path[0]), joblib.load(surrogate_path[1])
        self.model1, self.model2 = joblib.load(surrogate_path[0]), joblib.load(surrogate_path[1])

    def get_state(self, knob_vals_arr):
        Y1, Y2 = [], []
        for i,record in enumerate(knob_vals_arr):
            conf_value = [self.knob_stand.scale_back(self.names[j], knob_val)[0] for j,knob_val in enumerate(record)]
            print(f"Index parameters changed: {conf_value}")

            y1 = self.model1.predict([conf_value])[0]
            y2 = self.model2.predict([conf_value])[0]

            self.sampled_times += 1
            print(f'[{self.sampled_times}] {int(time.time()-self.t1)} {y1} {y2}')
            
            Y1.append(y1)
            Y2.append(y2)
        return np.concatenate((np.array(Y1).reshape(-1,1), np.array(Y2).reshape(-1,1)), axis=1)

class RealEnv:
    def __init__(self, dataset_name, bench_path=RUN_ENGINE_PATH, knob_path=KNOB_PATH) -> None:
        self.dataset_name = dataset_name
        self.bench_path = bench_path
        self.knob_stand = KnobStand(knob_path)
        self.names = list(self.knob_stand.knobs_detail.keys())
        self.t1 = time.time()
        self.t2 = time.time()
        self.sampled_times = 0

        self.X_record = []
        self.Y1_record = []
        self.Y2_record = []
        self.Y_record = []
        

    def get_state(self, conf_value):
        Y1, Y2, Y3 = [], [], []
        index_value, system_value = conf_value[:9], conf_value[9:]
        index_name, system_name = self.names[:9], self.names[9:]

        index_conf = dict(zip(index_name,index_value))
        system_conf = dict(zip(system_name,system_value))

        configure_index(*filter_index_rule(index_conf))
        configure_system(filter_system_rule(system_conf))

        # print(f"Parameters changed to: {index_conf} {system_conf}")

        try:
            result = sp.run(f'{RUN_ENGINE_PATH} "" "" {self.dataset_name}', env=env, shell=True, stdout=sp.PIPE)
            result = result.stdout.decode().split()
            y1, y2 = float(result[-2]), float(result[-3]) 
            
            self.Y1_record.append(y1)
            self.Y2_record.append(y2)
        except Exception as e:
            print(e)
            y1, y2 = min(self.Y1_record), min(self.Y2_record)
        
        y3 = int(time.time()-self.t2)
        self.sampled_times += 1

        self.t2 = time.time()
        print(f'[{self.sampled_times}] {int(self.t2-self.t1)} {y1} {y2} {y3}')
        sp.run(f'echo [{self.sampled_times}] {int(self.t2-self.t1)} {index_conf} {system_conf} {y1} {y2} {y3} >> record.log', shell=True, stdout=sp.PIPE)

        Y1.append(y1)
        Y2.append(y2)
        Y3.append(y3)

        return np.array([Y1,Y2,Y3]).T

    def default_conf(self):
        return [self.knob_stand.scale_forward(k, v['default']) for k,v in self.knob_stand.knobs_detail.items()]





@problem_registry.register("CSSTuning_VDBMS")
class VDBMSTuning(NonTabularProblem):
    problem_type = 'vdbms'
    workloads = ['glove-25-angular', 'glove-100-angular', 'deep-image-96-angular', 'gist-960-euclidean', 'gist-960-angula', 'laion-small-clip', 'dbpedia-openai-1M-1536-angular', 'dbpedia-openai-100K-1536-angular']
    num_variables = 16
    num_objectives = 1
    fidelity = None
    def __init__(self, task_name, budget_type, budget, seed, workload, description, knobs=None, **kwargs):
        self.dataset_name = VDBMSTuning.workloads[workload]
        self.real_env = RealEnv(dataset_name=self.dataset_name)
        super().__init__(task_name=task_name, budget_type=budget_type, budget=budget, seed=seed, workload=workload, description=description)
        self.data_info = get_datesets_info()
        for d in self.data_info:
            if d['name'] == self.dataset_name:
                self.dataset_dimension = d['vector_size']
        
    def get_configuration_space(self):
        variables = []
        for knob_name in self.real_env.names:
            knob_details = self.real_env.knob_stand.knobs_detail[knob_name]
            knob_type = knob_details["type"]
            if knob_type == "enum":
                categories = knob_details["enum_values"]
                variables.append(Categorical(knob_name, categories))
            elif knob_type == "integer":
                range_ = (knob_details["min"], knob_details["max"])
                variables.append(Integer(knob_name, range_))
        
        ss = SearchSpace(variables)

        return ss

    def get_fidelity_space(self):
        return FidelitySpace([])
    
    def get_objectives(self):
        return {
            "f1": "minimize",
            # "f2": "minimize",
            # "f3": "minimize",
        }
    
    def get_problem_type(self):
        return self.problem_type
    
    def objective_function(self, configuration: dict, fidelity=None, seed=None, **kwargs):
        X = []
        # X = [v for k, v in configuration.items()]
        for k, v in configuration.items():
            if k == 'm':
                m_val = int(v)
                # If dataset_dimension cannot be divided evenly by m_val, pick the nearest divisible m
                if self.dataset_dimension % m_val != 0:
                    # Try to find the nearest m (both smaller and larger) that divides dataset_dimension
                    candidates = []
                    # Search downwards
                    lower = m_val - 1
                    while lower > 0:
                        if self.dataset_dimension % lower == 0:
                            candidates.append(lower)
                            break
                        lower -= 1
                    # Search upwards
                    upper = m_val + 1
                    while upper <= self.dataset_dimension:
                        if self.dataset_dimension % upper == 0:
                            candidates.append(upper)
                            break
                        upper += 1
                    # Choose the closest m value
                    if candidates:
                        best_m = min(candidates, key=lambda x: abs(x - m_val))
                        m_val = best_m
                X.append(m_val)
            else:
                X.append(v)
        try:
            results = self.real_env.get_state(X)
        except Exception as e:
            print(e)
            return {'f1': 1000000}
        
        return {'f1': float(results[0][1])}

if __name__ == '__main__':
    vdbms = VDBMSTuning(task_name='vdbms', budget_type='time', budget=1000, seed=0, workload=0, description='vdbms')
    vdbms.objective_function(configuration={})