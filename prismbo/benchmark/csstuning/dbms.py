import numpy as np
from csstuning.dbms.dbms_benchmark import MySQLBenchmark

from prismbo.agent.registry import problem_registry
from prismbo.benchmark.problem_base.non_tab_problem import NonTabularProblem
from prismbo.space.fidelity_space import FidelitySpace
from prismbo.space.search_space import SearchSpace
from prismbo.space.variable import *


@problem_registry.register("CSSTuning_MySQL")
class MySQLTuning(NonTabularProblem):
    problem_type = 'dbms'
    workloads = MySQLBenchmark.AVAILABLE_WORKLOADS
    num_variables = 5
    num_objectives = 1
    fidelity = None
    
    def __init__(self, task_name, budget_type, budget, seed, workload, description, knobs=None, **kwargs):        
        self.workload = MySQLBenchmark.AVAILABLE_WORKLOADS[workload]

        self.benchmark = MySQLBenchmark(workload=self.workload)
        self.knobs = self.benchmark.get_config_space()
        self.num_variables = len(self.knobs)
        
        super().__init__(task_name=task_name, budget_type=budget_type, budget=budget, workload=workload, description=description, seed=seed)
        np.random.seed(seed)


    def get_configuration_space(self):
        variables = []
        tuning_knobs = {
            "innodb_buffer_pool_size",
            "innodb_flush_log_at_trx_commit",
            "innodb_flush_neighbors",
            "innodb_doublewrite",
            "innodb_io_capacity"
        }
        for knob_name in tuning_knobs:
            knob_details = self.knobs[knob_name]
            knob_type = knob_details["type"]
            range_ = knob_details["range"]
            
            if knob_type == "enum":
                variables.append(Categorical(knob_name, range_))
            elif knob_type == "integer":
                if range_[1] > np.iinfo(np.int64).max:
                    variables.append(ExponentialInteger(knob_name, range_))
                else:
                    variables.append(Integer(knob_name, range_))
            else:
                variables.append(Continuous(knob_name, range_))


        return SearchSpace(variables)
    
    def get_fidelity_space(self):
        return FidelitySpace([])
    
    def get_objectives(self) -> dict:
        return {
            # "latency": "minimize",
            "f1": "minimize",
        }
        
    def get_problem_type(self):
        return self.problem_type
        
    def objective_function(self, configuration: dict, fidelity = None, seed = None, **kwargs):
        try:
            perf = self.benchmark.run(configuration)
            return {'f1': -perf.get('throughput', 0)}
        except Exception as e:
            return {'f1': 0}

if __name__ == "__main__":
    a = MySQLTuning('1', 'Num_FEs', 121, 0, 'sibench', knobs=None)
    print(a.f({}))
