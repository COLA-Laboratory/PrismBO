import numpy as np
from csstuning.compiler.compiler_benchmark import GCCBenchmark, LLVMBenchmark

from prismbo.agent.registry import problem_registry
from prismbo.benchmark.problem_base.non_tab_problem import NonTabularProblem
from prismbo.space.fidelity_space import FidelitySpace
from prismbo.space.search_space import SearchSpace
from prismbo.space.variable import *


@problem_registry.register("CSSTuning_GCC")
class GCCTuning(NonTabularProblem):
    problem_type = 'compiler'
    workloads = GCCBenchmark.AVAILABLE_WORKLOADS
    num_variables = 5
    num_objectives = 1
    fidelity = None
    
    def __init__(self, task_name, budget_type, budget, seed, workload, description, knobs=None, **kwargs):        
        self.workload = GCCBenchmark.AVAILABLE_WORKLOADS[workload]
        self.benchmark = GCCBenchmark(workload=self.workload)
        
        all_knobs = self.benchmark.get_config_space()
        self.knobs = {k: all_knobs[k] for k in (knobs or all_knobs)}
        self.num_variables = len(self.knobs)
        
        super().__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
            description=description,
        )
        np.random.seed(seed)

    def get_configuration_space(self):
        variables = []
        tuning_knobs = {
            "ipa-cp", "ipa-icf", "devirtualize", "inline-functions-called-once", "tree-pta"
        }
        for knob_name in tuning_knobs:
            knob_details = self.knobs[knob_name]
            knob_type = knob_details["type"]
            range_ = knob_details["range"]
            
            if knob_type == "enum":
                variables.append(Categorical(knob_name, range_))
            elif knob_type == "integer":
                variables.append(Integer(knob_name, range_))

        return SearchSpace(variables)
    
    def get_fidelity_space(self):
        return FidelitySpace([])
    
    def get_objectives(self) -> dict:
        return {
            
            "execution_time": "minimize",
            # "compilation_time": "minimize",
            # "file_size": "minimize",
            # "maxrss": "minimize",
            # "PAPI_TOT_CYC": "minimize",
            # "PAPI_TOT_INS": "minimize",
            # "PAPI_BR_MSP": "minimize",
            # "PAPI_BR_PRC": "minimize",
            # "PAPI_BR_CN": "minimize",
            # "PAPI_MEM_WCY": "minimize",
        }
    
    def get_problem_type(self):
        return self.problem_type
    
    def objective_function(self, configuration: dict, fidelity = None, seed = None, **kwargs):        
        try:
            perf = self.benchmark.run(configuration)
            return {'f1': perf.get('execution_time', 1e10)}
        
        except Exception as e:
            return {'f1': 1e10}
        


@problem_registry.register("CSSTuning_LLVM")
class LLVMTuning(NonTabularProblem):
    problem_type = 'compiler'
    workloads = LLVMBenchmark.AVAILABLE_WORKLOADS
    num_variables = 5
    num_objectives = 1
    fidelity = None
    
    def __init__(self, task_name, budget_type, budget, seed, workload, description, knobs=None, **kwargs):
        self.workload = LLVMBenchmark.AVAILABLE_WORKLOADS[workload]
        self.benchmark = LLVMBenchmark(workload=self.workload)
        
        all_knobs = self.benchmark.get_config_space()
        self.knobs = {k: all_knobs[k] for k in (knobs or all_knobs)}
        self.num_variables = len(self.knobs)
        
        super().__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
            description=description,
        )
        np.random.seed(seed)

    def get_configuration_space(self):
        variables = []
        tuning_knobs = {
            "inline", "loop-rotate", "gvn", "dse", "licm"
        }
        for knob_name in tuning_knobs:
            knob_details = self.knobs[knob_name]
            knob_type = knob_details["type"]
            range_ = knob_details["range"]
            
            if knob_type == "enum":
                variables.append(Categorical(knob_name, range_))
            elif knob_type == "integer":
                variables.append(Integer(knob_name, range_))

        return SearchSpace(variables)
    
    def get_fidelity_space(self):
        return FidelitySpace([])
    
    def get_objectives(self) -> dict:
        return {
            "execution_time": "minimize",
            # "compilation_time": "minimize",
            # "file_size": "minimize",
        }
    
    def get_problem_type(self):
        return self.problem_type
    
    def objective_function(self, configuration: dict, fidelity = None, seed = None, **kwargs): 
        try:
            perf = self.benchmark.run(configuration)
            return {'f1': perf.get('execution_time', 1e10)}
        except Exception as e:
            return {'f1': 1e10}


if __name__ == "__main__":
    benchmark = GCCBenchmark(workload="cbench-automotive-bitcount")
    conf = {
        
    }
    llvmbenchmark = LLVMBenchmark(workload="cbench-automotive-bitcount")

    llvmbenchmark.run(conf)

