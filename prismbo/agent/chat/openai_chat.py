import json
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import yaml
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from prismbo.agent.config import Configer
from prismbo.agent.registry import *
from prismbo.benchmark.instantiate_problems import InstantiateProblems
from prismbo.datamanager.manager import DataManager
from prismbo.optimizer.construct_optimizer import ConstructOptimizer
from prismbo.utils.log import logger


def dict_to_string(dictionary):
    return json.dumps(dictionary, ensure_ascii=False, indent=4)


class Message(BaseModel):
    """Model for LLM messages"""

    role: str  # The role of the message author (system, user, assistant, or function).
    content: Optional[Union[str, List[Dict]]] = None  # The message content.
    tool_call_id: Optional[str] = None  # ID for the tool call response
    name: Optional[str] = None  # Name of the tool or function, if applicable
    metrics: Dict[str, Any] = {}  # Metrics for the message.
    

    def get_content_string(self) -> str:
        """Returns the content as a string."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return json.dumps(self.content)
        return ""

    def to_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude_none=True, exclude={"metrics"})
        # Manually add the content field if it is None
        if self.content is None:
            _dict["content"] = None
        return _dict

    def log(self, level: Optional[str] = None):
        """Log the message to the console."""
        _logger = getattr(logger, level or "debug")
        
        _logger(f"============== {self.role} ==============")
        message_detail = f"Content: {self.get_content_string()}"
        if self.tool_call_id:
            message_detail += f", Tool Call ID: {self.tool_call_id}"
        if self.name:
            message_detail += f", Name: {self.name}"
        _logger(message_detail)


class OpenAIChat:
    history: List[Message]

    def __init__(
        self,
        api_key,
        model="gpt-3.5-turbo",
        base_url="https://aihubmix.com/v1",
        client_kwargs: Optional[Dict[str, Any]] = None,
        data_manager: Optional[DataManager] = None,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key 
        self.client_kwargs = client_kwargs or {}

        self.prompt = self._get_prompt()
        self.is_first_msg = True
        
        self.history = []

        self.data_manager = DataManager() if data_manager is None else data_manager
        self.running_config = Configer()
        
        # 初始化函数调用计数器
        self.function_call_counts = {
            "get_all_datasets": 0,
            "get_dataset_info": 0,
            "get_all_problems": 0,
            "get_optimization_techniques": 0,
            "set_optimization_problem": 0,
            "set_space_refiner": 0,
            "set_sampler": 0,
            "set_pretrain": 0,
            "set_model": 0,
            "set_normalizer": 0,
            "set_metadata": 0,
            "run_optimization": 0,
            "show_configuration": 0,
            "install_package": 0,
        }
        self._initialize_modules()

    def _get_prompt(self):
        """Reads a prompt from a file."""
        current_dir = Path(__file__).parent
        file_path = current_dir / "prompt"
        with open(file_path, "r") as file:
            return file.read()


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

    @property
    def client(self):
        """Lazy initialization of the OpenAI client."""
        from openai import OpenAI
        return OpenAI(
            api_key=self.api_key, base_url=self.base_url,
            **self.client_kwargs
        )

    def invoke_model(self, messages: List[Dict]) -> ChatCompletion:
        self.history.extend(messages)
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_all_datasets",
                    "description": "Show all available datasets in our system",
                    "parameters": {},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_dataset_info",
                    "description": "Show detailed information of dataset according to the dataset name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dataset_name": {
                                "type": "string",
                                "description": "The name of the dataset",
                            },
                        },
                        "required": ["dataset_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_all_problems",
                    "description": "Show all optimization problems that our system supoorts",
                    "parameters": {},
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "get_optimization_techniques",
                    "description": "Show all optimization techniques supported in  our system,",
                    "parameters": {},
                },
            },
                        
            {
                "type": "function",
                "function": {
                    "name": "set_optimization_problem",
                    "description": "Define or set an optimization problem based on user inputs for 'problem name', 'workload' and 'budget'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "problem_name": {
                                "type": "string",
                                "description": "The name of the optimization problem",
                            },
                            "workload": {
                                "type": "integer",
                                "description": "The number of workload",
                            },
                            "budget": {
                                "type": "integer",
                                "description": "The number of budget to do function evaluations",
                            },
                        },
                        "required": ["problem_name", "workload", "budget"],
                    },
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "set_model",
                    "description": "Set the model used as surrogate model in the  Bayesian optimization, The input model name should be one of the available models.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Model": {
                                "type": "string",
                                "description": "The model name",
                            },
                        },
                        "required": ["Model"],
                    },
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "set_sampler",
                    "description": "Set the sampler for the optimization process as user input. The input sampler name should be one of the available samplers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Sampler": {
                                "type": "string",
                                "description": "The name of Sampler",
                            },
                        },
                        "required": ["Sampler"],
                    },
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "set_pretrain",
                    "description": "Set the Pretrain methods. The input of users should include one of the available pretrain methods.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Pretrain": {
                                "type": "string",
                                "description": "The name of Pretrain method",
                            },
                        },
                        "required": ["Pretrain"],
                    },
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "set_normalizer",
                    "description": "Set the normalization method to nomalize function evaluation and parameters. It requires one of the available normalization methods as input.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Normalizer": {
                                "type": "string",
                                "description": "The name of Normalization method",
                            },
                        },
                        "required": ["Normalizer"],
                    },
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "set_metadata",
                    "description": "Set the metadata using a dataset stored in our system and specify a module to utilize this metadata.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Normalizer": {
                                "type": "string",
                                "description": "The name of Normalization method",
                            },
                        },
                        "required": ["module_name", "dataset_name"],
                    },
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "run_optimization",
                    "description": "Set the normalization method to nomalize function evaluation and parameters. It requires one of the available normalization methods as input.",
                    "parameters": {},
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "show_configuration",
                    "description": "Display all configurations set by the user so far, including the optimizer configuration, metadata configuration, and optimization problems",
                    "parameters": {},
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "install_package",
                    "description": "Install a Python package using pip",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "package_name": {
                                "type": "string",
                                "description": "The name of the package to install",
                            },
                        },
                        "required": ["package_name"],
                    },
                },
            },      
        ]
                
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        # Process tool calls if there are any
        if tool_calls:
            self.history.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                function_response = self.call_manager_function(function_name, **function_args)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                }
                self.history.append(tool_message)
                
            # Refresh the model with the function response and get a new response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
            )
        
        self.history.append(response.choices[0].message) 
        logger.debug(f"Response: {response.choices[0].message.content}")
        return response

    def get_response(self, user_input) -> str:
        logger.debug("---------- OpenAI Response Start ----------")
        user_message = {"role": "user", "content": user_input}
        logger.debug(f"User: {user_input}")
        messages = [user_message]

        if self.is_first_msg:
            system_message = {"role": "system", "content": self.prompt}
            messages.insert(0, system_message)
            self.is_first_msg = False
        else:
            system_message = {"role": "system", "content": "Don't tell me which function to use, just call it. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous"}
            messages.insert(0, system_message)
            

        response = self.invoke_model(messages)
        logger.debug(f"Assistant: {response.choices[0].message.content}")
        logger.debug("---------- OpenAI Response End ----------")
        return response.choices[0].message.content 
    
    def call_manager_function(self, function_name, **kwargs):
        # 增加函数调用计数
        if function_name in self.function_call_counts:
            self.function_call_counts[function_name] += 1
            logger.debug(f"Function {function_name} called. Count: {self.function_call_counts[function_name]}")
        
        available_functions = {
            "get_all_datasets": self.data_manager.get_all_datasets,
            "get_all_problems": self.get_all_problems,
            "get_optimization_techniques": self.get_optimization_techniques,
            "get_dataset_info": lambda: self.data_manager.get_dataset_info(kwargs['dataset_name']),
            "set_optimization_problem": lambda: self.set_optimization_problem(kwargs['problem_name'], kwargs['workload'], kwargs['budget']),
            'set_space_refiner': lambda: self.set_space_refiner(kwargs['refiner']),
            'set_sampler': lambda: self.set_sampler(kwargs['Sampler']),
            'set_pretrain': lambda: self.set_pretrain(kwargs['Pretrain']),
            'set_model': lambda: self.set_model(kwargs['Model']),
            'set_normalizer': lambda: self.set_normalizer(kwargs['Normalizer']),
            'set_metadata': lambda: self.set_metadata(kwargs['module_name'], kwargs['dataset_name']),
            'run_optimization': self.run_optimization,
            'show_configuration': self.show_configuration,
            "install_package": lambda: self.install_package(kwargs['package_name']),
        }
        function_to_call = available_functions[function_name]
        return json.dumps({"result": function_to_call()})
    
    def _initialize_modules(self):
        import prismbo.benchmark.synthetic
        import prismbo.optimizer.acquisition_function
        import prismbo.optimizer.model
        import prismbo.optimizer.pretrain
        import prismbo.optimizer.refiner
        import prismbo.optimizer.initialization

    def get_all_problems(self):
        tasks_info = []

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
                task_info = {
                    "name": name,
                    "problem_type": "synthetic",
                    "anyDim": False,
                    "num_vars": [num_var],
                    "num_objs": [num_obj],
                    "workloads": [workloads],
                    "fidelity": [fidelity],
                }
            tasks_info.append(task_info)
        return tasks_info
    
    def get_optimization_techniques(self):
        basic_info = {}

        selector_info = []
        model_info = []
        sampler_info = []
        acf_info = []
        pretrain_info = []
        refiner_info = []
        normalizer_info = []
        
        # tasks information
        sampler_names = sampler_registry.list_names()
        for name in sampler_names:
            sampler_info.append(name)
        basic_info["Sampler"] = ','.join(sampler_info)

        refiner_names = space_refiner_registry.list_names()
        for name in refiner_names:
            refiner_info.append(name)
        basic_info["SpaceRefiner"] = ','.join(refiner_info)

        pretrain_names = pretrain_registry.list_names()
        for name in pretrain_names:
            pretrain_info.append(name)
        basic_info["Pretrain"] = ','.join(pretrain_info)

        model_names = model_registry.list_names()
        for name in model_names:
            model_info.append(name)
        basic_info["Model"] = ','.join(model_info)

        acf_names = acf_registry.list_names()
        for name in acf_names:
            acf_info.append(name)
        basic_info["ACF"] = ','.join(acf_info)

        selector_names = selector_registry.list_names()
        for name in selector_names:
            selector_info.append(name)
        basic_info["DataSelector"] = ','.join(selector_info)
        
        normalizer_names = selector_registry.list_names()
        for name in normalizer_names:
            normalizer_info.append(name)
        basic_info["Normalizer"] = ','.join(normalizer_info)
        
        
        return basic_info
    
    def set_optimization_problem(self, problem_name, workload, budget):        
        problem_info = {}
        if problem_name in problem_registry:
            problem_info[problem_name] = {
                'budget': budget,
                'workload': workload,
                'budget_type': 'Num_FEs',
                "params": {},
            }

        self.running_config.set_tasks(problem_info)
        return "Succeed"
    
    def set_space_refiner(self, refiner):
        self.running_config.optimizer['SpaceRefiner'] = refiner
        return f"Succeed to set the space refiner {refiner}"

    def set_sampler(self, Sampler):
        self.running_config.optimizer['Sampler'] = Sampler
        return f"Succeed to set the sampler {Sampler}"
    
    
    def set_pretrain(self, Pretrain):
        self.running_config.optimizer['Pretrain'] = Pretrain
        return f"Succeed to set the pretrain {Pretrain}"
    
    def set_model(self, Model):
        self.running_config.optimizer['Model'] = Model
        return f"Succeed to set the model {Model}"
    
    def set_normalizer(self, Normalizer):
        self.running_config.optimizer['Normalizer'] = Normalizer
        return f"Succeed to set the normalizer {Normalizer}"
    
    def set_metadata(self, module_name, dataset_name):
        self.running_config.metadata[module_name] = dataset_name
        return f"Succeed to set the metadata {dataset_name} for {module_name}"
    
    def run_optimization(self):
        task_set = InstantiateProblems(self.running_config.tasks, 0)
        optimizer = ConstructOptimizer(self.running_config.optimizer, 0)
        
        try:
            while (task_set.get_unsolved_num()):
                iteration = 0
                search_space = task_set.get_cur_searchspace()
                dataset_info, dataset_name = self.construct_dataset_info(task_set, self.running_config, seed=0)
                
                self.data_manager.db.create_table(dataset_name, dataset_info, overwrite=True)
                optimizer.link_task(task_name=task_set.get_curname(), search_sapce=search_space)
                
                metadata, metadata_info = self.get_metadata('SpaceRefiner')
                optimizer.search_space_refine(metadata, metadata_info)
                
                metadata, metadata_info = self.get_metadata('Sampler')
                samples = optimizer.sample_initial_set(metadata, metadata_info)
                
                parameters = [search_space.map_to_design_space(sample) for sample in samples]
                observations = task_set.f(parameters)
                self.save_data(dataset_name, parameters, observations, iteration)
                
                optimizer.observe(samples, observations)
                
                #Pretrain
                metadata, metadata_info = self.get_metadata('Model')
                optimizer.meta_fit(metadata, metadata_info)
        
                while (task_set.get_rest_budget()):
                    optimizer.fit()
                    suggested_samples = optimizer.suggest()
                    parameters = [search_space.map_to_design_space(sample) for sample in suggested_samples]
                    observations = task_set.f(parameters)
                    self.save_data(dataset_name, parameters, observations, iteration)
                    
                    optimizer.observe(suggested_samples, observations)
                    iteration += 1
                    
                    print("Seed: ", 0, "Task: ", task_set.get_curname(), "Iteration: ", iteration)
                    # if self.verbose:
                    #     self.visualization(testsuits, suggested_sample)
                task_set.roll()
        except Exception as e:
            raise e
    def show_configuration(self):
        conf = {'Optimization problem': self.running_config.tasks, 'Optimizer': self.running_config.optimizer, 'Metadata': self.running_config.metadata}
        return dict_to_string(conf)
    
    def install_package(self, package_name: str) -> str:
        """Install a Python package using pip."""
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return f"Package '{package_name}' installed successfully."
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install package '{package_name}': {e}")
            return f"Failed to install package '{package_name}'. Error: {str(e)}"
    
    def reset_function_call_counts(self):
        """重置所有函数调用计数器"""
        for function_name in self.function_call_counts:
            self.function_call_counts[function_name] = 0
        logger.debug("Function call counts reset to zero")
    
    def get_function_call_counts(self) -> Dict[str, int]:
        """获取当前函数调用计数"""
        return self.function_call_counts.copy()
        

@dataclass
class TestResult:
    """测试结果数据类"""
    test_id: str
    input_text: str
    expected_function: Optional[str]
    response: str
    response_time: float
    success: bool
    error_message: Optional[str] = None
    function_called: Optional[str] = None
    user_rating: Optional[int] = None

class ChatbotTester:
    """Chatbot测试器"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = "https://aihubmix.com/v1", case_number: int = 1):
        self.chat = OpenAIChat(api_key=api_key, model=model, base_url=base_url)
        self.results: List[TestResult] = []
        self.case_number = case_number
        
        # 重置函数调用计数器
        self.chat.reset_function_call_counts()
        
        # 定义测试用例
        self.test_cases = self._define_test_cases()
        
    def _define_test_cases(self) -> List[Dict]:
        """定义测试用例"""
        # 导入测试用例生成函数
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'EXP 3'))
        from test_prompt import generate_test_cases
        
        cases = generate_test_cases(self.case_number)
        
        return cases
    
    def _extract_function_calls(self, response: str) -> List[str]:
        """从函数调用计数器中提取被调用的函数"""
        called_functions = []
        current_counts = self.chat.get_function_call_counts()
        
        # 检查哪些函数的计数大于0，表示被调用了
        for function_name, count in current_counts.items():
            if count > 0:
                called_functions.append(function_name)
        
        logger.debug(f"Detected function calls: {called_functions}")
        return called_functions
    
    def _evaluate_success(self, test_case: Dict, response: str, function_called: List[str]) -> bool:
        """评估测试是否成功"""
        expected = test_case.get("expected_function")
        if expected is None:
            # 对于模糊输入，期望系统要求澄清
            return "clarification" in response.lower() or "please clarify" in response.lower()
        
        if isinstance(expected, list):
            # 复合问题，检查是否调用了所有期望的函数
            return all(func in function_called for func in expected)
        else:
            # 单一函数调用
            return expected in function_called
    
    def run_single_test(self, test_case: Dict) -> TestResult:
        """运行单个测试"""
        test_id = test_case["id"]
        input_text = test_case["input"]
        expected_function = test_case.get("expected_function")
        
        logger.info(f"Running test: {test_id}")
        logger.info(f"Input: {input_text}")
        
        # 重置函数调用计数器
        self.chat.reset_function_call_counts()
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 获取响应
            response = self.chat.get_response(input_text)
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            # 提取函数调用
            function_called = self._extract_function_calls(response)
            
            # 评估成功
            success = self._evaluate_success(test_case, response, function_called)
            
            result = TestResult(
                test_id=test_id,
                input_text=input_text,
                expected_function=expected_function,
                response=response,
                response_time=response_time,
                success=success,
                function_called=function_called[0] if function_called else None
            )
            
            logger.info(f"Test {test_id} completed. Success: {success}, Time: {response_time:.2f}s")
            
        except Exception as e:
            response_time = time.time() - start_time
            result = TestResult(
                test_id=test_id,
                input_text=input_text,
                expected_function=expected_function,
                response="",
                response_time=response_time,
                success=False,
                error_message=str(e)
            )
            logger.error(f"Test {test_id} failed with error: {e}")
        
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("Starting comprehensive chatbot testing...")
        
        for test_case in self.test_cases:
            result = self.run_single_test(test_case)
            self.results.append(result)
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # 计算平均响应时间
        avg_response_time = sum(r.response_time for r in self.results) / total_tests if total_tests > 0 else 0
        
        # 按类别分组统计
        category_stats = {}
        for result in self.results:
            test_case = next(tc for tc in self.test_cases if tc["id"] == result.test_id)
            category = test_case["category"]
            if category not in category_stats:
                category_stats[category] = {"total": 0, "success": 0}
            category_stats[category]["total"] += 1
            if result.success:
                category_stats[category]["success"] += 1
        
        # 按输入类型分组统计
        input_type_stats = {}
        for result in self.results:
            test_case = next(tc for tc in self.test_cases if tc["id"] == result.test_id)
            input_type = test_case["input_type"]
            if input_type not in input_type_stats:
                input_type_stats[input_type] = {"total": 0, "success": 0}
            input_type_stats[input_type]["total"] += 1
            if result.success:
                input_type_stats[input_type]["success"] += 1
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate,
                "avg_response_time": avg_response_time
            },
            "category_performance": {
                category: {
                    "success_rate": stats["success"] / stats["total"],
                    "total": stats["total"],
                    "success": stats["success"]
                }
                for category, stats in category_stats.items()
            },
            "input_type_performance": {
                input_type: {
                    "success_rate": stats["success"] / stats["total"],
                    "total": stats["total"],
                    "success": stats["success"]
                }
                for input_type, stats in input_type_stats.items()
            },
            "detailed_results": [
                {
                    "test_id": r.test_id,
                    "input": r.input_text,
                    "expected_function": r.expected_function,
                    "function_called": r.function_called,
                    "success": r.success,
                    "response_time": r.response_time,
                    "error": r.error_message
                }
                for r in self.results
            ]
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """打印测试报告"""
        print("\n" + "="*60)
        print("CHATBOT COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        summary = report["summary"]
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Successful: {summary['successful_tests']}")
        print(f"   Success Rate: {summary['success_rate']:.2%}")
        print(f"   Average Response Time: {summary['avg_response_time']:.2f}s")
        
        print(f"\n📈 CATEGORY PERFORMANCE:")
        for category, stats in report["category_performance"].items():
            print(f"   {category}: {stats['success_rate']:.2%} ({stats['success']}/{stats['total']})")
        
        print(f"\n🎯 INPUT TYPE PERFORMANCE:")
        for input_type, stats in report["input_type_performance"].items():
            print(f"   {input_type}: {stats['success_rate']:.2%} ({stats['success']}/{stats['total']})")
        
        print(f"\n📋 DETAILED RESULTS:")
        for result in report["detailed_results"]:
            status = "✅" if result["success"] else "❌"
            print(f"   {status} {result['test_id']}: {result['input'][:50]}...")
            if not result["success"]:
                print(f"      Expected: {result['expected_function']}, Called: {result['function_called']}")
        
        print("\n" + "="*60)

def test_openai_chat(case_number: int = 1):
    """运行指定编号的chatbot测试"""
    # 使用测试API密钥
    api_key = "sk-RkYVrUuk7H05cHtO264f5b155b1b41FdB6D0C3C710704e9f"
    
    # 创建测试器
    tester = ChatbotTester(api_key=api_key, model="gpt-4o-mini", base_url="https://aihubmix.com/v1", case_number=case_number)
    
    # 运行所有测试
    report = tester.run_all_tests()
    
    # 打印报告
    tester.print_report(report)
    
    # 保存报告到文件
    report_file = f"chatbot_test_report_case_{case_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return report

def test_all_cases():
    """运行所有20个测试用例"""
    # 使用测试API密钥
    api_key = "sk-RkYVrUuk7H05cHtO264f5b155b1b41FdB6D0C3C710704e9f"
    
    all_reports = {}
    case_report_files = {}  # 存储每个case的报告文件路径
    overall_stats = {
        "total_tests": 0,
        "total_successful": 0,
        "total_failed": 0,
        "avg_response_time": 0,
        "case_performance": {}
    }
    
    print("🚀 Starting comprehensive testing of all 20 test cases...")
    print("="*80)
    
    for case_number in range(1, 21):
        print(f"\n📋 Running Test Case {case_number}/20...")
        
        try:
            # 创建测试器
            tester = ChatbotTester(api_key=api_key, model="gpt-3.5-turbo", base_url="https://aihubmix.com/v1", case_number=case_number)
            
            # 运行测试
            report = tester.run_all_tests()
            all_reports[f"case_{case_number}"] = report
            
            # 单独保存每个case的报告
            case_report_file = f"test_case_{case_number}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(case_report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            case_report_files[f"case_{case_number}"] = case_report_file
            print(f"📄 Case {case_number} report saved to: {case_report_file}")
            
            # 更新总体统计
            summary = report["summary"]
            overall_stats["total_tests"] += summary["total_tests"]
            overall_stats["total_successful"] += summary["successful_tests"]
            overall_stats["total_failed"] += (summary["total_tests"] - summary["successful_tests"])
            overall_stats["avg_response_time"] += summary["avg_response_time"]
            
            # 记录每个case的性能
            overall_stats["case_performance"][f"case_{case_number}"] = {
                "success_rate": summary["success_rate"],
                "avg_response_time": summary["avg_response_time"],
                "total_tests": summary["total_tests"],
                "successful_tests": summary["successful_tests"]
            }
            
            print(f"✅ Case {case_number} completed - Success Rate: {summary['success_rate']:.2%}")
            
        except Exception as e:
            print(f"❌ Case {case_number} failed with error: {e}")
            
            # 即使失败也保存错误报告
            error_report = {
                "timestamp": datetime.now().isoformat(),
                "case_number": case_number,
                "error": str(e),
                "summary": {
                    "total_tests": 0,
                    "successful_tests": 0,
                    "success_rate": 0.0,
                    "avg_response_time": 0.0
                }
            }
            case_report_file = f"test_case_{case_number}_error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(case_report_file, 'w', encoding='utf-8') as f:
                json.dump(error_report, f, indent=2, ensure_ascii=False)
            case_report_files[f"case_{case_number}"] = case_report_file
            print(f"📄 Case {case_number} error report saved to: {case_report_file}")
            
            overall_stats["case_performance"][f"case_{case_number}"] = {
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "total_tests": 0,
                "successful_tests": 0,
                "error": str(e)
            }
    
    # 计算总体平均响应时间
    if overall_stats["total_tests"] > 0:
        overall_stats["avg_response_time"] /= 20  # 20个case
        overall_success_rate = overall_stats["total_successful"] / overall_stats["total_tests"]
    else:
        overall_success_rate = 0.0
    
    # 打印总体报告
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE TESTING SUMMARY")
    print("="*80)
    print(f"📊 Overall Performance:")
    print(f"   Total Tests Across All Cases: {overall_stats['total_tests']}")
    print(f"   Total Successful: {overall_stats['total_successful']}")
    print(f"   Total Failed: {overall_stats['total_failed']}")
    print(f"   Overall Success Rate: {overall_success_rate:.2%}")
    print(f"   Average Response Time: {overall_stats['avg_response_time']:.2f}s")
    
    print(f"\n📈 Case-by-Case Performance:")
    for case_num in range(1, 21):
        case_key = f"case_{case_num}"
        if case_key in overall_stats["case_performance"]:
            perf = overall_stats["case_performance"][case_key]
            status = "✅" if perf["success_rate"] >= 0.8 else "⚠️" if perf["success_rate"] >= 0.5 else "❌"
            print(f"   {status} Case {case_num:2d}: {perf['success_rate']:.2%} ({perf['successful_tests']}/{perf['total_tests']}) - {perf['avg_response_time']:.2f}s")
    
    # 保存总体报告
    comprehensive_report = {
        "timestamp": datetime.now().isoformat(),
        "overall_stats": overall_stats,
        "all_reports": all_reports,
        "case_report_files": case_report_files
    }
    
    report_file = f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Comprehensive report saved to: {report_file}")
    
    return comprehensive_report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run chatbot tests')
    parser.add_argument('--case', type=int, default=1, help='Test case number (1-20)')
    parser.add_argument('--all', action='store_true', help='Run all 20 test cases')
    
    args = parser.parse_args()
    
    print("Running all 20 test cases...")
    test_all_cases()
