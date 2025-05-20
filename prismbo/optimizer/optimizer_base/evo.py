import abc
import numpy as np
import math
from typing import Union, Dict, List
from prismbo.optimizer.acquisition_function.sequential import Sequential
from prismbo.optimizer.optimizer_base.base import OptimizerBase
from prismbo.space.fidelity_space import FidelitySpace
from prismbo.space.search_space import SearchSpace
from prismbo.utils.serialization import (multioutput_to_ndarray,
                                          output_to_ndarray)



class EVOBase(OptimizerBase):
    """
    The abstract Model for Evolutionary Optimization
    """
    def __init__(self, config):
        super(EVOBase, self).__init__(config=config)
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self.config = config
        self.search_space = None
        self.design_space = None
        self.mapping = None
        self.ini_num = None
        self.population = None
        self.pop_size = None
    
    def link_task(self, task_name:str, search_space: SearchSpace):
        self.task_name = task_name
        self.search_space = search_space
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))

