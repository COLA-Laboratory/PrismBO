import abc
import copy
import math
from typing import Dict, List, Union

import GPyOpt
import numpy as np

from prismbo.optimizer.acquisition_function.sequential import Sequential
from prismbo.optimizer.optimizer_base.base import OptimizerBase
from prismbo.space.fidelity_space import FidelitySpace
from prismbo.space.search_space import SearchSpace
from prismbo.utils.serialization import (multioutput_to_ndarray,
                                          output_to_ndarray)


class BO(OptimizerBase):
    """
    The abstract Model for Bayesian Optimization
    """

    def __init__(self, Refiner, Sampler, ACF, Pretrain, Model, Normalizer, config):
        super(BO, self).__init__(config=config)
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self.config = config
        self.search_space = None
        
        self.SpaceRefiner = Refiner
        self.Sampler = Sampler
        self.ACF = ACF
        self.Pretrain = Pretrain
        self.Model = Model
        self.Normalizer = Normalizer

        
        self.ACF.link_model(model=self.Model)
        
        self.MetaData = None
    
    def link_task(self, task_name:str, search_space: SearchSpace):
        self.task_name = task_name
        self.search_space = search_space
        self._X = np.empty((0,))  # Initializes an empty ndarray for input vectors
        self._Y = np.empty((0,))
        self.ACF.link_space(self.search_space)
        self.evaluator = Sequential(self.ACF, batch_size=1)
        self.Normalizer.clear()
            
    
    def search_space_refine(self, search_space, metadata = None, metadata_info = None):
        if self.SpaceRefiner is not None:
            search_space = self.SpaceRefiner.prune(search_space, (self._X, self._Y), metadata, metadata_info)
            self.search_space = search_space
            self.ACF.link_space(self.search_space)
            self.evaluator = Sequential(self.ACF, batch_size=1)

            
    def sample_initial_set(self, metadata = None, metadata_info = None):
        return self.Sampler.sample(self.search_space, metadata, metadata_info)
    
    def pretrain(self, metadata = None, metadata_info = None):
        if self.Pretrain:
            self.Pretrain.set_data(metadata, metadata_info)
            self.Pretrain.meta_train()
    
    
    def ACF_meta(self, metadata = None, metadata_info = None):
        datasets = []
        for key, v in metadata.items():
            variables_order = [i['name'] for i in metadata_info[key]['variables']]
            objective_name = metadata_info[key]['objectives'][0]['name']
            dataset_X = np.array([[point[name] for name in variables_order] for point in v])
            dataset_Y = np.array([point[objective_name] for point in v])
            datasets.append({'X': dataset_X, 'Y': dataset_Y})
        self.ACF.link_data(datasets)
    
    def meta_fit(self, metadata = None, metadata_info = None):
        if metadata:
            source_X = []
            source_Y = []
            for key, datasets in metadata.items():
                data_info = metadata_info[key]
                source_X.append(np.array([[data[var['name']] for var in data_info['variables']] for data in datasets]))
                source_Y.append(np.array([[data[var['name']] for var in data_info['objectives']] for data in datasets]))
                
            self.Model.meta_fit(source_X, source_Y)
    
    def fit(self):

        Y = copy.deepcopy(self._Y)

        X = copy.deepcopy(self._X)

        self.Model.fit(X, Y, optimize = True)
    
    def suggest(self):
        suggested_sample, acq_value = self.evaluator.compute_batch(None, context_manager=None)

        return suggested_sample

        
    def observe(self, X: np.ndarray, Y: List[Dict]) -> None:
        # Check if the lists are empty and return if they are
        if X.shape[0] == 0 or len(Y) == 0:
            return

        Y = np.array(output_to_ndarray(Y))
        if self.Normalizer:
            self.Normalizer.update(Y)
            Y = self.Normalizer.transform(Y)
        
        self._X = np.vstack((self._X, X)) if self._X.size else X
        self._Y = np.vstack((self._Y, Y)) if self._Y.size else Y
        
    def meta_observe(self, metadata = None, searchspace_info = None):
        self.Model.meta_update()


