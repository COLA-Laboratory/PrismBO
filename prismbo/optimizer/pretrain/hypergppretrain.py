import copy
import os

import gpytorch
import numpy as np
import torch
import torch.nn as nn
import pickle

from prismbo.agent.registry import pretrain_registry
from prismbo.optimizer.pretrain.pretrain_base import PretrainBase


from external.hyperbo.basics import definitions as defs
from external.hyperbo.basics import params_utils
from external.hyperbo.gp_utils import gp
from external.hyperbo.gp_utils import kernel
from external.hyperbo.gp_utils import mean
from external.hyperbo.gp_utils import utils
from external.hyperbo.bo_utils import data
from external.hyperbo.gp_utils import objectives as obj
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt


DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
SubDataset = defs.SubDataset


@pretrain_registry.register("HyperGP")
class HyperGPretrain(PretrainBase):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.mean_func = mean.constant
        self.cov_func = kernel.squared_exponential
        self.warp_func = DEFAULT_WARP_FUNC
        self.key = jax.random.PRNGKey(0)
        self._X = None
        self._Y = None

        self.params = GPParams(
            model={
                'constant': 5.,
                'lengthscale': 1.,
                'signal_variance': 1.0,
                'noise_variance': 0.01,
            },
            config={
                'method': 'adam',
                'learning_rate': 1e-5,
                'beta': 0.9,
                'max_training_step': 1000,
                'batch_size': 100,
                'retrain': 1,
            })

    
    def set_data(self, metadata, metadata_info= None):
        
        train_data = {}
        for dataset_name, data in metadata.items():
            objectives = metadata_info[dataset_name]["objectives"]
            obj = objectives[0]["name"]

            obj_data = [d[obj] for d in data]
            var_data = [[d[var["name"]] for var in metadata_info[dataset_name]["variables"]] for d in data]
            self.input_size = metadata_info[dataset_name]['num_variables']
            train_data[dataset_name] = {'X':np.array(var_data), 'y':np.array(obj_data)[:, np.newaxis]}
            
        self.train_data = train_data
        self.get_tasks()


    def get_tasks(self,):
        self.tasks = list(self.train_data.keys())

    def save_params(self, params, save_path):
        """Save GP parameters and functions to file.
        
        Args:
            params: GP parameters to save
            save_path: Path to save the parameters and functions
        """
        params_model = {k: np.array(v) for k, v in params.model.items()}
        params_config = {k: np.array(v) for k, v in params.config.items()}
        params_copy = {}
        params_copy['model'] = params_model
        params_copy['config'] = params_config

        with open(save_path, 'wb') as f:
            pickle.dump(params_copy, f)
    
    def meta_train(self):
        dataset = {}
        num_train_functions = len(self.train_data)
        for sub_dataset_id, data in self.train_data.items():
            x = jax.numpy.array(data['X'])
            y = jax.numpy.array(data['y'])
            dataset[str(sub_dataset_id)] = SubDataset(x, y)

        self.model = gp.GP(
            dataset=dataset,
            params=self.params,
            mean_func=self.mean_func,
            cov_func=self.cov_func,
            warp_func=self.warp_func,
        )
        assert self.key is not None, ('Cannot initialize with '
                                             'init_random_key == None.')
        key, subkey = jax.random.split(self.key)
        self.model.initialize_params(subkey)
        # Infer GP parameters.
        key, subkey = jax.random.split(self.key)
        self.model.train(subkey)
        
        # Save trained parameters
        save_dir = './external/model/hypergp/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'gp_params.pkl')
        self.save_params(self.model.params, save_path)