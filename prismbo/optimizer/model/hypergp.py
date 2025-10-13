import random
import time
import pickle

from typing import Sequence
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
from typing import Any, Callable, Dict, List, Tuple, Union

font = {
    'family': 'serif',
    'weight': 'normal',
    'size': 7,
}
axes = {'titlesize': 7, 'labelsize': 7}
matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)

DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
SubDataset = defs.SubDataset


import copy
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler

from prismbo.optimizer.model.model_base import  Model
from prismbo.agent.registry import model_registry


@model_registry.register('HyperGP')
class HyperGP(Model):
    def __init__(self, config, seed = 0):
        self.config = config
        self.n_features = None
        self.key = jax.random.PRNGKey(seed)
        self._X = None 
        self._Y = None
        self.target_dataset_id = None
        self._meta_data = {}
        self.mean_func = mean.constant
        self.cov_func = kernel.squared_exponential
        self.warp_func = DEFAULT_WARP_FUNC

        # Try to load saved GP parameters
        load_path = './external/model/hypergp/gp_params.pkl'
        try:
            with open(load_path, 'rb') as f:
                saved_dict = pickle.load(f)
                self.params = GPParams(
                    model={
                        'constant': saved_dict['model']['constant'],
                        'lengthscale': saved_dict['model']['lengthscale'],
                        'signal_variance': saved_dict['model']['signal_variance'],
                        'noise_variance': saved_dict['model']['noise_variance'],
                    },
                    config={
                        'method': saved_dict['config']['method'],
                        'learning_rate': saved_dict['config']['learning_rate'],
                        'beta': saved_dict['config']['beta'],
                        'max_training_step': saved_dict['config']['max_training_step'],
                        'batch_size': saved_dict['config']['batch_size'],
                        'retrain': saved_dict['config']['retrain'],
                    })
            self.model = gp.GP(
                dataset={},
                params=self.params,
                mean_func=self.mean_func,
                cov_func=self.cov_func,
                warp_func=self.warp_func,
            )
        except:
            # Initialize new parameters if loading fails
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
            
            self.model = gp.GP(
                dataset={},
                params=self.params,
                mean_func=self.mean_func,
                cov_func=self.cov_func,
                warp_func=self.warp_func,
            )
            key, subkey = jax.random.split(self.key)
            self.model.initialize_params(subkey)


    def meta_fit(self,
            metadata : Dict,
            optimize: Union[bool, Sequence[bool]] = True):
        # metadata, _ = SourceSelection.the_k_nearest(source_datasets)
        self._meta_data = metadata
        train_data = {}

        for key in metadata.keys():
            X = jax.numpy.array(metadata[key]['X'])
            Y = jax.numpy.array(metadata[key]['Y'])
            dataset =  SubDataset(X, Y)
            train_data[key] = {'X':X, 'Y':Y}
            self.model.update_sub_dataset(
                dataset, sub_dataset_key=str(key), is_append=False)
        
        key, subkey = jax.random.split(self.key)

        self.model.train(subkey)
        
    def fit(self, 
            X: np.ndarray,
            Y: np.ndarray,
            optimize: bool = False):

        self._X = copy.deepcopy(X)
        self._Y = copy.deepcopy(Y)

        self.n_samples, n_features = self._X.shape
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Number of features in model and input data mismatch.")
        
        if self.target_dataset_id is None:
            self.target_dataset_id = len(self._meta_data)
        
        self.retrain(data={'X':self._X, 'Y':self._Y})

            

    def retrain(self, data):
        self._X = data['X']
        self._Y = data['Y']
        x = jax.numpy.array(self._X)
        y = jax.numpy.array(self._Y)
        dataset =  SubDataset(x, y)

        self.model.update_sub_dataset(
            dataset, sub_dataset_key=str(self.target_dataset_id), is_append=False)

        retrain_condition = 'retrain' in self.model.params.config and self.model.params.config[
            'retrain'] > 0 and self.model.dataset[str(self.target_dataset_id)].x.shape[0] > 0
        if not retrain_condition:
            return
        if self.model.params.config['objective'] in [obj.regkl, obj.regeuc]:
            raise ValueError('Objective must include NLL to retrain.')
        max_training_step = self.model.params.config['retrain']
        self.model.params.config['max_training_step'] = max_training_step
        key, subkey = jax.random.split(self.key)
        self.model.train(subkey)

    def predict(self, X, subset_data_id:Union[int, str] = 0):
        _X = jnp.array(X)
        mu, var = self.model.predict(_X, subset_data_id)
        
        return np.array(mu), np.array(var)



    def get_fmin(self):
        return np.min(self._Y)
    
    def set_XY(self, Data:Dict):
        self._X = copy.deepcopy(Data['X'])
        self._Y = copy.deepcopy(Data['Y'])
        
    def meta_update(self):
        n_metadata = len(self._metadata)
        self._metadata[self.target_dataset_id] = {'X': self._X, 'Y': self._Y}
        x = jax.numpy.array(self._X)
        y = jax.numpy.array(self._Y)
        dataset =  SubDataset(x, y)
        
        self.model.update_sub_dataset(
                dataset, sub_dataset_key=str(self.target_dataset_id), is_append=False)
        self.target_dataset_id = len(self._metadata)
