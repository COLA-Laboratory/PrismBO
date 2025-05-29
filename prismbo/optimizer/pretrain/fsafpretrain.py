# Copyright (c) 2021
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# ******************************************************************
# train_metabo_gps.py
# Train MetaBO on GP-samples
# The weights, stats, logs, and the learning curve are stored in metabo/log and can
# be evaluated using metabo/eval/evaluate.py
# ******************************************************************


import copy
import os

import numpy as np
import torch
import torch.nn as nn
import pickle

from datetime import datetime
from external.fsaf.policies.policies import NeuralAF
from external.fsaf.RL.DQN import DQN
import shutil

from external.fsaf.RL.plot_learning_curve_online import plot_learning_curve_online
from gym.envs.registration import register, registry


from prismbo.agent.registry import pretrain_registry
from prismbo.optimizer.pretrain.pretrain_base import PretrainBase



@pretrain_registry.register("FSAF")
class FSAFPretrain(PretrainBase):
    def __init__(self, config) -> None:
        super().__init__(config)
        
        torch.cuda.set_device(0)
        lengthScale = [0.1,0.2,0.3,0.1,0.2,0.3,0.3,0.5,0.6]
        dim = [3]
        # specifiy environment
        kernels = ["RBF"]*3+["Matern32"]*3+["SM"]*3
        kernel = "Matern32"
        inner_loop_steps = 5
        guide_points = 5

        n_iterations = 1000
        batch_size = 128
        n_workers = 5
        arch_spec = 4 * [200]
        num_particles = 5
        
        self.env_spec = {
                "env_id": "FSAF-GP-v0",
                "D": dim[0],
                "f_type": "GP",
                "f_opts": {
                            "kernel": kernel,
                        "lengthscale_low": 0.05,
                        "lengthscale_high": 0.6,
                        "noise_var_low": 0.1,
                        "noise_var_high": 0.1,
                        "signal_var_low": 1.0,
                        "signal_var_high": 1.0,
                        "min_regret": 1e-20,
                        "mix_kernel": True,
                        "periods":[0.3,0.6],
                        "kernel_list" : kernels,
                        "inner_loop_steps" : inner_loop_steps},
                "features": ["posterior_mean", "posterior_std", "incumbent", "timestep_perc"],
                "T_min": 30,
                "T_max": 100,
                "n_init_samples": 0,
                "pass_X_to_pi": False,
                # will be set individually for each new function to the sampled hyperparameters
                "kernel": kernel,
                "kernel_lengthscale": None,
                "kernel_variance": None,
                "noise_variance": None,
                "use_prior_mean_function": False,
                "local_af_opt": False,
                "cardinality_domain": 200,
                "reward_transformation": "neg_log10"  # true maximum not known
            }
        

        self.dqn_spec = {
                "batch_size": batch_size,
                "max_steps": n_iterations * batch_size,
                "lr": 1e-3,
                "inner_lr":1e-2,
                "gamma": 0.98,
                "buffer_size":1e3,
                "prior_alpha":0.3,
                "prior_beta":0.6,
                "outer_w":0.01,
                "n_steps":3,
                "task_size":3,
                "max_norm":40,
                "target_update_interval":5,
                "n_workers": n_workers,
                "env_id": self.env_spec["env_id"],
                "seed": 0,
                "env_seeds": list(range(n_workers)),
                "policy_options": {
                    "activations": "relu",
                    "arch_spec": arch_spec,
                    "use_value_network": True,
                    "t_idx": -2,
                    "T_idx": -1,
                    "arch_spec_value": arch_spec,
                },
                "kernels" : kernels,
                "lengthScale" : lengthScale,
                "num_particles" : num_particles,
                "ML" : False,
                "inner_loop_steps":inner_loop_steps,
                "using_chaser":True,
                "demo_prob" : 1/128,
            }

    
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

    
    def meta_train(self):
        # Create log directory
        logpath = os.path.join('./external/model/fsaf', "log", 
                             datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
        os.makedirs(logpath, exist_ok=True)

        # Set up policy
        policy_fn = lambda observation_space, action_space, deterministic: NeuralAF(
            observation_space=observation_space,
            action_space=action_space,
            deterministic=deterministic,
            options=self.dqn_spec["policy_options"]
        )

        # Initialize DQN
        print(f"Starting training...\nFind logs and weights at {logpath}\n\n")
        dqn = DQN(policy_fn=policy_fn, params=self.dqn_spec, logpath=logpath, save_interval=10)
        
        # Train using the provided data
        for dataset_name, data in self.train_data.items():
            X = data['X']
            y = data['y']
            print(f"Training on dataset: {dataset_name}")
            dqn.train_with_data(X, y)

        # Plot final learning curve
        plot_learning_curve_online(logpath=logpath, reload=False)
