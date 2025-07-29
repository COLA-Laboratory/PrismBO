
import os
import sys

import copy

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace
import torch
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.compose import ColumnTransformer
from scipy.special import logit, expit
# from scipy.stats import qmc
from torch.quasirandom import SobolEngine
from external.pfns4bo.scripts import acquisition_functions, tune_input_warping
from external.pfns4bo import hebo_plus_model
import external.pfns4bo as pfns4bo_module
from sklearn.preprocessing import power_transform, PowerTransformer

# 注册一个伪模块名 alias
sys.modules['pfns4bo'] = pfns4bo_module

from typing import Dict, Hashable, Union, Sequence, Tuple, List

from prismbo.optimizer.model.model_base import Model
from prismbo.agent.registry import model_registry

@model_registry.register("PFNs4BO")
class PFNs4BO(Model):
    def __init__(self, config = {}):
        super(PFNs4BO, self).__init__()
        # assert not 'fit_encoder' in acqf_kwargs
        # AbstractOptimizer.__init__(self, api_config)
        # Do whatever other setup is needed
        # ...
        
        from bayesmark.experiment import _build_test_problem

        # function_instance = _build_test_problem(model_name='ada', dataset='diabetes', scorer='mse', path=None)

        self.n_features = None
        # Setup optimizer
        # api_config = function_instance.get_api_config()
        
        self.config = {
            "pfn_file": hebo_plus_model,
            # alternatively give a relative path from pfns4bo
            #"pfn_file" : "final_models/model_hebo_morebudget_9_unused_features_3.pt",
            "minimize": 1,
            "fit_encoder_from_step": None,
            "sample_only_valid": 1,
            "pre_sample_size": 1000,
            "num_candidates": 10,
            "max_initial_design": 1,
            "fixed_initial_guess": 0.0
        }
        
        self.pfn_file = self.config['pfn_file']
        self.minimize = self.config['minimize']
        self.fit_encoder_from_step = self.config['fit_encoder_from_step']
        self.sample_only_valid = self.config['sample_only_valid']
        self.pre_sample_size = self.config['pre_sample_size']
        self.num_candidates = self.config['num_candidates']
        self.max_initial_design = self.config['max_initial_design']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.load(self.pfn_file, weights_only=False) if self.pfn_file.startswith('/') else torch.load(os.path.dirname(__file__) + '/' + self.pfn_file, weights_only=False)

        # self.api_config = {key: value for key, value in sorted(api_config.items())}
        # self.hp_names = list(self.api_config.keys())

        self.epsilon = 1e-8
        # self.create_scaler()
        # self.sobol = SobolEngine(len(self.max_values), scramble=True)
        self.acqf_optimizer_name = "lbfgs"
        
        
        
        # assert not (rand_bool and sample_only_valid)
        
        self.round_suggests_to = 4
        self.min_initial_design = 0
        self.max_initial_design = None
        self.rand_sugg_after_x_steps_of_stagnation = None
        self.fixed_initial_guess = None
        self.minmax_encode_y = False
        
        self.rand_bool = False
        self.sample_only_valid = False
        self.verbose = False
        self.model.to(self.device)
        self.model.eval()

        # print(api_config)
        # self.space_x = JointSpace(api_config)
        # self.bounds = self.space_x.get_bounds()
        
    def create_scaler(self):

        list_of_scalers = []
        self.min_values = []
        self.max_values = []
        self.spaces = []
        self.types = []

        for i, feature in enumerate(self.api_config):
            # list_of_scalers.append((feature, MinMaxScaler(feature_range),i))
            self.spaces.append(self.api_config[feature].get("space", "bool"))
            self.types.append(self.api_config[feature]["type"])

            if self.types[-1] == "bool":
                feature_range = [0, 1]
            else:
                feature_range = list(self.api_config[feature]["range"])

            feature_range[0] = self.transform_feature(feature_range[0], -1)
            feature_range[1] = self.transform_feature(feature_range[1], -1)

            self.min_values.append(feature_range[0])
            self.max_values.append(feature_range[1])

        self.column_scaler = ColumnTransformer(list_of_scalers)
        self.max_values: np.array = np.array(self.max_values)
        self.min_values: np.array = np.array(self.min_values)

    def transform_feature_inverse(self, x, feature_index):

        if self.spaces[feature_index] == "log":
            x = np.exp(x)
        elif self.spaces[feature_index] == "logit":
            x = expit(x)
        if self.types[feature_index] == "int":
            if self.rand_bool:
                x = int(x) + int(np.random.rand() < (x-int(x)))
            else:
                x = int(np.round(x))
        elif self.types[feature_index] == "bool":
            if self.rand_bool:
                x = np.random.rand() < x
            else:
                x = bool(np.round(x))

        return x

    def transform_feature(self, x, feature_index):

        if np.isinf(x) or np.isnan(x):
            return 0

        if self.spaces[feature_index] == "log":
            x = np.log(x)

        elif self.spaces[feature_index] == "logit":
            x = logit(x)

        elif self.types[feature_index] == "bool":
            x = int(x)
        return x


    def get_model_likelihood_mll(self, train_size):
        pass



    def fit(self,
            X: np.ndarray,
            Y: np.ndarray,
            optimize: bool = False,):

        self._X = copy.deepcopy(X)
        self._Y = copy.deepcopy(Y)

        self.n_samples, n_features = self._X.shape
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Number of features in model and input data mismatch.")

                
        
    def meta_fit(self,
                source_X : List[np.ndarray],
                source_Y : List[np.ndarray],
                optimize: Union[bool, Sequence[bool]] = True):
        # metadata, _ = SourceSelection.the_k_nearest(source_datasets)
        _metadata = {'X': source_X, 'Y':source_Y}
        
        return
    
    def load_checkpoint(self, checkpoint):
        pass


    @torch.no_grad()
    def predict(self, X):
        try:
            temp_X = np.array(self.X)
            temp_X = self.min_max_encode(temp_X)
            if self.minmax_encode_y:
                temp_y = MinMaxScaler().fit_transform(np.array(self.Y).reshape(-1, 1)).reshape(-1)
            else:
                temp_y = np.array(self.y)
            temp_y = torch.tensor(temp_y).to(torch.float32)
            if self.rand_sugg_after_x_steps_of_stagnation is not None \
                    and len(self.y) > self.rand_sugg_after_x_steps_of_stagnation\
                    and not self.rand_prev:
                if temp_y[:-self.rand_sugg_after_x_steps_of_stagnation].max() == temp_y.max():
                    print(f"Random suggestion after >= {self.rand_sugg_after_x_steps_of_stagnation} steps of stagnation")
                    x_guess = self.random_suggest()
                    return x_guess
            if self.verbose:
                from matplotlib import pyplot as plt
                print(f"{temp_X=}, {temp_y=}")
                if temp_X.shape[1] == 2:
                    from scipy.stats import rankdata
                    plt.title('Observations, red -> blue.')
                    plt.scatter(temp_X[:,0], temp_X[:,1], cmap='RdBu', c=rankdata(temp_y))
                    plt.show()

            temp_X = temp_X.to(self.device)
            temp_y = temp_y.to(self.device)

            if self.fit_encoder_from_step and self.fit_encoder_from_step <= len(self.X):
                with torch.enable_grad():
                    w = tune_input_warping.fit_input_warping(self.model, temp_X, temp_y)
                temp_X_warped = w(temp_X).detach()
            else:
                temp_X_warped = temp_X
                
            eval_X = np.array(X)
            eval_X = self.min_max_encode(eval_X)
            eval_X = eval_X.to(self.device)

            with torch.enable_grad():
                """
                Predict logits for x_eval based on x_given and y_given using a transformer model.
                """
                temp_y = temp_y.reshape(-1)
                self.predict_with_transformer_model(self.model, temp_X, temp_y, eval_X, self.device)


        #         if input_znormalize:
        #             std = x_given.std(dim=0)
        #             std[std == 0.] = 1.
        #             mean = x_given.mean(dim=0)
        #             x_given = (x_given - mean) / std
        #             x_eval = (x_eval - mean) / std

        #         if input_power_transform:
        #             x_given = general_power_transform(x_given, x_given, power_transform_eps)
        #             x_eval = general_power_transform(x_given, x_eval, power_transform_eps)

        #         x_predict = torch.cat([x_given, x_eval], dim=0)

        #         logits_list = []
        #         for x_feed in torch.split(x_predict, max_dataset_size, dim=0):
        #             x_full_feed = torch.cat([x_given, x_feed], dim=0).unsqueeze(1)
        #             y_full_feed = y_given.unsqueeze(1)

        #             if ensemble_log_dims == '01':
        #                 x_full_feed = log01_batch(x_full_feed)
        #             elif ensemble_log_dims == 'global01' or ensemble_log_dims is True:
        #                 x_full_feed = log01_batch(x_full_feed, input_between_zero_and_one=True)
        #             elif ensemble_log_dims == '01-10':
        #                 x_full_feed = torch.cat((log01_batch(x_full_feed)[:, :-1], log01_batch(1. - x_full_feed)), 1)
        #             elif ensemble_log_dims == 'norm':
        #                 x_full_feed = lognormed_batch(x_full_feed, len(x_given))
        #             elif ensemble_log_dims is not False:
        #                 raise NotImplementedError

        #             if ensemble_feature_rotation:
        #                 x_full_feed = torch.cat([x_full_feed[:, :, (i+torch.arange(x_full_feed.shape[2])) % x_full_feed.shape[2]] for i in range(x_full_feed.shape[2])], dim=1)

        #             if ensemble_input_rank_transform == 'train' or ensemble_input_rank_transform is True:
        #                 x_full_feed = torch.cat(
        #                     [rank_transform(x_given, x_full_feed[:, i, :])[:, None] for i in range(x_full_feed.shape[1])] + [x_full_feed],
        #                     dim=1
        #                 )

        #             if style is not None:
        #                 if callable(style):
        #                     style = style()
        #                 if isinstance(style, torch.Tensor):
        #                     style = style.to(x_full_feed.device)
        #                 else:
        #                     style = torch.tensor(style, device=x_full_feed.device).view(1, 1).repeat(x_full_feed.shape[1], 1)

        #             logits = model(
        #                 (style,
        #                 x_full_feed.repeat_interleave(dim=1, repeats=y_full_feed.shape[1]),
        #                 y_full_feed.repeat(1, x_full_feed.shape[1])),
        #                 single_eval_pos=len(x_given)
        #             )

        #             if ensemble_type == 'mean_probs':
        #                 logits = logits.softmax(-1).mean(1, keepdim=True).log_()

        #             logits_list.append(logits)

        #         logits = torch.cat(logits_list, dim=0)

        #         logits_given = logits[:len(x_given)]
        #         logits_eval = logits[len(x_given):]

        #         return logits_eval, logits_given
                
                
                
                
                
                
        #         if self.acqf_optimizer_name == "lbfgs":
        #             def rand_sample_func(n):
        #                 pre_samples = torch.rand(n, temp_X_warped.shape[1], device='cpu')
        #                 back_transformed_samples = [self.transform_back(sample) for sample in pre_samples]
        #                 samples = np.array([self.transform(deepcopy(bt_sample)) for bt_sample in back_transformed_samples])
        #                 samples = self.min_max_encode(samples)
        #                 return samples.to(self.device)

        #             if self.sample_only_valid:
        #                 rand_sample_func = rand_sample_func
        #                 # dims with bool or int are not continuous, thus no gradient opt is applied
        #                 dims_wo_gradient_opt = [i for i, t in enumerate(self.types) if t != "real"]
        #             else:
        #                 rand_sample_func = None
        #                 dims_wo_gradient_opt = []

        #             x_guess, x_options, eis, x_rs, x_rs_eis = acquisition_functions.optimize_acq_w_lbfgs(
        #                 self.model, temp_X_warped, temp_y, device=self.device,
        #                 verbose=self.verbose, rand_sample_func=rand_sample_func,
        #                 dims_wo_gradient_opt=dims_wo_gradient_opt, **{'apply_power_transform':True,**self.acqf_kwargs}
        #             )

        #         elif self.acqf_optimizer_name == 'adam':
        #             x_guess = acquisition_functions.optimize_acq(self.model, temp_X_warped, temp_y, apply_power_transform=True, device=self.device, **self.acqf_kwargs
        #                                     ).detach().cpu().numpy()
        #         else:
        #             raise ValueError("Optimizer not recognized, set `acqf_optimizer_name` to 'lbfgs' or 'adam'")


        #     back_transformed_x_options = [self.transform_back(x) for x in x_options]
        #     opt_X = np.array([self.transform(deepcopy(transformed_x_options)) for transformed_x_options in back_transformed_x_options])
        #     opt_X = self.min_max_encode(opt_X)
        #     opt_new = ~(opt_X[:,None] == temp_X[None].cpu()).all(-1).any(1)
        #     for i, x in enumerate(opt_X):
        #         if opt_new[i]:
        #             if self.verbose: print(f"New point at pos {i}: {back_transformed_x_options[i], x_options[i]}")
        #             self.rand_prev = False
        #             return [back_transformed_x_options[i]]
        #     print('backup from initial rand search')
        #     back_transformed_x_options = [self.transform_back(x) for x in x_rs]
        #     opt_X = np.array([self.transform(deepcopy(transformed_x_options)) for transformed_x_options in back_transformed_x_options])
        #     opt_X = self.min_max_encode(opt_X)
        #     opt_new = ~(opt_X[:,None] == temp_X[None].cpu()).all(-1).any(1)
        #     for i, x in enumerate(opt_X):
        #         if opt_new[i]:
        #             if self.verbose: print(f"New point at pos {i}: {back_transformed_x_options[i], x_rs[i]} with ei {x_rs_eis[i]}")
        #             self.rand_prev = False
        #             return [back_transformed_x_options[i]]
        #     print("No new points found, random suggestion")
        #     return self.random_suggest()
        except Exception as e:
            raise e

    def predict_with_transformer_model(self, model, x_given, y_given, x_eval, device,
                                        max_dataset_size=10000,
                                        apply_power_transform=True,
                                        input_znormalize=False,
                                        input_power_transform=False,
                                        power_transform_eps=0.0,
                                        ensemble_log_dims=False,
                                        ensemble_type='mean_probs',
                                        ensemble_input_rank_transform=False,
                                        ensemble_feature_rotation=False,
                                        style=None):
        """
        Predict logits for x_eval based on x_given and y_given using a transformer model.
        """

        y_given = y_given.reshape(-1).to(device)

        if apply_power_transform:
            y_given = self.general_power_transform(y_given.unsqueeze(1), y_given.unsqueeze(1), power_transform_eps).squeeze(1)

        if input_znormalize:
            std = x_given.std(dim=0)
            std[std == 0.] = 1.
            mean = x_given.mean(dim=0)
            x_given = (x_given - mean) / std
            x_eval = (x_eval - mean) / std

        if input_power_transform:
            x_given = self.general_power_transform(x_given, x_given, power_transform_eps)
            x_eval = self.general_power_transform(x_given, x_eval, power_transform_eps)

        x_predict = torch.cat([x_given, x_eval], dim=0)

        logits_list = []
        for x_feed in torch.split(x_predict, max_dataset_size, dim=0):
            x_full_feed = torch.cat([x_given, x_feed], dim=0).unsqueeze(1)
            y_full_feed = y_given.unsqueeze(1)

            if ensemble_log_dims == '01':
                x_full_feed = log01_batch(x_full_feed)
            elif ensemble_log_dims == 'global01' or ensemble_log_dims is True:
                x_full_feed = log01_batch(x_full_feed, input_between_zero_and_one=True)
            elif ensemble_log_dims == '01-10':
                x_full_feed = torch.cat((log01_batch(x_full_feed)[:, :-1], log01_batch(1. - x_full_feed)), 1)
            elif ensemble_log_dims == 'norm':
                x_full_feed = lognormed_batch(x_full_feed, len(x_given))
            elif ensemble_log_dims is not False:
                raise NotImplementedError

            if ensemble_feature_rotation:
                x_full_feed = torch.cat([x_full_feed[:, :, (i+torch.arange(x_full_feed.shape[2])) % x_full_feed.shape[2]] for i in range(x_full_feed.shape[2])], dim=1)

            if ensemble_input_rank_transform == 'train' or ensemble_input_rank_transform is True:
                x_full_feed = torch.cat(
                    [rank_transform(x_given, x_full_feed[:, i, :])[:, None] for i in range(x_full_feed.shape[1])] + [x_full_feed],
                    dim=1
                )

            if style is not None:
                if callable(style):
                    style = style()
                if isinstance(style, torch.Tensor):
                    style = style.to(x_full_feed.device)
                else:
                    style = torch.tensor(style, device=x_full_feed.device).view(1, 1).repeat(x_full_feed.shape[1], 1)

            logits = model(
                (style,
                x_full_feed.repeat_interleave(dim=1, repeats=y_full_feed.shape[1]),
                y_full_feed.repeat(1, x_full_feed.shape[1])),
                single_eval_pos=len(x_given)
            )

            if ensemble_type == 'mean_probs':
                logits = logits.softmax(-1).mean(1, keepdim=True).log_()

            logits_list.append(logits)

        logits = torch.cat(logits_list, dim=0)

        logits_given = logits[:len(x_given)]
        logits_eval = logits[len(x_given):]

        return logits_eval, logits_given

    def transform_back(self, x_guess):
        if self.round_suggests_to is not None:
            x_guess = np.round(x_guess, self.round_suggests_to)  # make sure very similar values are actually the same
        x_guess = x_guess * (self.max_values - self.min_values) + self.min_values
        x_guess = x_guess.tolist()
        return self.transform_inverse(x_guess)

    def min_max_encode(self, temp_X):
        self.min_values = np.min(self.X, axis=0)
        self.max_values = np.max(self.X, axis=0)
        # this, combined with transform is the inverse of transform_back
        temp_X = (temp_X - self.min_values) / (self.max_values - self.min_values)
        temp_X = torch.tensor(temp_X).to(torch.float32)
        temp_X = torch.clamp(temp_X, min=0., max=1.)
        return temp_X

    def transform(self, X_dict):
        X_tf = []
        for i, feature in enumerate(X_dict.keys()):
            X_dict[feature] = self.transform_feature(X_dict[feature], i)
            X_tf.append(X_dict[feature])
        return X_tf

    def transform_inverse(self, X_list):
        X_tf = {}
        for i, hp_name in enumerate(self.hp_names):
            X_tf[hp_name] = self.transform_feature_inverse(X_list[i], i)
        return X_tf
    
    def observe(self, X, y):
            """Feed an observation back.

            Parameters
            ----------
            X : list of dict-like
                Places where the objective function has already been evaluated.
                Each suggestion is a dictionary where each key corresponds to a
                parameter being optimized.
            y : array-like, shape (n,)
                Corresponding values where objective has been evaluated
            """
            # Update the model with new objective function observations
            # ...
            # No return statement needed
            if np.isinf(y) and y > 0:
                y[:] = 1e10

            if not np.isnan(y) and not np.isinf(y):
                assert len(y) == 1 and len(X) == 1, "Only one suggestion at a time is supported"
                X = {key: value for key, value in sorted(X[0].items())}
                assert list(X.keys()) == list(self.api_config.keys()) == list(self.hp_names) == list(
                    self.space_x.param_list)
                if self.verbose:
                    print(f"{X=}, {y=}")
                X = self.transform(X)
                if self.verbose:
                    print(f"transformed {X=}")
                self.X.append(X)
                if self.minimize:
                    self.y.append(-y[0])
                else:
                    self.y.append(y[0])
            else:
                assert False

    def general_power_transform(self, x_train, x_apply, eps, less_safe=False):
        if eps > 0:
            try:
                pt = PowerTransformer(method='box-cox')
                pt.fit(x_train.cpu()+eps)
                x_out = torch.tensor(pt.transform(x_apply.cpu()+eps), dtype=x_apply.dtype, device=x_apply.device)
            except ValueError as e:
                print(e)
                x_out = x_apply - x_train.mean(0)
        else:
            pt = PowerTransformer(method='yeo-johnson')
            if not less_safe and (x_train.std() > 1_000 or x_train.mean().abs() > 1_000):
                x_apply = (x_apply - x_train.mean(0)) / x_train.std(0)
                x_train = (x_train - x_train.mean(0)) / x_train.std(0)
                print('inputs are LAARGEe, normalizing them')
            try:
                pt.fit(x_train.cpu().double())
            except ValueError as e:
                print('caught this errrr', e)
                if less_safe:
                    x_train = (x_train - x_train.mean(0)) / x_train.std(0)
                    x_apply = (x_apply - x_train.mean(0)) / x_train.std(0)
                else:
                    x_train = x_train - x_train.mean(0)
                    x_apply = x_apply - x_train.mean(0)
                pt.fit(x_train.cpu().double())
            x_out = torch.tensor(pt.transform(x_apply.cpu()), dtype=x_apply.dtype, device=x_apply.device)
        if torch.isnan(x_out).any() or torch.isinf(x_out).any():
            print('WARNING: power transform failed')
            print(f"{x_train=} and {x_apply=}")
            x_out = x_apply - x_train.mean(0)
        return x_out

    def continuous_maximization( self, dim, bounds, acqf):

        return result.x.reshape(-1,dim)


    def get_fmin(self):
        return np.min(self.y_obs.detach().to("cpu").numpy())
    
    
    def meta_update(self):
        pass
    
    
# #function_instance = _build_test_problem(model_name='ada', dataset='breast', scorer='nll', path=None)
# function_instance = _build_test_problem(model_name='ada', dataset='boston', scorer='mse', path=None)

# # Setup optimizer
# api_config = function_instance.get_api_config()
# # check is file

# opt = PFNOptimizer(api_config, verbose=True, device="cpu:0", **config)

# function_evals, timing, suggest_log = run_study(
#     opt, function_instance, n_calls=3, n_suggestions=1, callback=None, n_obj=len(OBJECTIVE_NAMES),
# )