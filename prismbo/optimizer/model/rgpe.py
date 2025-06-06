#Practical gaussian process
import copy
from typing import Dict, List, Union, Sequence

import GPy
import numpy as np
from GPy.kern import RBF, Kern

from prismbo.agent.registry import model_registry
from prismbo.optimizer.model.gp import GP
from prismbo.optimizer.model.model_base import Model
from sklearn.ensemble import RandomForestRegressor

from prismbo.optimizer.model.tpe import TPE

import matplotlib.pyplot as plt

def roll_col(X: np.ndarray, shift: int) -> np.ndarray:
    """
    Rotate columns to right by shift.
    """
    return np.concatenate((X[:, -shift:], X[:, :-shift]), axis=1)


def compute_ranking_loss(
    f_samps: np.ndarray,
    target_y: np.ndarray,
    target_model: bool,
) -> np.ndarray:
    """
    Compute ranking loss for each sample from the posterior over target points.
    """
    y_stack = np.tile(target_y.reshape((-1, 1)), f_samps.shape[0]).transpose()
    rank_loss = np.zeros(f_samps.shape[0])
    if not target_model:
        for i in range(1, target_y.shape[0]):
            rank_loss += np.sum(
                (roll_col(f_samps, i) < f_samps) ^ (roll_col(y_stack, i) < y_stack),
                axis=1
            )
    else:
        for i in range(1, target_y.shape[0]):
            rank_loss += np.sum(
                (roll_col(f_samps, i) < y_stack) ^ (roll_col(y_stack, i) < y_stack),
                axis=1
            )

    return rank_loss


@model_registry.register('RGPE')
class RGPE(Model):
    def __init__(
            self,
            kernel: Kern = None,
            noise_variance: float = 1.0,
            normalize: bool = True,
            Seed = 0,
            sampling_mode: str = 'bootstrap',
            weight_dilution_strategy = 'probabilistic',
            **options: dict,
    ):
        super().__init__()
        # GP on difference between target data and last source data set
        self._noise_variance = noise_variance
        self._metadata = {}
        self._metadata_info = {}
        self._source_gps = {}
        self._source_gp_weights = {}
        self.model_name = 'RF'
        self.sampling_mode = sampling_mode
        self._normalize = normalize
        self.Seed = Seed
        self.n_features = None
        self.rng = np.random.RandomState(self.Seed)
        self.weight_dilution_strategy = weight_dilution_strategy
        self._weights_need_update = True  # Initialize the flag

        self.target_model = None
        self._target_model_weight = 1
    
    
    def _meta_fit_single_model(
        self,
        X : np.ndarray,
        Y : np.ndarray,
        optimize: bool,
    ):
        """Train a new source GP on `data`.

        Args:
            data: The source dataset.
            optimize: Switch to run hyperparameter optimization.

        Returns:
            The newly trained GP.
        """
        self.n_features = X.shape[1]

        if self.model_name == 'GP':
            kern = GPy.kern.RBF(self.n_features, ARD=False)
            new_model = GPy.models.GPRegression(X, Y, kernel=kern)
            try:
                new_model.optimize_restarts(num_restarts=1, verbose=True, robust=True)
            except np.linalg.linalg.LinAlgError as e:
                print('Error: np.linalg.linalg.LinAlgError')

        elif self.model_name == 'RF':
            new_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5, min_samples_leaf=1, min_samples_split=2)
            new_model.fit(X, Y.ravel())
        elif self.model_name == 'TPE':
            # Initialize TPE model
            new_model = TPE()
            # Fit TPE model with observed data
            new_model.fit(X, Y.ravel())
        else:
            raise ValueError(f'Invalid model name: {self.model_name}')
        
        
        return new_model
    
    def meta_fit(self,
                source_X : List[np.ndarray],
                source_Y : List[np.ndarray],
                optimize: Union[bool, Sequence[bool]] = True):
        # metadata, _ = SourceSelection.the_k_nearest(source_datasets)
        self._metadata = {'X': source_X, 'Y':source_Y}
        self._source_gps = {}
        
        
        assert isinstance(optimize, bool) or isinstance(optimize, list)
        if isinstance(optimize, list):
            assert len(source_X) == len(optimize)
        optimize_flag = copy.copy(optimize)

        if isinstance(optimize_flag, bool):
            optimize_flag = [optimize_flag] * len(source_X)
        
        for i in range(len(source_X)):
            new_gp = self._meta_fit_single_model(
                source_X[i],
                source_Y[i],
                optimize=optimize_flag[i],
            )
            self._source_gps[i] = new_gp

        self._calculate_weights()

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

        if self.model_name == 'GP':
            kern = GPy.kern.RBF(self.n_features, ARD=False)
            self.target_model = GPy.models.GPRegression(self._X, self._Y, kernel=kern)
            self.target_model['Gaussian_noise.*variance'].constrain_bounded(1e-9, 1e-3)
            try:
                self.target_model.optimize_restarts(num_restarts=1, verbose=True, robust=True)
            except np.linalg.linalg.LinAlgError as e:
                print('Error: np.linalg.linalg.LinAlgError')

        elif self.model_name == 'RF':
            self.target_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5, min_samples_leaf=1, min_samples_split=2)
            self.target_model.fit(self._X, self._Y.ravel())
        elif self.model_name == 'TPE':
            # Initialize TPE model
            self.target_model = TPE()
            # Fit TPE model with observed data
            self.target_model.fit(self._X, self._Y.ravel())
        else:
            raise ValueError(f'Invalid model name: {self.model_name}')

        # Call _calculate_weights only if needed
        if self._weights_need_update:
            self._calculate_weights()
            self._weights_need_update = False  # Reset the flag after updating weights
        
        # self.plot_target_model()

    def predict(
        self, X, return_full: bool = False, with_noise: bool = False
    ):

        X_test = X
        n_models = len(self._source_gp_weights)
        if self._target_model_weight > 0:
            n_models += 1
        n_sample = X_test.shape[0]
        means = np.empty((n_models, n_sample, 1))
        weights = np.empty((n_models, 1))
        if return_full == False:
            vars_ = np.empty((n_models, n_sample, 1))
        else:
            vars_ = np.empty((n_models, n_sample, n_sample))

        # Predict from source models
        for task_uid, weight in enumerate(self._source_gp_weights):
            model = self._source_gps[task_uid]
            
            # Predict based on model type
            if self.model_name == 'GP':
                means[task_uid], vars_[task_uid] = model.predict(X)
            elif self.model_name == 'RF':
                tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
                means[task_uid] = np.mean(tree_predictions, axis=0).reshape(-1, 1)
                vars_[task_uid] = np.var(tree_predictions, axis=0).reshape(-1, 1)
            elif self.model_name == 'TPE':
                means[task_uid], vars_[task_uid] = model.predict(X)
            else:
                raise ValueError(f'Invalid model name: {self.model_name}')

            weights[task_uid] = weight

        # Predict from target model based on model type
        if self._target_model_weight > 0:
            if self.model_name == 'GP':
                means, vars_ = self.target_model.predict(X_test)
            elif self.model_name == 'RF':
                # For RF, get mean prediction and variance across trees
                tree_predictions = np.array([tree.predict(X_test) for tree in self.target_model.estimators_])
                means = np.mean(tree_predictions, axis=0).reshape(-1, 1)
                vars_ = np.var(tree_predictions, axis=0).reshape(-1, 1)
            elif self.model_name == 'TPE':
                # TPE provides mean and variance directly
                means[-1], vars_[-1] = self.target_model.predict(X_test)
            else:
                raise ValueError(f'Invalid model name: {self.model_name}')

            weights[-1] = self._target_model_weight

        weights = weights[:, :, np.newaxis]
        mean = np.sum(weights * means, axis=0)
        var = np.sum(weights ** 2 * vars_, axis=0)
        return mean, var


    def _calculate_weights(self, alpha: float = 0.0):
        if len(self._source_gps) == 0:
            self._target_model_weight = 1
            return

        if self._X is None:
            weight = 1 / len(self._source_gps)
            self._source_gp_weights = [weight for task_uid in self._source_gps]
            self._target_model_weight = 0
            return
        
        if self.sampling_mode == 'bootstrap':
            predictions = []
            for model_idx in range(len(self._source_gps)):
                model = self._source_gps[model_idx]
                # Predict based on model type
                if self.model_name == 'GP':
                    pred = model.predict(self._X)[0].flatten()
                elif self.model_name == 'RF':
                    tree_predictions = np.array([tree.predict(self._X) for tree in model.estimators_])
                    pred = np.mean(tree_predictions, axis=0)
                elif self.model_name == 'TPE':
                    pred = model.predict(self._X)[0].flatten()
                else:
                    raise ValueError(f'Invalid model name: {self.model_name}')
                
                predictions.append(pred)

            # Add target model predictions based on model type
            # if self.model_name == 'GP':
            #     loo_prediction = self.target_model.predict(self._X)[0].flatten()
            # elif self.model_name == 'RF':
            #     tree_predictions = np.array([tree.predict(self._X) for tree in self.target_model.estimators_])
            #     loo_prediction = np.mean(tree_predictions, axis=0)
            # elif self.model_name == 'TPE':
            #     loo_prediction = self.target_model.predict(self._X)[0].flatten()
            # else:
            #     raise ValueError(f'Invalid model name: {self.model_name}')

            # predictions.append(loo_prediction)
            # predictions = np.array(predictions)

            masks = np.eye(len(self._X), dtype=bool)
            train_x_cv = np.stack([self._X[~m] for m in masks])
            train_y_cv = np.stack([self._Y[~m] for m in masks])
            test_x_cv = np.stack([self._X[m] for m in masks])
            
            # Initialize model based on model type
            if self.model_name == 'GP':
                kernel = RBF(self.n_features, ARD=False)
                model = GP(copy.deepcopy(kernel), noise_variance=self._noise_variance)
            elif self.model_name == 'RF':
                model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5, min_samples_leaf=1, min_samples_split=2)
            elif self.model_name == 'TPE':
                model = TPE()
            else:
                raise ValueError(f'Invalid model name: {self.model_name}')

            loo_prediction = []
            for i in range(self._Y.shape[0]):
                model.fit(train_x_cv[i], train_y_cv[i])
                if self.model_name == 'GP':
                    loo_pred = model.predict(test_x_cv[i])[0][0][0]
                elif self.model_name == 'RF':
                    loo_pred = model.predict(test_x_cv[i])[0]
                elif self.model_name == 'TPE':
                    loo_pred = model.predict(test_x_cv[i])[0][0][0]
                else:
                    raise ValueError(f'Invalid model name: {self.model_name}')
                loo_prediction.append(loo_pred)
            predictions.append(loo_prediction)
            predictions = np.array(predictions)

            bootstrap_indices = self.rng.choice(predictions.shape[1],
                                           size=(self.n_samples, predictions.shape[1]),
                                           replace=True)

            bootstrap_predictions = []
            bootstrap_targets = self._Y[bootstrap_indices].reshape((self.n_samples, len(self._Y)))
            for m in range(len(self._source_gps) + 1):
                bootstrap_predictions.append(predictions[m, bootstrap_indices])

            ranking_losses = np.zeros((len(self._source_gps) + 1, self.n_samples))
            for i in range(len(self._source_gps)):

                for j in range(len(self._Y)):
                    ranking_losses[i] += np.sum(
                        (
                            roll_col(bootstrap_predictions[i], j) < bootstrap_predictions[i])
                            ^ (roll_col(bootstrap_targets, j) < bootstrap_targets
                        ), axis=1
                    )
            for j in range(len(self._Y)):
                ranking_losses[-1] += np.sum(
                    (
                        (roll_col(bootstrap_predictions[-1], j) < bootstrap_targets)
                        ^ (roll_col(bootstrap_targets, j) < bootstrap_targets)
                    ), axis=1
                )
        # elif self.sampling_mode in ['simplified', 'correct']:
        #     # Use the original strategy as described in v1: https://arxiv.org/pdf/1802.02219v1.pdf
        #     ranking_losses = []
        #     # compute ranking loss for each base model
        #     for model_idx in range(len(self.source_gps)):
        #         model = self.source_gps[model_idx]
        #         # compute posterior over training points for target task
        #         f_samps = sample_sobol(model, self._X, self.n_samples, self.rng.randint(10000))
        #         # compute and save ranking loss
        #         ranking_losses.append(compute_ranking_loss(f_samps, self._Y, target_model=False))
        #
        #     # compute ranking loss for target model using LOOCV
        #     if self.sampling_mode == 'simplified':
        #         # Independent draw of the leave one out sample, other "samples" are noise-free and the
        #         # actual observation
        #         f_samps = get_target_model_loocv_sample_preds(self._X, self._Y, self.n_samples, target_model,
        #                                                       self.rng.randint(10000))
        #         ranking_losses.append(compute_ranking_loss(f_samps, self._Y, target_model=True))
        #     elif self.sampling_mode == 'correct':
        #         # Joint draw of the leave one out sample and the other observations
        #         ranking_losses.append(
        #             compute_target_model_ranking_loss(train_x, train_y, num_samples, target_model,
        #                                               rng.randint(10000))
        #         )
        #     else:
        #         raise ValueError(self.sampling_mode)
        else:
            raise NotImplementedError(self.sampling_mode)

        if isinstance(self.weight_dilution_strategy, int):
            weight_dilution_percentile_target = self.weight_dilution_strategy
            weight_dilution_percentile_base = 50
        elif self.weight_dilution_strategy is None or self.weight_dilution_strategy in ['probabilistic', 'probabilistic-ld']:
            pass
        else:
            raise ValueError(self.weight_dilution_strategy)
        ranking_loss = np.array(ranking_losses)

        # perform model pruning
        p_drop = []
        if self.weight_dilution_strategy in ['probabilistic', 'probabilistic-ld']:
            for i in range(len(self._source_gps)):
                better_than_target = np.sum(ranking_loss[i, :] < ranking_loss[-1, :])
                worse_than_target = np.sum(ranking_loss[i, :] >= ranking_loss[-1, :])
                correction_term = alpha * (better_than_target + worse_than_target)
                proba_keep = better_than_target / (better_than_target + worse_than_target + correction_term)
                if self.weight_dilution_strategy == 'probabilistic-ld':
                    proba_keep = proba_keep * (1 - len(self._X) / float(self.number_of_function_evaluations))
                proba_drop = 1 - proba_keep
                p_drop.append(proba_drop)
                r = self.rng.rand()
                if r < proba_drop:
                    ranking_loss[i, :] = np.max(ranking_loss) * 2 + 1
        elif self.weight_dilution_strategy is not None:
            # Use the original strategy as described in v1: https://arxiv.org/pdf/1802.02219v1.pdf
            percentile_base = np.percentile(ranking_loss[: -1, :], weight_dilution_percentile_base, axis=1)
            percentile_target = np.percentile(ranking_loss[-1, :], weight_dilution_percentile_target)
            for i in range(len(self._source_gps)):
                if percentile_base[i] >= percentile_target:
                    ranking_loss[i, :] = np.max(ranking_loss) * 2 + 1

        # compute best model (minimum ranking loss) for each sample
        # this differs from v1, where the weight is given only to the target model in case of a tie.
        # Here, we distribute the weight fairly among all participants of the tie.
        minima = np.min(ranking_loss, axis=0)
        assert len(minima) == self.n_samples
        best_models = np.zeros(len(self._source_gps) + 1)
        for i, minimum in enumerate(minima):
            minimum_locations = ranking_loss[:, i] == minimum
            sample_from = np.where(minimum_locations)[0]

            for sample in sample_from:
                best_models[sample] += 1. / len(sample_from)

        # compute proportion of samples for which each model is best
        rank_weights = best_models / self.n_samples

        self._source_gp_weights = [rank_weights[task_uid] for task_uid in self._source_gps]
        self._target_model_weight = rank_weights[-1]
        
        # self.plot_predictions()

        return rank_weights, p_drop

    def _calculate_weights_with_no_observations(self):
        """Calculate weights according to the given start Method when no target
        task observations exist.
        """

        first, _, _ = self._start.partition("-")

        if first == "random":
            # do nothing, predict should not yet be used
            return

        if first == "mean":
            # assign equal weights to all base models
            weight = 1 / len(self._source_gps)
            self._source_gp_weights = {
                task_uid: weight for task_uid in self._source_gps
            }
            self._target_model_weight = 0
            return

        raise RuntimeError(f"Predict called without observations, first = {first}")

    def _calculate_weights_with_one_observation(self):
        """Calculate weights according to the given start Method when only one
        unique target task observation is available.
        """

        _, _, second = self._start.partition("-")

        if second == "random":
            # do nothing, predict should not be used yet
            return

        if second == "mean":
            # assign equal weights to all base models and the target model
            weight = 1 / (len(self._source_gps) + 1)
            self._source_gp_weights = {
                task_uid: weight for task_uid in self._source_gps
            }
            self._target_model_weight = weight
            return

        if second == "weighted":
            # get unique observed point
            X, indices = np.unique(self._X, axis=0, return_index=True)

            # draw _n_samples for each unique observed point from each
            # base model
            all_samples = np.empty((len(self._source_gps), self._n_samples))
            for i, task_uid in enumerate(self._source_gps):
                model = self._source_gps[task_uid]
                samples = model.sample(
                    X, size=self._n_samples, with_noise=True
                )
                all_samples[i] = samples

            # compare drawn samples to observed values
            y = self._y[indices]
            diff = np.abs(all_samples - y)

            # get base model with lowest absolute difference for each sample
            best = np.argmin(diff, axis=0)

            # compute weight as proportion of samples where a base model is best
            occurences = np.bincount(best, minlength=len(self._source_gps))
            weights = occurences / self._n_samples
            self._source_gp_weights = dict(zip(self._source_gps, weights))
            self._target_model_weight = 0
            return

        raise RuntimeError(
            f"Weight calculation with one observation, second = {second}"
        )

    def _update_meta_data(self, *gps: GPy.models.GPRegression):
        """Cache the meta data after meta training."""
        n_models = len(self._source_gps)
        for task_uid, gp in enumerate(gps):
            self._source_gps[n_models + task_uid] = gp
            self._metadata[n_models + task_uid] = {'X': self._X, 'Y': self._Y}
            
        
    def meta_update(self):
        self._update_meta_data(self.target_model)
        self._weights_need_update = True  # Set the flag to update weights after meta_update

    def set_XY(self, Data:Dict):
        self._X = copy.deepcopy(Data['X'])
        self._Y = copy.deepcopy(Data['Y'])

    def print_Weights(self):
        print(f'Source weights:{self._source_gp_weights}')
        print(f'Target weights:{self._target_model_weight}')

    def get_Weights(self):
        weights = self._source_gp_weights.copy()
        weights.append(self._target_model_weight)
        return weights


    def loss(self, task_uid: int) -> np.ndarray:
        model = self._source_gps[task_uid]
        X = self._X
        y = self._Y
        samples = model.sample(X, size=self.n_samples, with_noise=True)
        sample_comps = samples[:, np.newaxis, :] < samples
        target_comps = np.tile(y[:, np.newaxis, :] < y, self.n_samples)
        return np.sum(sample_comps ^ target_comps, axis=(1, 0))

    def posterior_samples_f(self,X, size=10, **predict_kwargs):
        """
        Samples the posterior GP at the points X.

        :param X: The points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim)
        :param size: the number of a posteriori samples.
        :type size: int.
        :returns: set of simulations
        :rtype: np.ndarray (Nnew x D x samples)
        """


        predict_kwargs["full_cov"] = True  # Always use the full covariance for posterior samples.
        m, v = self._raw_predict(X,  **predict_kwargs)

        def sim_one_dim(m, v):
            return np.random.multivariate_normal(m, v, size).T

        return sim_one_dim(m.flatten(), v)[:, np.newaxis, :]


    def posterior_samples(self, X, size=10, Y_metadata=None, likelihood=None, **predict_kwargs):
        """
        Samples the posterior GP at the points X.

        :param X: the points at which to take the samples.
        :type X: np.ndarray (Nnew x self.input_dim.)
        :param size: the number of a posteriori samples.
        :type size: int.
        :param noise_model: for mixed noise likelihood, the noise model to use in the samples.
        :type noise_model: integer.
        :returns: Ysim: set of simulations,
        :rtype: np.ndarray (D x N x samples) (if D==1 we flatten out the first dimension)
        """


        fsim = self.posterior_samples_f(X, size, **predict_kwargs)
        if likelihood is None:
            likelihood = self.likelihood
        if fsim.ndim == 3:
            for d in range(fsim.shape[1]):
                fsim[:, d] = likelihood.samples(fsim[:, d], Y_metadata=Y_metadata)
        else:
            fsim = likelihood.samples(fsim, Y_metadata=Y_metadata)
        return fsim

    def get_fmin(self):

        return np.min(self._Y)
    
    def plot_predictions(self):
        """
        Plot the data points and model predictions for 1D data.
        """
        if self.n_features != 1:
            raise ValueError("Plotting is only supported for 1-dimensional data.")

        # Generate test data points in the range 0-100
        X_test = np.linspace(0, 100, 1000).reshape(-1, 1)

        # Calculate mean and standard deviation for normalization
        y_mean = self._Y.mean()
        y_std = self._Y.std()

        # Normalize the real data points
        Y_normalized = (self._Y - y_mean) / y_std

        # Plot the normalized real data points
        plt.figure(figsize=(10, 6))
        plt.scatter(self._X, Y_normalized, color='blue', label='Normalized Real Data Points')

        # Plot the target model's prediction
        mean, _ = self.predict(X_test)
        mean_normalized = (mean - y_mean) / y_std
        plt.plot(X_test, mean_normalized, color='red', label='Target Model Prediction')

        # Adjust Y_normalized to the range of mean_normalized
        y_min, y_max = mean_normalized.min(), mean_normalized.max()
        Y_normalized_scaled = (Y_normalized - Y_normalized.min()) / (Y_normalized.max() - Y_normalized.min())
        Y_normalized_scaled = Y_normalized_scaled * (y_max - y_min) + y_min

        # Plot the adjusted normalized real data points
        plt.scatter(self._X, Y_normalized_scaled, color='green', label='Adjusted Normalized Real Data Points')

        # Plot source models' predictions if available
        if self._source_gps:
            for i, model in self._source_gps.items():
                if self.model_name == 'GP':
                    mean_source, _ = model.predict(X_test)
                elif self.model_name == 'RF':
                    tree_predictions = np.array([tree.predict(X_test) for tree in model.estimators_])
                    mean_source = np.mean(tree_predictions, axis=0).reshape(-1, 1)
                elif self.model_name == 'TPE':
                    mean_source, _ = model.predict(X_test)
                else:
                    raise ValueError(f'Invalid model name: {self.model_name}')
                
                mean_source_normalized = (mean_source - y_mean) / y_std
                plt.plot(X_test, mean_source_normalized, linestyle='--', label=f'Source Model {i} Prediction')

        plt.xlabel('X')
        plt.ylabel('Normalized Y')
        plt.title('Model Predictions and Normalized Real Data Points')
        plt.legend()
        plt.savefig(f'{self.model_name}_predictions_source{len(self._source_gps)}.png')
        plt.close()

    def plot_target_model(self):
        """
        Plot the target model's prediction and the real data points for 1D data.
        """
        if self.n_features != 1:
            raise ValueError("Plotting is only supported for 1-dimensional data.")

        # Generate test data points in the range 0-100
        X_test = np.linspace(0, 100, 1000).reshape(-1, 1)

        # Calculate mean and standard deviation for normalization

        # Plot the normalized real data points
        plt.figure(figsize=(10, 6))
        plt.scatter(self._X, self._Y, color='blue', label='Normalized Real Data Points')

        # Plot the target model's prediction
        mean, _ = self.predict(X_test)
        plt.plot(X_test, mean, color='red', label='Target Model Prediction')

        plt.xlabel('X')
        plt.ylabel('Normalized Y')
        plt.title('Target Model Prediction and Real Data Points')
        plt.legend()
        plt.savefig(f'{self.model_name}_target_model_prediction.png')