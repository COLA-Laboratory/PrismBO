import copy
import GPy

import numpy as np
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.util.general import get_quantiles
from prismbo.optimizer.model.gp import GP

from prismbo.agent.registry import acf_registry
from prismbo.optimizer.acquisition_function.acf_base import AcquisitionBase

def roll_col(X: np.ndarray, shift: int) -> np.ndarray:
    """
    Rotate columns to right by shift.
    """
    return np.concatenate((X[:, -shift:], X[:, :-shift]), axis=1)


@acf_registry.register('TAF')
class AcquisitionTAF(AcquisitionBase):
    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """
    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    def __init__(self, config):
        super(AcquisitionTAF, self).__init__()
        if 'jitter' in config:
            self.jitter = config['jitter']
        else:
            self.jitter = 0.01

        if 'threshold' in config:
            self.threshold = config['threshold']
        else:
            self.threshold = 0

        self.rng = np.random.RandomState(0)

        self.cost_withGradients = constant_cost_withGradients
        self._source_models = []
        self._source_model_weights = []
        self._target_model_weight = 1
        self.bandwidth = 1

    def _compute_acq(self, x):
        if  self.metadata is None:
            m, s = self.model.predict(x)
            fmin = self.model.get_fmin()
            phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
            f_acqu_ei = s * (u * Phi + phi)

            return f_acqu_ei
        if len(self._source_models) == 0:
            for dataset in self.metadata:
                X = dataset['X']
                Y = dataset['Y']
                # Build GP model for each dataset in metadata
                kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=1.0, lengthscale=1.0)
                gp_model = GPy.models.GPRegression(X, Y.reshape(-1,1), kernel)
                gp_model.optimize()
                
                self._source_models.append(gp_model)
                
            self._calculate_weights()
        n_sample = len(x)
        source_num = len(self._source_models)
        n_models = source_num + 1
        acf_ei = np.empty((n_models, n_sample, 1))

        for task_uid in range(source_num):
            m, s = self._source_models[task_uid].predict(x)
            _X = self._source_models[task_uid].X
            fmin = self._source_models[task_uid].predict(_X)[0].min()
            phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
            acf_ei[task_uid] =  s * (u * Phi + phi)
        m,s = self.model.predict(x)
        
        for task_uid in range(source_num):
            acf_ei[task_uid] = acf_ei[task_uid] * self._source_model_weights[task_uid]
        acf_ei[-1] = self._target_model_weight
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        acf_ei[-1] = acf_ei[-1] * (s * (u * Phi + phi))
        f_acqu_ei = np.sum(acf_ei, axis=0)

        return f_acqu_ei

    def _compute_acq_withGradients(self, x):
        # --- DEFINE YOUR AQUISITION (TO BE MAXIMIZED) AND ITS GRADIENT HERE HERE
        #
        # Compute here the value of the new acquisition function. Remember that x is a 2D  numpy array
        # with a point in the domanin in each row. f_acqu_x should be a column vector containing the
        # values of the acquisition at x. df_acqu_x contains is each row the values of the gradient of the
        # acquisition at each point of x.
        #
        # NOTE: this function is optional. If note available the gradients will be approxiamted numerically.
        raise NotImplementedError()
    


    def _calculate_weights(self, alpha: float = 0.0):
        if self.model._X is None:
            weight = 1 / len(self._source_models)
            self._source_model_weights = [weight for task_uid in self._source_models]
            self._target_model_weight = 0
            return

        predictions = []
        self.n_samples, n_features = self.model._X.shape

        for model_idx in range(len(self._source_models)):
            model = self._source_models[model_idx]
            predictions.append(model.predict(self.model._X)[0].flatten())  # ndarray(n,)


        predictions.append(self.model.predict(self.model._X)[0].flatten())
        predictions = np.array(predictions)

        bootstrap_indices = self.rng.choice(predictions.shape[1],
                                            size=(self.n_samples, predictions.shape[1]),
                                            replace=True)

        bootstrap_predictions = []
        bootstrap_targets = self.model._Y[bootstrap_indices].reshape((self.n_samples, len(self.model._Y)))
        for m in range(len(self._source_models) + 1):
            bootstrap_predictions.append(predictions[m, bootstrap_indices])

        ranking_losses = np.zeros((len(self._source_models) + 1, self.n_samples))
        for i in range(len(self._source_models)):
            for j in range(1, len(self.model._Y)):
                ranking_losses[i] += np.sum(
                    (
                        ~(roll_col(bootstrap_predictions[i], j) < bootstrap_predictions[i])
                    ^ (roll_col(bootstrap_targets, j) < bootstrap_targets)
                       ), axis=1

                )
        for j in range(1, len(self.model._Y)):
            ranking_losses[-1] += np.sum(
                (
                        ~((roll_col(bootstrap_predictions[-1], j) < bootstrap_targets)
                        ^ (roll_col(bootstrap_targets, j) < bootstrap_targets))
                ), axis=1
            )
        total_compare = len(self.model._Y) *(len(self.model._Y - 1))
        ranking_loss = np.array(ranking_losses) / total_compare

        weights = [self.Epanechnikov_kernel(ranking_loss[task_uid], ranking_loss[-1]) for task_uid in range(len(self._source_models))]
        weights.append(1.0)
        weights = np.array(weights)/np.sum(weights)
        self._source_model_weights = [weights[task_uid] for task_uid in range(len(self._source_models))]
        self._target_model_weight = weights[-1]
    

    def Epanechnikov_kernel(self, X1, X2):
        diff_matrix = X1 - X2
        u = np.linalg.norm(diff_matrix, ord=2) / self.bandwidth**2  # 计算归一化距离
        if u < 1:
            weight = 0.75 * (1 - u**2)  # 根据 Epanechnikov 核计算权重
        else:
            weight = 0 
        return weight

