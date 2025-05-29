import copy
import os
import pickle as pkl

from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.util.general import get_quantiles

from prismbo.agent.registry import acf_registry
from prismbo.optimizer.acquisition_function.acf_base import AcquisitionBase

from external.fsaf.policies.policies import *

def load_fsaf_policy(logpath, load_iter, env, device, deterministic):
    with open(os.path.join(logpath, "params_" + str(load_iter)), "rb") as f:
        train_params = pkl.load(f)

    pi = NeuralAF(observation_space=env.observation_space,
                  action_space=env.action_space,
                  deterministic=deterministic,
                  options=train_params["policy_options"]).to(device)
    with open(os.path.join(logpath, "weights_" + str(load_iter)), "rb") as f:
        pi.load_state_dict(torch.load(f,map_location="cpu"))
    with open(os.path.join(logpath, "stats_" + str(load_iter)), "rb") as f:
        stats = pkl.load(f)

    return pi, train_params, stats


@acf_registry.register('FSAF')
class AcquisitionFSAF(AcquisitionBase):
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
        super(AcquisitionFSAF, self).__init__()
        self.config = config
        logpath = './external/fsaf/logs/PPO_2025-05-28_10-30-20'
        
        self.pi, self.policy_specs, _ = load_fsaf_policy(logpath=config['logpath'],     load_iter=config['load_iter'], env=config['env'],
                                            device="cpu", deterministic=config['deterministic'])

    def _compute_acq(self, x):

        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu_ei = s * (u * Phi + phi)

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
