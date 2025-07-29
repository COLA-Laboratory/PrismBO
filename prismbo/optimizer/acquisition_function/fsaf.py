import copy
import os
import pickle as pkl

from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.util.general import get_quantiles
from GPyOpt import Design_space
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
from GPyOpt.util import epmgp

from prismbo.agent.registry import acf_registry
from prismbo.optimizer.acquisition_function.acf_base import AcquisitionBase
from external.fsaf.RL.test_recorder import BatchRecorder, Transition
from external.fsaf.policies.policies import *
from gym.envs.registration import register, registry
import gym

def load_fsaf_policy(logpath, env, device, deterministic):
    with open(os.path.join(logpath, "params"), "rb") as f:
        train_params = pkl.load(f)

    pi = NeuralAF(observation_space=env.observation_space,
                  action_space=env.action_space,
                  deterministic=deterministic,
                  options=train_params["policy_options"]).to(device)
    with open(os.path.join(logpath, "weights"), "rb") as f:
        pi.load_state_dict(torch.load(f,map_location="cpu"))
    with open(os.path.join(logpath, "stats"), "rb") as f:
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
        self.logpath = './external/model/fsaf/'
        self.device = "cuda"
        self.deterministic = True
    
    def link_space(self, space):
        opt_space = []
        for var_name in space.variables_order:
            var_dic = {
                'name': var_name,
                'type': 'continuous',
                'domain': space[var_name].search_space_range,
            }
            if space[var_name].type == 'categorical':
                var_dic['type'] = 'discrete'

            opt_space.append(var_dic.copy())
        
        self._space_prismbo = space
        self.space = Design_space(opt_space)
        self.optimizer = AcquisitionOptimizer(self.space, self.optimizer_name)
        
        dim = len(self._space_prismbo.variables_order)
        self.env_spec = {
            "env_id": "FSAF-gp-v0",
            "D": dim,
            "f_type": "GP",
            "f_opts": {
                    "bound_translation":0.1,
                    "bound_scaling":0.1,
                    "kernel": "RBF",
                    "min_regret": 1e-20,
                    "mix_kernel": False,
                    "metaTrainShot":5},
            "features": ["posterior_mean", "posterior_std", "incumbent", "timestep_perc"],
            "T": dim*11 + 50,
            "n_init_samples": dim*11,
            "pass_X_to_pi": False,
            "kernel": "RBF",
            "kernel_lengthscale": [0.175]*dim,
            "kernel_variance": 1,
            "noise_variance": 8.9e-16,
            "use_prior_mean_function": False,
            "local_af_opt": True,
                "N_MS": 10000,
                "N_S":2000,
                "N_LS": 1000,
                "k": 10,
            "reward_transformation": "neg_log10",  # true maximum not known
            "space": self._space_prismbo,
            "model": self.model,
        }
        self.env_spec_ppo = copy.deepcopy(self.env_spec)
        self.env_spec_ppo["features"] = ["posterior_mean", "posterior_std", "incumbent", "timestep_perc", "timestep", "budget"]
    
        self.n_workers = 2
        self.n_episodes = 200
        
        register(
                id=self.env_spec["env_id"],
                entry_point="external.fsaf.environment.fsaf_acf:FSAF",
                max_episode_steps=self.env_spec["T"],
                reward_threshold=None,
                kwargs=self.env_spec
        )
        self.env = gym.make("FSAF-gp-v0")
        self.pi, self.policy_specs, _ = load_fsaf_policy(logpath=self.logpath,  env=self.env, device="cuda", deterministic=True)
        self.env_seeds = 100 + np.arange(self.n_workers)

        self.latent = None
        self.theta = None
        eval_spec = {
            "env_id": self.env_spec["env_id"],
            "env_seed_offset": 100,
            "policy": 'FSAF',
            "logpath": self.logpath,
            "load_iter": 0,
            "deterministic": True,
            "policy_specs": self.policy_specs,
            "savepath": self.logpath,
            "n_workers": self.n_workers,
            "n_episodes": self.n_episodes,
            "T": self.env_spec["T"],
            "bmaml":True,
        }
        
        if "bmaml" in eval_spec and eval_spec["bmaml"]:
            with open(os.path.join(self.logpath, "theta"), "rb") as f:
                self.theta = torch.load(f,map_location="cpu").detach().cpu()
        if "mmaml" in eval_spec and eval_spec["mmaml"]:
            with open(os.path.join(self.logpath, "latent"), "rb") as f:
                self.latent = torch.load(f,map_location="cpu")
        
        self.pi.set_requires_grad(False)  # we need no gradients here

        # connect policy and environment
        self.env.unwrapped.set_af_functions(af_fun=self.pi.af)
        
        
        self.size = self.env_spec["T"]
        self.memory = []
        self.cur_size = self.size
        self.reward_sum = 0
        self.n_new = 0
        self.initial_rewards = []
        self.terminal_rewards = []
        self.next_new = None
        self.next_state = None
        self.next_value = None
        
        

    def _compute_acq(self, x):
        if self.next_state is None:
            state = self.env.reset()

        else:
            state = self.next_state
            new = self.next_new
        self.clear()
        
        if type(self.theta) == type(None):
            action, value, acqu = self.act(state)
        else:
            logits = []
            values = 0
            num_particles = len(self.theta)
            for particle_id in range(num_particles):
                names_weights_copy = self.get_inner_loop_parameter_dict(self.pi.named_parameters())
                w = self.get_weights_target_net(w_generated=self.theta, row_id=particle_id, w_target_shape=names_weights_copy)
                if self.latent == None:
                    with torch.no_grad():
                        for name,p in self.pi.named_parameters():
                            p.data.copy_(w[name])
                else:
                    index = 0
                    with torch.no_grad():
                        for name,p in self.pi.named_parameters():
                            if "policy" in name and "weight" in name and index < len(self.latent[particle_id]):
                                tmp_tau = self.latent[particle_id][index].view(-1,1)
                                tmp_tau = tmp_tau + torch.ones(tmp_tau.shape)
                                new_weight = (w[name]*tmp_tau)
                                p.data.copy_(new_weight)
                                index += 1
                            else:
                                p.data.copy_(w[name])
                _, value, acqu = self.act(state)
                logits.append(acqu[0])
                values += (value)
            logits = torch.mean(torch.stack(logits), dim=0, keepdim=True)
            values = values / num_particles
            if self.deterministic:
                action = torch.argmax(logits)
            else:
                temperature = 1/2
                prob_temp = F.softmax(logits / temperature, dim=1) 
                distr = Categorical(probs=prob_temp)
                action = distr.sample()
            action = action.squeeze(0).cpu().numpy()
        new_x = self.convert_idx_to_x(action)
        self.env.unwrapped.setAcqu(acqu)

        return new_x, acqu.cpu().numpy().T
        

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
    
    def push(self, state, action, reward, value, new):
        assert not self.is_full()
        self.memory.append(Transition(state, action, reward, value, new, None, None))
        self.reward_sum += reward
        self.n_new += int(new)
    def get_weights_target_net(self,w_generated, row_id, w_target_shape):
        w = {}
        temp = 0
        for key in w_target_shape.keys():
            w_temp = w_generated[row_id, temp:(temp + np.prod(w_target_shape[key].shape))]
            if 'b' in key:
                w[key] = w_temp
            else:
                w[key] = w_temp.view(w_target_shape[key].shape)
            temp += np.prod(w_target_shape[key].shape)

        return w
    
    def clear(self):
        self.memory = []
        self.cur_size = self.size
        self.reward_sum = 0
        self.n_new = 0
        self.initial_rewards = []
        self.terminal_rewards = []
        self.next_new = None
        self.next_state = None
        self.next_value = None
        
        
    def act(self, state):
        torch.set_num_threads(1)
        with torch.no_grad():
            # to sample the action, the policy uses the current PROCESS-local random seed, don't re-seed in pi.act
            if not self.env.unwrapped.pass_X_to_pi:
                action, value, acqu, _ = self.pi.act(torch.from_numpy(state.astype(np.float32)).to(self.device),demonstration=1)
            else:
                action, value, acqu, _ = self.pi.act(torch.from_numpy(state.astype(np.float32)),
                                            self.env.unwrapped.X,
                                            self.env.unwrapped.gp)

        action = action.cpu().numpy()
        value = value.cpu().numpy()
        if self.pi.isFSAF():
            return action, value, acqu
        else:
            return action, value

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            param_dict[name] = param

        return param_dict

    def convert_idx_to_x(self, idx):
        if not isinstance(idx, np.ndarray):
            idx = np.array([idx])
        return self.env.unwrapped.xi_t[idx, :].reshape(idx.size, self.env.unwrapped.D)
    
    
    
    def optimize(self, duplicate_manager=None):
        """
        Optimizes the acquisition function (uses a flag from the model to use gradients or not).
        """
        x, acqu = self._compute_acq(None)

        return x, acqu