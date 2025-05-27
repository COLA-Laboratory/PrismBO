import numpy as np
from scipy.stats import qmc

from prismbo.optimizer.initialization.initialization_base import Sampler
from prismbo.agent.registry import sampler_registry

@sampler_registry.register("sobol")
class SobolSampler(Sampler):
    def sample(self, search_space, n_points = None, metadata = None, metadata_info = None):
        self.n_samples = n_points
        d = len(search_space.variables_order)
        sampler = qmc.Sobol(d=d, scramble=True)
        sample_points = sampler.random(n=self.n_samples)
        sample_points = sample_points.reshape(self.n_samples, d)
        for i, name in enumerate(search_space.variables_order):
            var_range = search_space.ranges[name]
            if search_space.var_discrete[name]:
                # 对离散变量进行处理
                continuous_vals = qmc.scale(
                    sample_points[:, i:i+1], var_range[0], var_range[1]
                )
                sample_points[:, i] = np.round(continuous_vals).astype(int).flatten()
            else:
                sample_points[:, i] = qmc.scale(
                    sample_points[:, i:i+1], var_range[0], var_range[1]
                ).flatten()
        return sample_points
