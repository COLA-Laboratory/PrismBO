import numpy as np

from prismbo.optimizer.initialization.initialization_base import Sampler
from prismbo.agent.registry import sampler_registry

@sampler_registry.register("random")
class RandomSampler(Sampler):
    def sample(self, search_space, n_points = None, metadata = None, metadata_info = None):
        samples = np.zeros((self.n_samples, len(search_space.variables_order)))
        for i, name in enumerate(search_space.variables_order):
            var_range = search_space.ranges[name]
            if search_space.var_discrete[name]:  # 判断是否为离散变量
                samples[:, i] = np.random.randint(
                    var_range[0], var_range[1] + 1, size=self.n_samples
                )
            else:
                samples[:, i] = np.random.uniform(
                    var_range[0], var_range[1], size=self.n_samples
                )
        return samples
