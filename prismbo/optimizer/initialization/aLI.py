import numpy as np
from scipy.stats import qmc
import GPy

from prismbo.optimizer.initialization.initialization_base import Sampler
from prismbo.agent.registry import sampler_registry

@sampler_registry.register("aLI")
class LearningInitialization(Sampler):
    def sample(self, search_space, metadata = None, metadata_info = None):
        if metadata is None or len(metadata) == 0:
            # If no metadata, use random sampling
            sampler = qmc.Sobol(d=len(search_space.variables_order))
            sample_points = sampler.random(n=self.init_num)
            
            # Scale points to search space
            for i, name in enumerate(search_space.variables_order):
                var_range = search_space.ranges[name]
                sample_points[:, i] = sample_points[:, i] * (var_range[1] - var_range[0]) + var_range[0]
                if search_space.var_discrete[name]:
                    sample_points[:, i] = np.round(sample_points[:, i])
            
            return sample_points

        # Initialize GP models for each metadata
        gp_models = {}
        for k, v in metadata.items():
            # Extract X and Y values
            X = metadata[k]['X']
            Y = metadata[k]['Y']
            
            # Create and fit GP model
            kernel = GPy.kern.RBF(input_dim=len(search_space.variables_order))
            model = GPy.models.GPRegression(X, Y, kernel)
            model.optimize()
            gp_models[k] = model
            
        # Generate random candidate points
        sampler = qmc.Sobol(d=len(search_space.variables_order))
        candidates = sampler.random(n=1000)
        
        # Scale candidates to search space
        for i, name in enumerate(search_space.variables_order):
            var_range = search_space.ranges[name]
            candidates[:, i] = candidates[:, i] * (var_range[1] - var_range[0]) + var_range[0]
            if search_space.var_discrete[name]:
                candidates[:, i] = np.round(candidates[:, i])
                
        # Get predictions from all models
        all_predictions = []
        for model in gp_models.values():
            mean, _ = model.predict(candidates)
            all_predictions.append(mean)
            
        # Average predictions across models
        avg_predictions = np.mean(all_predictions, axis=0)
        
        # Select points with lowest predicted values
        best_indices = np.argsort(avg_predictions.flatten())[:self.init_num]
        sample_points = candidates[best_indices]

        return sample_points
            

    
