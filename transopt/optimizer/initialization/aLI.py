import numpy as np
from scipy.stats import qmc

from transopt.optimizer.initialization.initialization_base import Sampler
from transopt.agent.registry import sampler_registry

@sampler_registry.register("aLI")
class LearningInitialization(Sampler):
    def sample(self, search_space, n_points = None, metadata = None, metadata_info = None, ):
        if n_points is None:
            n_points =  len(search_space.variables_order) * 11

        sample_points = []
        best_indices = []
        
        # Get best points from each metadata
        for k, v in metadata.items():
            # Extract Y values (f1) from each data point
            candidate_Y = np.array([point['f1'] for point in v])

            # Sort indices by Y values (ascending)
            sorted_indices = np.argsort(candidate_Y.flatten())
            best_indices.append((k, sorted_indices))
            
        # Collect points until we have enough
        idx = 0  # Start with best points
        while len(sample_points) < n_points:
            for k, indices in best_indices:
                candidate_X = np.array([[point[name] for name in search_space.variables_order] for point in metadata[k]])

                if idx < len(indices):
                    # Add the point at current rank if we still need more points
                    if len(sample_points) < n_points:
                        point = candidate_X[indices[idx]]
                        sample_points.append(point)
            idx += 1  # Move to next best points if we need more
            

        return sample_points
    
