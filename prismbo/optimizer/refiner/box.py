import numpy as np
import copy
from prismbo.optimizer.refiner.refiner_base import RefinerBase
from prismbo.agent.registry import space_refiner_registry

@space_refiner_registry.register("box")
class BoxRefiner(RefinerBase):
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def prune(self, origin_space, current_data, metadata, metadata_info):
        """
        Compute the tight bounding box (low-volume search space) for a set of optimal solutions.

        Parameters:
            x_stars (np.ndarray): A T x p array of T optimal solutions in R^p.
        
        Returns:
            l_star (np.ndarray): Lower bound of the bounding box (1D array of length p).
            u_star (np.ndarray): Upper bound of the bounding box (1D array of length p).
        """
        if len(metadata) <= 1:
            return origin_space
        new_space = copy.deepcopy(origin_space)
        # Initialize arrays to store min and max values
        min_points = []
        
        # Iterate through each dataset
        for k, v in metadata.items():
            # Extract x values for points with minimum f1 value
            min_f1_idx = np.argmin([point['f1'] for point in v])
            min_point = [v[min_f1_idx][var['name']] for var in metadata_info[k]['variables']]
            min_points.append(min_point)
            
        # Convert to numpy arrays and find overall bounds
        min_points = np.array(min_points)
        
        l_star = np.min(min_points, axis=0)
        u_star = np.max(min_points, axis=0)
        
        # Update search space bounds
        for i, var in enumerate(origin_space.variables_order):
            new_space.update_range(var, (l_star[i]-1e-3, u_star[i]+1e-3))

        return new_space



    
    
    

