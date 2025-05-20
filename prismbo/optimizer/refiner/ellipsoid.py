import copy
import numpy as np
from scipy.linalg import sqrtm, inv
from prismbo.optimizer.refiner.refiner_base import RefinerBase
from prismbo.agent.registry import space_refiner_registry

@space_refiner_registry.register("ellipsoid")
class EllipseRefiner(RefinerBase):
    def __init__(self, config) -> None:
        super().__init__(config)
        
    def prune(self, origin_space, current_data, metadata, metadata_info):
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
            
        min_points = np.array(min_points)
        
        # Calculate center of points
        center = np.mean(min_points, axis=0)
        
        # Center the points
        centered_points = min_points - center
        
        # Calculate covariance matrix
        cov = np.cov(centered_points.T)
        
        # Calculate the inverse square root of covariance matrix
        # This will define the shape of our ellipsoid
        try:
            A = inv(sqrtm(cov))
        except:
            # Fallback if matrix is singular
            A = np.eye(len(center))
            
        # Store the ellipsoid parameters
        self.src_A = A
        self.src_b = -A @ center
        
        print('A', self.src_A, 'b', self.src_b)
        
        return 
        
    def convert(self, X_ALL_):
        valid_idx = []
        for i in range(len(X_ALL_)):
            if np.linalg.norm(self.src_A @ X_ALL_[i] + self.src_b) <= 1:
                valid_idx.append(i)
        if len(valid_idx) == 0:
            print('[Warning] no candidates in ellipsoid area!')
            X_candidate = self.configuration_list
        else:
            X_candidate = [self.configuration_list[i] for i in valid_idx]

        assert len(X_candidate) > 0
        return X_candidate
        
