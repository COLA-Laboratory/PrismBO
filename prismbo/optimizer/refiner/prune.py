
import GPy
import numpy as np
import copy
from itertools import combinations
from prismbo.optimizer.refiner.refiner_base import RefinerBase
from prismbo.agent.registry import space_refiner_registry

@space_refiner_registry.register("Prune")
class Prune(RefinerBase):
    def __init__(self, config) -> None:
        super().__init__(config)
            
    def _calculate_most_similar_sets(self, current_data, candidate_sets, n=1):
        current_X, current_Y = current_data
        
        if len(candidate_sets) < n:
            raise ValueError(f"Number of candidate sets ({len(candidate_sets)}) is less than n ({n})")
            
        scores = []
        for k, v in candidate_sets.items():
            candidate_X = v['X'] 
            candidate_Y = v['Y']
            ktrc_score = ktrc(current_X, current_Y, candidate_X, candidate_Y)
            scores.append((k, ktrc_score))
            
        # Sort by score ascending and return top n keys
        scores.sort(key=lambda x: x[1])
        selected_sets = {}
        for k, _ in scores[:n]:
            selected_sets[k] = candidate_sets[k]
        return selected_sets
        
    def prune(self, origin_space, current_data, metadata, metadata_info, radius=0.3, fraction=0.2):
        # Convert metadata to numpy arrays using variables order from original space        
        # Initialize empty arrays for each dataset
        candidate_sets = {}
        for k, v in metadata.items():
            # Extract X values in correct order from each data point
            x_array = np.array([[point[name] for name in origin_space.variables_order] for point in v])
            # Extract Y values (f1) from each data point
            y_array = np.array([point['f1'] for point in v])
            candidate_sets[k] = {'X': x_array, 'Y': y_array}
            
        most_similar_sets = self._calculate_most_similar_sets(current_data, candidate_sets)
        print('most_similar_sets: ', list(most_similar_sets.keys()))
        
        # Sample points from original space
        num_samples = int(1/fraction)
        samples = []
        for _ in range(num_samples):
            sample = []
            for var in origin_space.variables_order:
                sample.append(np.random.uniform(origin_space[var].range[0], origin_space[var].range[1]))
            samples.append(sample)
        samples = np.array(samples)
        
        # Calculate potential for each sample using GP model from most similar dataset
        current_X, current_Y = current_data
        best_x = current_X[np.argmin(current_Y)]
        # Initialize array to store potentials for each sample
        potentials = np.zeros(len(samples))

        # For each similar dataset, build a model and calculate potentials
        for set_name, data in most_similar_sets.items():
            kernel = GPy.kern.RBF(input_dim=data['X'].shape[1], variance=1.0, lengthscale=1.0)
            model = GPy.models.GPRegression(data['X'], data['Y'].reshape(-1,1), kernel)
            model.optimize()
            
            # Predict values for all samples
            pred_samples, _ = model.predict(samples)
            # Predict value for best point seen so far
            pred_best, _ = model.predict(best_x.reshape(1,-1))
            
            # Add contribution to potential for each sample
            # Potential is prediction difference from best point
            potentials += (pred_samples.flatten() - pred_best[0,0])

        # Step 5: Select nu * |Lambda_all| low-potential points
        sorted_indices = np.argsort(potentials)
        # Get the sample with highest potential as center point
        high_potential_sample = samples[sorted_indices[0]]
        
        # Create new search space centered around high potential sample
        new_space = copy.deepcopy(origin_space)
        for i, var_name in enumerate(origin_space.variables_order):
            var_config = origin_space[var_name]
            center = high_potential_sample[i]
            
            new_min = max(var_config.range[0], center - radius)
            new_max = min(var_config.range[1], center + radius)
            new_space[var_name].range = [new_min, new_max]
            
            
        return new_space


def ktrc(X1, Y1, X2, Y2):
    
    total = 0
    inconsistent_count = 0
    
    # Fit GP model on X2, Y2 using GPy
    kernel = GPy.kern.RBF(input_dim=X2.shape[1], variance=1.0, lengthscale=1.0)
    model = GPy.models.GPRegression(X2, Y2.reshape(-1,1), kernel)
    model.optimize()

    # Predict values for X1 using trained GP
    yhat_D2, _ = model.predict(X1)
    yhat_D2 = yhat_D2.flatten()

    # Predict values for X2 using same GP
    yhat_D1 = Y1

    # Get indices for all points
    Lambda_t = range(len(X1))

    for l1, l2 in combinations(Lambda_t, 2):
        d1_order = yhat_D1[l1] > yhat_D1[l2]
        d2_order = yhat_D2[l1] > yhat_D2[l2]
        if d1_order != d2_order:  # exclusive or
            inconsistent_count += 1
        total += 1

    return inconsistent_count / total if total > 0 else 0.0