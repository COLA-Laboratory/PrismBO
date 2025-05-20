import numpy as np
from scipy.stats import qmc

from transopt.optimizer.initialization.initialization_base import Sampler
from transopt.agent.registry import sampler_registry

@sampler_registry.register("MI")
class MetaInitialization(Sampler):
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
    
    def negative_spearman_correlation(self, metadata, metadata_info, p=2):
        """
        Calculate the negative Spearman correlation coefficient between ranked results of hyperparameter configurations.
        
        Args:
            metadata: Dictionary containing datasets with their configurations and results
            metadata_info: Dictionary containing metadata information
            p: The order of the norm (default=2)
            
        Returns:
            Dictionary of pairwise distances between datasets
        """
        distances = {}
        dataset_keys = list(metadata.keys())
        
        for i in range(len(dataset_keys)):
            for j in range(i+1, len(dataset_keys)):
                key_i = dataset_keys[i]
                key_j = dataset_keys[j]
                
                # Get results for each dataset
                results_i = np.array([point['f1'] for point in metadata[key_i]])
                results_j = np.array([point['f1'] for point in metadata[key_j]])
                
                # Calculate ranks
                rank_i = np.argsort(np.argsort(results_i))
                rank_j = np.argsort(np.argsort(results_j))
                
                # Calculate Spearman correlation
                n = len(rank_i)
                if n != len(rank_j):
                    n = min(len(rank_i), len(rank_j))
                    rank_i = rank_i[:n]
                    rank_j = rank_j[:n]
                    
                correlation = np.corrcoef(rank_i, rank_j)[0,1]
                
                # Store negative correlation as distance
                distances[(key_i, key_j)] = 1 - correlation
                
        return distances
    
    def random_forest_distances(self, metadata, metadata_info, n_estimators=100):
        """
        Calculate distances between datasets using Random Forest predictions on meta-features.
        
        Args:
            metadata: Dictionary containing datasets with their configurations and results
            metadata_info: Dictionary containing metadata information
            n_estimators: Number of trees in the random forest
            
        Returns:
            Dictionary of pairwise distances between datasets based on RF predictions
        """
        from sklearn.ensemble import RandomForestRegressor
        
        distances = {}
        dataset_keys = list(metadata.keys())
        
        # Extract meta-features and results for each dataset
        meta_features = {}
        results = {}
        for key in dataset_keys:
            # Get meta-features from metadata_info
            meta_features[key] = np.array([
                metadata_info[key].get('mean', 0),
                metadata_info[key].get('std', 0),
                metadata_info[key].get('skewness', 0),
                metadata_info[key].get('kurtosis', 0)
            ])
            
            # Get results
            results[key] = np.array([point['f1'] for point in metadata[key]])
            
        # Calculate distances between each pair of datasets
        for i in range(len(dataset_keys)):
            for j in range(i+1, len(dataset_keys)):
                key_i = dataset_keys[i]
                key_j = dataset_keys[j]
                
                # Train RF on dataset i
                rf_i = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                rf_i.fit(meta_features[key_i].reshape(1,-1), results[key_i])
                
                # Train RF on dataset j  
                rf_j = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                rf_j.fit(meta_features[key_j].reshape(1,-1), results[key_j])
                
                # Make predictions
                pred_i = rf_i.predict(meta_features[key_j].reshape(1,-1))
                pred_j = rf_j.predict(meta_features[key_i].reshape(1,-1))
                
                # Calculate distance as mean squared error between predictions
                distance = np.mean((pred_i - results[key_j])**2 + (pred_j - results[key_i])**2)
                distances[(key_i, key_j)] = distance
                
        return distances