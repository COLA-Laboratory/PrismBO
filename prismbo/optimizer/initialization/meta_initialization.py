import numpy as np
from scipy.stats import qmc

from prismbo.optimizer.initialization.initialization_base import Sampler
from prismbo.agent.registry import sampler_registry

@sampler_registry.register("MI")
class MetaInitialization(Sampler):
    def sample(self, search_space, metadata=None, metadata_info=None):
        # Gather sorted idx of the best Y values for each metadata
        indices_per_task = []
        X_per_task = []
        max_points = 0
        var_names = search_space.variables_order

        # Preprocess: for each dataset, get sorted indices of Y, and create numpy array of X
        for k, v in metadata.items():
            candidate_Y = v['Y'].flatten()
            sorted_indices = np.argsort(candidate_Y)
            candidate_X = np.array([[point[name] for name in var_names] for point in metadata[k]])
            indices_per_task.append(sorted_indices)
            X_per_task.append(candidate_X)
            max_points = max(max_points, len(sorted_indices))

        sample_points = []
        used = set()

        for rank in range(max_points):
            for t, indices in enumerate(indices_per_task):
                if len(sample_points) >= self.init_num:
                    break
                if rank < len(indices):
                    idx = indices[rank]
                    # 避免重复采样同一个点（如多个dataset有重复或同源点）
                    x_tuple = tuple(X_per_task[t][idx])
                    if x_tuple in used:
                        continue
                    sample_points.append(X_per_task[t][idx])
                    used.add(x_tuple)
            if len(sample_points) >= self.init_num:
                break

        # 万一不够，再补随机采样
        if len(sample_points) < self.init_num:
            left_num = self.init_num - len(sample_points)
            sampler = qmc.Sobol(d=len(var_names))
            random_points = sampler.random(n=left_num)
            for i, name in enumerate(var_names):
                var_range = search_space.ranges[name]
                random_points[:, i] = random_points[:, i] * (var_range[1] - var_range[0]) + var_range[0]
                if search_space.var_discrete[name]:
                    random_points[:, i] = np.round(random_points[:, i])
            sample_points.extend(list(random_points))

        return np.array(sample_points)[:self.init_num]
    
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