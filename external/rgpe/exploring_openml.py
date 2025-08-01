import copy
import csv
import gzip
import logging
import os
import pickle
from urllib.request import urlretrieve

import lockfile

from ConfigSpace import (
    ConfigurationSpace,
    Configuration,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    EqualsCondition,
)
import ConfigSpace.util
import numpy as np
import scipy.stats
import sklearn.compose
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.utils.validation

# import hpolib
# from hpolib.abstract_benchmark import AbstractBenchmark


__version__ = 0.2


# Differences to the paper are in comments at the end of the line
_expected_amount_of_data = {
    'GLMNET': {
        3: 15546,  # Paper says that there are 15547 entries, but one of them has a negative runtime
        31: 15528,  # Paper says that there are 15547 entries, but 19 of them have a negative runtime
        37: 15488,  # Paper says that there are 15546 entries, but 58 of them have a negative runtime
        44: 15527,  # Paper says that there are 15547 entries, but 20 of them have a negative runtime
        50: 15545,  # Paper says that there are 15547 entries, but 2 of them have a negative runtime
        151: 15547,
        312: 6613,
        333: 15441,  # Paper says that there are 15547 entries, but 105 of them have a negative runtime
        334: 15486,  # Paper says that there are 15547 entries, but 61 of them have a negative runtime
        335: 15516,  # Paper says that there are 15547 entries, but 31 of them have a negative runtime
        1036: 14937,
        1038: 15547,
        1043: 6365,  # Paper says that there are 6466 entries, but 101 of them have a negative runtime
        1046: 15462,
        1049: 7396,  # Paper says that there are 7423 entries, but 27 of them have a negative runtime
        1050: 15521,  # Paper says that there are 15521 entries, but 26 of them have a negative runtime
        1063: 15518,  # Paper says that there are 15518 entries, but 29 of them have a negative runtime
        1067: 15523,  # Paper says that there are 15518 entries, but 23 of them have a negative runtime
        1068: 15546,
        1120: 15531,
        1176: 13005,  # Paper does not mention this dataset
        1461: 6970,
        1462: 8955,
        1464: 15531, # Paper says that there are 15518 entries, but 16 of them have a negative runtime
        1467: 15387, # Paper says that there are 15518 entries, but 160 of them have a negative runtime
        1471: 15479,  # Paper says that there are 15518 entries, but 68 of them have a negative runtime
        1479: 15546,
        1480: 15000,  # Paper says that there are 15518 entries, but 24 of them have a negative runtime
        1485: 8247,
        1486: 3866,
        1487: 15543,  # Paper says that there are 15518 entries, but 4 of them have a negative runtime
        1489: 15522,  # Paper says that there are 15518 entries, but 25 of them have a negative runtime
        1494: 15515,  # Paper says that there are 15518 entries, but 15 of them have a negative runtime
        1504: 15527,  # Paper says that there are 15518 entries, but 20 of them have a negative runtime
        1510: 15406,  # Paper says that there are 15518 entries, but 141 of them have a negative runtime
        1570: 15452,  # Paper says that there are 15518 entries, but 94 of them have a negative runtime
        4134: 1493,
        4534: 2801,
    },
    'RPART': {
        3: 14624,  # Paper says that there are 14633 entries, but 9 of them have a negative runtime
        31: 14624,  # Paper says that there are 14633 entries, but 9 of them have a negative runtime
        37: 14598,  # Paper says that there are 14633 entries, but 35 of them have a negative runtime
        44: 14633,
        50: 14618,  # Paper says that there are 14633 entries, but 15 of them have a negative runtime
        151: 14632,
        312: 13455,
        333: 14585,  # Paper says that there are 14633 entries, but 47 of them have a negative runtime
        334: 14580,  # Paper says that there are 14633 entries, but 53 of them have a negative runtime
        335: 14625,  # Paper says that there are 14633 entries, but 8 of them have a negative runtime
        1036: 14633,
        1038: 5151,
        1043: 14633,
        1046: 14624,  # Paper says that there are 14633 entries, but 8 of them have a negative runtime
        1049: 14549,  # Paper says that there are 14633 entries, but 83 of them have a negative runtime
        1050: 14497,  # Paper says that there are 14633 entries, but 136 of them have a negative runtime
        1063: 14497,  # Paper says that there are 14633 entries, but 136 of them have a negative runtime
        1067: 14632,
        1068: 14633,
        1120: 7477,
        1176: 14632,  # Paper does not mention this dataset
        1461: 14073,
        1462: 14536,  # Paper says that there are 14633 entries, but 97 of them have a negative runtime
        1464: 14609,  # Paper says that there are 14633 entries, but 23 of them have a negative runtime
        1467: 14626,  # Paper says that there are 14633 entries, but 7 of them have a negative runtime
        1471: 14616,  # Paper says that there are 14633 entries, but 17 of them have a negative runtime
        1479: 14633,
        1480: 14576,  # Paper says that there are 14633 entries, but 57 of them have a negative runtime
        1485: 10923,
        1486: 11389,
        1487: 6005,
        1489: 14628,  # Paper says that there are 14633 entries, but 5 of them have a negative runtime
        1494: 14632,
        1504: 14629,  # Paper says that there are 14633 entries, but 4 of them have a negative runtime
        1510: 14561,  # Paper says that there are 14633 entries, but 72 of them have a negative runtime
        1570: 14515,  # Paper says that there are 14633 entries, but 117 of them have a negative runtime
        4134: 3947,
        4534: 3231,
    },
    'SVM': {
        3: 19644,
        31: 19644,
        37: 15985,
        44: 19644,
        50: 19644,
        151: 2384,
        312: 18740,
        333: 19634,  # Paper says that there are 19644 entries, but 10 of them have a negative runtime
        334: 19629,  # Paper says that there are 19644 entries, but 15 of them have a negative runtime
        335: 15123,
        1036: 2338,
        1038: 5716,
        1043: 10121,
        1046: 5422,
        1049: 12064,
        1050: 19644,
        1063: 19644,
        1067: 10229,
        1068: 13893,
        1120: 3908,
        1176: 14451,  # Paper does not mention this dataset
        1461: 2678,
        1462: 6320,
        1464: 19644,
        1467: 4441,
        1471: 9725,
        1479: 19644,
        1480: 19644,
        1485: 10334,
        1486: 1490,
        1487: 19644,
        1489: 17298,
        1494: 19644,
        1504: 19644,
        1510: 19644,
        1570: 19644,
        4134: 560,
        4534: 2476,
    },
    'Ranger': {
        3: 15135,  # Paper says 15139, but for 4 instances min.node.size > 1
        31: 14965,  # Paper says 15139, but for 158 instances min.node.size > 1 and for 16 instances mtry > 1
        37: 15060,  # Paper says 15139, but for 79 instances min.node.size > 1
        44: 15129,  # Paper says 15139, but for 79 instances min.node.size > 1
        50: 13357,  # Paper says 15139, but for 219 instances min.node.size > 1 and for 219 instances mtry > 1563
        151: 12381,  # Paper says 12517, but for 136 instances mtry > 1
        312: 12937,  # Paper says 12985, but for 48 instances min.node.size > 1
        333: 15066,  # Paper says 15139, but for 73 instances min.node.size > 1
        334: 14441,  # Paper says 14492, but for 51 instances min.node.size > 1
        335: 14295,  # Paper says 15139, but for 299 instances min.node.size > 1 and for 219 instances mtry > 545
        1036: 7394,  # Paper says 15139, but for 3 instances min.node.size > 1
        1038: 4827,
        1043: 3788,
        1046: 8838,  # Paper says 8842, but for 4 instances min.node.size > 1
        1049: 14819,  # Paper says 15139, but for 320 instances min.node.size > 1
        1050: 11328,  # Paper says 11357, but for 29 instances min.node.size > 1
        1063: 7883,  # Paper says 7914, but for 29 instances min.node.size > 1
        1067: 7364,  # Paper says 7386, but for 22 instances min.node.size > 1
        1068: 8135,  # Paper says 8173, but for 38 instances min.node.size > 1
        1120: 9760,
        1176: 15117,  # Paper does not mention this dataset
        1461: 14279,  # Paper says 14323, but for 44 instances mtry > 1
        1462: 15103,  # Paper says 15139, but for 36 instances min.node.size > 1
        1464: 15034,  # Paper says 15139, but for 105 instances min.node.size > 1
        1467: 14896,  # Paper says 15139, but for 243 instances min.node.size > 1
        1471: 13522,  # Paper says 13523, but for 1 instances min.node.size > 1
        1479: 15092,  # Paper says 15140, but for 48 instances min.node.size > 1
        1480: 15074,  # Paper says 15139, but for 65 instances min.node.size > 1
        1485: 15015,  # Paper says 15139, but for 124 instances min.node.size > 1
        1486: 15139,
        1487: 15108,  # Paper says 15139, but for 31 instances min.node.size > 1
        1489: 15137,  # Paper says 15139, but for 2 instances min.node.size > 1
        1494: 14807,  # Paper says 15139, but for 332 instances min.node.size > 1
        1504: 14938,  # Paper says 15140, but for 202 instances min.node.size > 1
        1510: 15071,  # Paper says 15139, but for 68 instances min.node.size > 1
        1570: 15136,  # Paper says 15139, but for 3 instances min.node.size > 1
        4134: 14472,  # Paper says 14516, but for 44 instances min.node.size > 1
        4534: 15129,  # Paper says 14516, but for 10 instances min.node.size > 1
    },
    'XGBoost': {
        3: 16867,
        31: 16867,
        37: 16866,
        44: 16867,
        50: 16866,
        151: 16272,  # should be 16866, but eta, colsample_by_tree and colsample_by_level sometimes missing
        312: 15886,
        333: 16865,  # should be 16867, but 2 samples have negative runtime
        334: 16866,  # should be 16867, but 1 sample has a negative runtime
        335: 10002,
        1036: 2581,
        1038: 1370,
        1043: 16867,
        1046: 11812,
        1049: 4453,
        1050: 13758,
        1063: 16865,  # should be 16866, but 1 sample has a negative runtime
        1067: 16866,
        1068: 16866,
        1120: 8143,
        1176: 13047,  # Paper does not mention this dataset
        1461: 2215,
        1462: 16859,  # should be 16867, but 8 sample have a negative runtime
        1464: 16865,  # should be 16867, but 2 sample have a negative runtime
        1467: 16865,  # should be 16866, but 1 sample has a negative runtime
        1471: 16866,
        1479: 16867,
        1480: 16254,
        1485: 9237,
        1486: 5813,
        1487: 11194,
        1489: 16867,
        1494: 16867,
        1504: 16867,
        1510: 16867,
        1570: 16866,  # should be 16867, but 1 sample has a negative runtime
        4134: 2222,
        4534: 947,
    },
}


class ExploringOpenML(AbstractBenchmark):
    """Surrogate benchmarks based on the data from Automatic Exploration of Machine Learning
    Benchmarks on OpenML by Kühn et al..

    This is a base class that should not be used directly. Instead, use one of the automatically
    constructed classes at the bottom of the file. This benchmark does not contain the KNN
    algorithm as it only allows for 30 different hyperparameter settings.

    Data is obtained from:
    https://figshare.com/articles/OpenML_R_Bot_Benchmark_Data_final_subset_/5882230
    """
    url = None

    def __init__(self, dataset_id, n_splits=10, n_iterations=30, rebuild=False, rng=None, n_jobs=1):
        """

        Parameters
        ----------
        dataset_id: int
            Dataset Id as given in Table 2 of the paper by Kühn et al..
        n_splits : int
            Number of cross-validation splits for optimizing the surrogate hyperparameters.
        n_iterations : int
            Number of iterations of random search to construct a surrogate model
        rebuild : bool
            Whether to construct a new surrogate model if there is already one stored to disk.
            This is important because changing the ``n_splits`` and the ``n_iterations``
            arguments do not trigger a rebuild of the surrogate.
        rng: int/None/RandomState
            set up rng
        """

        super().__init__(rng=rng)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset_id = dataset_id
        self.classifier = self.__class__.__name__.split('_')[0]
        self.n_jobs = n_jobs

        surrogate_dir = os.path.join(hpolib._config.data_dir, "ExploringOpenML", 'surrogates')
        os.makedirs(surrogate_dir, exist_ok=True)
        surrogate_file_name = os.path.join(
            surrogate_dir,
            'surrogate_%s_%d.pkl.gz' % (self.classifier, self.dataset_id),
        )
        while True:
            try:
                try:
                    with open(surrogate_file_name):
                        file_exists = True
                except FileNotFoundError:
                    file_exists = False
                if rebuild or not file_exists:
                    with lockfile.LockFile(surrogate_file_name, timeout=60):
                        self.construct_surrogate(dataset_id, n_splits, n_iterations)
                        with gzip.open(surrogate_file_name, 'wb') as fh:
                            pickle.dump(
                                (
                                    self.regressor_loss,
                                    self.regressor_runtime,
                                    self.configurations,
                                    self.features,
                                    self.targets,
                                    self.runtimes,
                                ), fh,
                            )
                    break
                else:
                    try:
                        with gzip.open(surrogate_file_name, 'rb') as fh:
                            (
                                self.regressor_loss,
                                self.regressor_runtime,
                                self.configurations,
                                self.features,
                                self.targets,
                                self.runtimes
                            ) = pickle.load(fh)
                        break
                    except:
                        with lockfile.LockFile(surrogate_file_name, timeout=60):
                            with gzip.open(surrogate_file_name, 'rb') as fh:
                                (
                                    self.regressor_loss,
                                    self.regressor_runtime,
                                    self.configurations,
                                    self.features,
                                    self.targets,
                                    self.runtimes
                                ) = pickle.load(fh)
                        break
            except lockfile.LockTimeout:
                self.logger.debug('Could not obtain file lock for %s', surrogate_file_name)

    def construct_surrogate(self, dataset_id, n_splits, n_iterations_rs):
        self.logger.info('Could not find surrogate pickle, constructing the surrogate.')

        save_to = os.path.join(hpolib._config.data_dir, "ExploringOpenML")
        if not os.path.isdir(save_to):
            os.makedirs(save_to)
        csv_path = os.path.join(save_to, self.classifier + '.csv')
        if not os.path.exists(csv_path):
            self.logger.info('Could not find surrogate data, downloading from %s', self.url)
            urlretrieve(self.url, csv_path)
            self.logger.info('Finished downloading surrogate data.')

        evaluations = []
        line_no = []

        self.logger.info('Starting to read in surrogate data.')
        with open(csv_path) as fh:
            csv_reader = csv.DictReader(fh)
            for i, line in enumerate(csv_reader):
                if int(line['data_id']) != dataset_id:
                    continue
                evaluations.append(line)
                line_no.append(i)
        hyperparameters_names = [
            hp.name for hp in self.configuration_space.get_hyperparameters()
        ]
        categorical_features = np.array([
            isinstance(
                self.configuration_space.get_hyperparameters()[i],
                CategoricalHyperparameter,
            )
            for i in range(len(self.configuration_space.get_hyperparameters()))
        ])
        target_features = 'auc'
        configurations = []
        features = []
        targets = []
        runtimes = []
        for i, evaluation in enumerate(evaluations):
            number_of_features = float(evaluation['NumberOfFeatures'])
            number_of_datapoints = float(evaluation['NumberOfInstances'])
            config = {
                key: value
                for key, value
                in evaluation.items()
                if key in hyperparameters_names and value != 'NA'
            }
            # Do some specific transformations
            if self.classifier == 'Ranger':
                config['mtry'] = float(config['mtry']) / number_of_features
                config['min.node.size'] = (
                        np.log(float(config['min.node.size']))
                        / np.log(number_of_datapoints)
                )
                if config['min.node.size'] > 1.0:
                    # MF: according to Philipp it is unclear why this is in the data
                    continue
                if config['mtry'] > 1.0:
                    # MF: according to Philipp it is unclear why this is in the data
                    continue
            elif self.classifier == 'XGBoost':
                if 'eta' not in config:
                    # MF: according to Philipp, the algorithm was run in the
                    # default and the OpenML R package did not upload the
                    # default in one of the earliest versions
                    continue
                elif 'colsample_bytree' not in config and config['booster'] == 'gbtree':
                    # MF: according to Philipp, the algorithm was run in the
                    # default and the OpenML R package did not upload the
                    # default in one of the earliest versions
                    continue
                elif 'colsample_bylevel' not in config and config['booster'] == 'gbtree':
                    # MF: according to Philipp, the algorithm was run in the
                    # default and the OpenML R package did not upload the
                    # default in one of the earliest versions
                    continue
            if float(evaluation[target_features]) > 1:
                raise ValueError(i, evaluation)
            # For unknown reasons, the runtimes can be negative...
            if float(evaluation['runtime']) < 0:
                continue
            config = ConfigSpace.util.fix_types(
                configuration=config,
                configuration_space=self.configuration_space,
            )
            try:
                config = ConfigSpace.util.deactivate_inactive_hyperparameters(
                    configuration_space=self.configuration_space,
                    configuration=config,
                )
            except ValueError as e:
                print(line_no[i], config, evaluation)
                raise e
            self.configuration_space.check_configuration(config)
            array = config.get_array()
            features.append(array)
            configurations.append(config)
            # HPOlib is about minimization!
            targets.append(1 - float(evaluation[target_features]))
            runtimes.append(float(evaluation['runtime']))

        features = np.array(features)
        targets = np.array(targets) + 1e-10
        runtimes = np.array(runtimes) + 1e-10
        features = self.impute(features)
        self.logger.info('Finished reading in surrogate data.')

        if len(configurations) != _expected_amount_of_data[self.classifier][dataset_id]:
            raise ValueError(
                'Expected %d configurations for classifier %s on dataset %d, but found only %d!' %
                (
                    _expected_amount_of_data[self.classifier][dataset_id],
                    self.classifier,
                    self.dataset_id,
                    len(configurations),
                )
            )

        self.configurations = configurations
        self.features = features
        self.targets = targets
        self.runtimes = runtimes

        self.logger.info('Start building the surrogate, this can take a few minutes...')
        cv = sklearn.model_selection.KFold(n_splits=n_splits, random_state=1, shuffle=True)
        cs = ConfigurationSpace()
        min_samples_split = UniformIntegerHyperparameter(
            'min_samples_split', lower=2, upper=20, log=True,
        )
        min_samples_leaf = UniformIntegerHyperparameter('min_samples_leaf', 1, 20, log=True)
        max_features = UniformFloatHyperparameter('max_features', 0.5, 1.0)
        bootstrap = CategoricalHyperparameter('bootstrap', [True, False])
        cs.add_hyperparameters([
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap,
        ])
        # This makes HPO deterministic
        cs.seed(1)
        highest_correlations_loss = -np.inf
        highest_correlations_loss_by_fold = np.array((n_splits,)) * -np.inf
        highest_correlations_runtime = -np.inf
        highest_correlations_runtime_by_fold = np.array((n_splits,)) * -np.inf
        best_config_loss = cs.get_default_configuration()
        best_config_runtime = cs.get_default_configuration()

        # HPO for surrogate hyperparameters
        for n_iterations in range(n_iterations_rs):
            self.logger.debug('Random search iteration %d/%d.', n_iterations, n_iterations_rs)
            check_loss = True
            new_config_loss = cs.sample_configuration()
            new_config_runtime = copy.deepcopy(new_config_loss)
            regressor_loss = self.get_unfitted_regressor(
                new_config_loss, categorical_features, 100,
            )
            regressor_runtime = self.get_unfitted_regressor(
                new_config_runtime, categorical_features, 100,
            )

            rank_correlations_loss = np.ones((n_splits, )) * -np.NaN
            rank_correlations_runtime = np.ones((n_splits, )) * -np.NaN
            for n_fold, (train_idx, test_idx) in enumerate(
                    cv.split(features, targets)
            ):
                train_features = features[train_idx]
                train_targets_loss = targets[train_idx]
                train_targets_runtime = runtimes[train_idx]

                regressor_loss.fit(train_features, np.log(train_targets_loss))
                regressor_runtime.fit(train_features, np.log(train_targets_runtime))

                test_features = features[test_idx]

                y_hat_loss = np.exp(regressor_loss.predict(test_features))
                y_hat_runtime = np.exp(regressor_runtime.predict(test_features))

                test_targets_loss = targets[test_idx]
                spearman_rank_loss = scipy.stats.spearmanr(test_targets_loss, y_hat_loss)[0]
                rank_correlations_loss[n_fold] = spearman_rank_loss

                test_targets_runtime = runtimes[test_idx]
                spearman_rank_runtime = scipy.stats.spearmanr(
                    test_targets_runtime, y_hat_runtime,
                )[0]
                rank_correlations_runtime[n_fold] = spearman_rank_runtime

                if (
                    np.nanmean(highest_correlations_loss) * 0.99
                    > np.nanmean(rank_correlations_loss)
                ) and (
                    (
                        np.nanmean(highest_correlations_loss_by_fold[: n_splits + 1])
                        * (0.99 + n_fold * 0.001)
                    )
                    > np.nanmean(rank_correlations_loss[: n_splits + 1])
                ) and (
                    np.nanmean(highest_correlations_runtime) * 0.99
                    > np.nanmean(rank_correlations_runtime)
                ) and (
                    (
                        np.nanmean(highest_correlations_runtime_by_fold[: n_splits + 1])
                        * (0.99 + n_fold * 0.001)
                    )
                    > np.nanmean(rank_correlations_runtime[: n_splits + 1])
                ):
                    check_loss = False
                    break

            if (
                check_loss
                and np.mean(rank_correlations_loss) > highest_correlations_loss
            ):
                highest_correlations_loss = np.mean(rank_correlations_loss)
                highest_correlations_loss_by_fold = rank_correlations_loss
                best_config_loss = new_config_loss
            if (
                check_loss
                and np.mean(rank_correlations_runtime) > highest_correlations_runtime
            ):
                highest_correlations_runtime = np.mean(rank_correlations_runtime)
                highest_correlations_runtime_by_fold = rank_correlations_runtime
                best_config_runtime = new_config_runtime

        # Now refit with 500 trees
        regressor_loss = self.get_unfitted_regressor(best_config_loss, categorical_features, 500)
        regressor_loss.fit(
            X=features,
            y=np.log(targets),
        )
        regressor_runtime = self.get_unfitted_regressor(
            best_config_runtime, categorical_features, 500,
        )
        regressor_runtime.fit(
            X=features,
            y=np.log(runtimes)
        )
        self.logger.info('Finished building the surrogate.')

        self.regressor_loss = regressor_loss
        self.regressor_runtime = regressor_runtime

    def get_unfitted_regressor(self, config, categorical_features, n_trees):
        return sklearn.pipeline.Pipeline([
            (
                'preproc', sklearn.compose.ColumnTransformer(
                    [
                        ('numerical', 'passthrough', ~categorical_features),
                        ('categorical', sklearn.preprocessing.OneHotEncoder(sparse=False), categorical_features,
                        )
                    ]
                )
            ),
            ('poly', sklearn.preprocessing.PolynomialFeatures(
                degree=2,
                interaction_only=True,
                include_bias=False,
            )),
            ('estimator', sklearn.ensemble.RandomForestRegressor(
                n_estimators=n_trees,
                n_jobs=self.n_jobs,
                random_state=1,
                **config.get_dictionary()
            ))
        ])

    def impute(self, features):
        features = features.copy()
        for i, hp in enumerate(self.configuration_space.get_hyperparameters()):
            nan_rows = ~np.isfinite(features[:, i])
            features[nan_rows, i] = -1
        return features

    @AbstractBenchmark._check_configuration
    def objective_function(self, x, **kwargs):
        # TODO: we should create a new configuration object with the configuration space of this
        #  benchmark to make sure we can deal with configuration space objects that were modified
        #  by the caller (for example by search space pruning). This should be done here instead
        #  of in `run_benchmark.py`
        x = x.get_array().reshape((1, -1))
        x = self.impute(x)
        y = self.regressor_loss.predict(x)
        y = y[0]
        runtime = self.regressor_runtime.predict(x)
        runtime = runtime[0]
        # Untransform and round to the resolution of the data file.
        y = np.round(np.exp(y) - 1e-10, 6)
        runtime = np.round(np.exp(runtime) - 1e-10, 6)
        return {'function_value': y, 'cost': runtime}

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, x, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_meta_information():
        return {
            'num_function_evals': 50,
            'name': 'Exploring_OpenML',
            'references': [
                """@article{kuhn_arxiv2018a,
    title = {Automatic {Exploration} of {Machine} {Learning} {Experiments} on {OpenML}},
    journal = {arXiv:1806.10961 [cs, stat]},
    author = {Daniel Kühn and Philipp Probst and Janek Thomas and Bernd Bischl},
    year = {2018},
    }""", """@inproceedings{eggensperger_aaai2015a,
   author = {Katharina Eggensperger and Frank Hutter and Holger H. Hoos and Kevin Leyton-Brown},
   title = {Efficient Benchmarking of Hyperparameter Optimizers via Surrogates},
   booktitle = {Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence},
   conference = {AAAI Conference},
   year = {2015},
}

    """
            ]
        }

    def get_empirical_f_opt(self):
        """Return the empirical f_opt.

        Because ``get_meta_information`` is a static function it has no access to the actual
        function values predicted by the surrogate. This helper function gives access.

        Returns
        -------
        Configuration
        """
        ms = []
        for t in self.regressor_loss.steps[-1][-1].estimators_:
            ms.append(np.min(t.tree_.value))
        return np.round(np.exp(np.mean(ms)) - 1e-10, 6)

    def get_empirical_f_max(self):
        """Return the empirical f_max.

        This is the configuration resulting in the worst predictive performance. Necessary to
        compute the average distance to the minimum metric typically used by Wistuba,
        Schilling and Schmidt-Thieme.

        Returns
        -------
        Configuration
        """
        ms = []
        for t in self.regressor_loss.steps[-1][-1].estimators_:
            ms.append(np.max(t.tree_.value))
        return np.round(np.exp(np.mean(ms)) - 1e-10, 6)

    def get_rs_difficulty(self, diff, n_evals, seed=None):
        rng = sklearn.utils.validation.check_random_state(seed)

        # Use the (optimistic) best value possible, but 0.5 as worst possible because an AUC of
        # 0.5 is random and we don't really care about anything below 0.5.
        f_opt = self.get_empirical_f_opt()
        f_max = 0.5
        difference = f_max - f_opt
        difference = difference if difference != 0 else 1
        cs = self.get_configuration_space()
        cs.seed(rng.randint(100000))
        configurations = cs.sample_configuration(n_evals * 100)
        scores = np.array(
            [self.objective_function(config)['function_value'] for config in configurations]
        )

        # Compute a bootstrapped score
        lower = 0
        for i in range(100000):
            subsample = rng.choice(scores, size=n_evals, replace=False)
            min_score = np.min(subsample)
            rescaled_min_score = (min_score - f_opt) / difference
            if rescaled_min_score < diff:
                lower += 1
        return lower / 100000


class GLMNET(ExploringOpenML):
    url = 'https://ndownloader.figshare.com/files/10462300'

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace()
        alpha = UniformFloatHyperparameter('alpha', lower=0, upper=1)
        lambda_ = UniformFloatHyperparameter(
            'lambda', lower=2**-10, upper=2**10, log=True,
        )
        cs.add_hyperparameters([alpha, lambda_])
        return cs


class RPART(ExploringOpenML):
    url = 'https://ndownloader.figshare.com/files/10462309'

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace()
        alpha = UniformFloatHyperparameter('cp', lower=0, upper=1)
        maxdepth = UniformIntegerHyperparameter(
            'maxdepth', lower=1, upper=30,
        )
        minbucket = UniformIntegerHyperparameter(
            'minbucket', lower=1, upper=60,
        )
        minsplit = UniformIntegerHyperparameter(
            'minsplit', lower=1, upper=60,
        )
        cs.add_hyperparameters([alpha, maxdepth, minbucket, minsplit])
        return cs


class SVM(ExploringOpenML):
    url = 'https://ndownloader.figshare.com/files/10462312'

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace()
        kernel = CategoricalHyperparameter(
            'kernel', choices=['linear', 'polynomial', 'radial'],
        )
        cost = UniformFloatHyperparameter('cost', 2**-10, 2**10, log=True)
        gamma = UniformFloatHyperparameter('gamma', 2**-10, 2**10, log=True)
        degree = UniformIntegerHyperparameter('degree', 2, 5)
        cs.add_hyperparameters([kernel, cost, gamma, degree])
        gamma_condition = EqualsCondition(gamma, kernel, 'radial')
        degree_condition = EqualsCondition(degree, kernel, 'polynomial')
        cs.add_conditions([gamma_condition, degree_condition])
        return cs


class Ranger(ExploringOpenML):

    url = 'https://ndownloader.figshare.com/files/10462306'

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace()
        num_trees = UniformIntegerHyperparameter(
            'num.trees', lower=1, upper=2000,
        )
        replace = CategoricalHyperparameter(
            'replace', choices=['FALSE', 'TRUE'],
        )
        sample_fraction = UniformFloatHyperparameter(
            'sample.fraction', lower=0, upper=1,
        )
        mtry = UniformFloatHyperparameter(
            'mtry', lower=0, upper=1,
        )
        respect_unordered_factors = CategoricalHyperparameter(
            'respect.unordered.factors', choices=['FALSE', 'TRUE'],
        )
        min_node_size = UniformFloatHyperparameter(
            'min.node.size', lower=0, upper=1,
        )
        cs.add_hyperparameters([
            num_trees, replace, sample_fraction, mtry,
            respect_unordered_factors, min_node_size,
        ])
        return cs


class XGBoost(ExploringOpenML):
    url = 'https://ndownloader.figshare.com/files/10462315'

    @staticmethod
    def get_configuration_space():
        cs = ConfigurationSpace()
        nrounds = UniformIntegerHyperparameter(
            'nrounds', lower=1, upper=5000,
        )
        eta = UniformFloatHyperparameter(
            'eta', lower=2**-10, upper=2**0, log=True,
        )
        subsample = UniformFloatHyperparameter(
            'subsample', lower=0, upper=1,
        )
        booster = CategoricalHyperparameter(
            'booster', choices=['gblinear', 'gbtree'],
        )
        max_depth = UniformIntegerHyperparameter(
            'max_depth', lower=1, upper=15,
        )
        min_child_weight = UniformFloatHyperparameter(
            'min_child_weight', lower=2**0, upper=2**7, log=True,
        )
        colsample_bytree = UniformFloatHyperparameter(
            'colsample_bytree', lower=0, upper=1,
        )
        colsample_bylevel = UniformFloatHyperparameter(
            'colsample_bylevel', lower=0, upper=1,
        )
        lambda_ = UniformFloatHyperparameter(
            'lambda', lower=2**-10, upper=2**10, log=True,
        )
        alpha = UniformFloatHyperparameter(
            'alpha', lower=2**-10, upper=2**10, log=True,
        )
        cs.add_hyperparameters([
            nrounds, eta, subsample, booster, max_depth, min_child_weight,
            colsample_bytree, colsample_bylevel, lambda_, alpha,
        ])
        colsample_bylevel_condition = EqualsCondition(
            colsample_bylevel, booster, 'gbtree',
        )
        colsample_bytree_condition = EqualsCondition(
            colsample_bytree, booster, 'gbtree',
        )
        max_depth_condition = EqualsCondition(
            max_depth, booster, 'gbtree',
        )
        min_child_weight_condition = EqualsCondition(
            min_child_weight, booster, 'gbtree',
        )
        cs.add_conditions([
            colsample_bylevel_condition, colsample_bytree_condition,
            max_depth_condition, min_child_weight_condition,
        ])
        return cs


all_datasets = [
    3, 31, 37, 44, 50, 151, 312, 333, 334, 335, 1036, 1038, 1043, 1046, 1049,
    1050, 1063, 1067, 1068, 1120, 1461, 1462, 1464, 1467, 1471, 1479, 1480,
    1485, 1486, 1487, 1489, 1494, 1504, 1510, 1570, 4134, 4534,
]
all_model_classes = [
    GLMNET,
    RPART,
    SVM,
    Ranger,
    XGBoost,
]

for model_class in all_model_classes:
    for dataset_id_ in all_datasets:
        benchmark_string = """class %s_%d(%s):
         def __init__(self, n_splits=10, n_iterations=30, rng=None, n_jobs=1):
             super().__init__(dataset_id=%d, n_splits=n_splits, n_iterations=n_iterations, rebuild=False, rng=rng, n_jobs=n_jobs)
    """ % (model_class.__name__, dataset_id_, model_class.__name__, dataset_id_)

        exec(benchmark_string)


if __name__ == '__main__':
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    # Call this script to construct all surrogates
    for model_class in all_model_classes:
        print(model_class)
        for dataset_id_ in all_datasets:
            print(dataset_id_)
            exec('rval = %s_%d(n_jobs=-1)' % (model_class.__name__, dataset_id_))
            print(rval)

            model_class_cs = rval.get_configuration_space()
            for _ in range(10):
                tmp_config = model_class_cs.sample_configuration()
                print(tmp_config, rval.objective_function(tmp_config))
