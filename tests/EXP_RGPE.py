import argparse
import copy
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--benchmark',
    choices=['alpine', 'quadratic', 'adaboost', 'svm', 'openml-svm', 'openml-xgb',
             'openml-glmnet', 'nn'],
    default='alpine',
)
parser.add_argument('--task', type=int)
parser.add_argument(
    '--method',
    choices=['gpmap', 'gcp', 'random', 'rgpe', 'ablr', 'tstr', 'taf', 'wac', 'rmogp',
             'gcp+prior', 'klweighting'],
    default='random',
)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n-init', type=int, default=3)
parser.add_argument('--output-file', type=str, default=None)
parser.add_argument('--iteration-multiplier', type=int, default=1)
parser.add_argument('--empirical-meta-configs', action='store_true')
parser.add_argument('--grid-meta-configs', action='store_true')
parser.add_argument('--learned-initial-design', choices=['None', 'unscaled', 'scaled', 'copula'],
                    default='None')
parser.add_argument('--search-space-pruning', choices=['None', 'complete', 'half'], default='None')
parser.add_argument('--percent-meta-tasks', default=1.0)
parser.add_argument('--percent-meta-data', default=1.0)
args, unknown = parser.parse_known_args()


import argparse
import copy
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--benchmark',
    choices=['alpine', 'quadratic', 'adaboost', 'svm', 'openml-svm', 'openml-xgb',
             'openml-glmnet', 'nn'],
    default='alpine',
)
parser.add_argument('--task', type=int)
parser.add_argument(
    '--method',
    choices=['gpmap', 'gcp', 'random', 'rgpe', 'ablr', 'tstr', 'taf', 'wac', 'rmogp',
             'gcp+prior', 'klweighting'],
    default='random',
)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n-init', type=int, default=3)
parser.add_argument('--output-file', type=str, default=None)
parser.add_argument('--iteration-multiplier', type=int, default=1)
parser.add_argument('--empirical-meta-configs', action='store_true')
parser.add_argument('--grid-meta-configs', action='store_true')
parser.add_argument('--learned-initial-design', choices=['None', 'unscaled', 'scaled', 'copula'],
                    default='None')
parser.add_argument('--search-space-pruning', choices=['None', 'complete', 'half'], default='None')
parser.add_argument('--percent-meta-tasks', default=1.0)
parser.add_argument('--percent-meta-data', default=1.0)
args, unknown = parser.parse_known_args()

output_file = args.output_file
if output_file is not None:
    try:
        with open(output_file, 'r') as fh:
            json.load(fh)
        print('Output file %s exists - shutting down.' % output_file)
        exit(1)
    except Exception as e:
        print(e)
        pass

from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from external.rgpe.exploring_openml import SVM, XGBoost, GLMNET
from smac.optimizer.acquisition import EI
from smac.optimizer.ei_optimization import FixedSet
from smac.scenario.scenario import Scenario
from smac.facade.roar_facade import ROAR
from smac.facade.smac_bo_facade import SMAC4BO
from smac.initial_design.latin_hypercube_design import LHDesign


if __name__ == '__main__':
    pass