import os
import argparse


from sklearn.metrics import mean_squared_error


from botorch.models import SingleTaskGP

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
from botorch.exceptions import OptimizationWarning
import warnings
from botorch.fit import fit_gpytorch_model
import numpy as np
import time
import json
import torch
from prismbo.benchmark.csstuning.compiler import LLVMTuning, GCCTuning
from prismbo.benchmark.csstuning.dbms import MySQLTuning
from prismbo.benchmark.csstuning.vdbms.vdbms import VDBMSTuning
from prismbo.benchmark.synthetic.singleobj import *
from prismbo.optimizer.initialization.random import RandomSampler


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist as ed
from scipy import special


DEFAULT_RIDGE = 0.01
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_MAGNITUDE = 1.0
#  Max training size in GPR model
MAX_TRAIN_SIZE = 7000
#  Batch size in GPR model
BATCH_SIZE = 3000

RESTART_FREQUENCY = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
NUM_RESTARTS = 10
RAW_SAMPLES = 512

task_class_dict = {
    'Ackley': Ackley,
    'Rastrigin': Rastrigin,
    'Rosenbrock': Rosenbrock,
    # 'XGB': XGBoostBenchmark,
    # 'HPO_PINN': HPO_PINN,
    # 'HPO_ResNet18': HPO_ResNet18,
    # 'HPO_ResNet32': HPO_ResNet32,
    'CSSTuning_GCC': GCCTuning,
    'CSSTuning_LLVM': LLVMTuning,
    'CSSTuning_MySQL': MySQLTuning,
    'CSSTuning_VDBMS': VDBMSTuning,
}


CONFIG_FILE = os.path.join("config", "running_config.json")


def read_config():
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
        return {
            'tasks': config.get('tasks'),
            'optimizer': config.get('optimizer'),
            'seeds': config.get('seeds', '42'),
            'remote': config.get('remote', False),
            'server_url': config.get('server_url', ''),
            'experimentName': config.get('experimentName', ''),
        }


class GPRResult(object):

    def __init__(self, ypreds=None, sigmas=None):
        self.ypreds = ypreds
        self.sigmas = sigmas

# numpy version of Gaussian Process Regression, not using Tensorflow
class GPRNP(object):

    def __init__(self, length_scale=1.0, magnitude=1.0, max_train_size=7000,
                 batch_size=3000, check_numerics=True, debug=False):
        assert np.isscalar(length_scale)
        assert np.isscalar(magnitude)
        assert length_scale > 0 and magnitude > 0
        self.length_scale = length_scale
        self.magnitude = magnitude
        self.max_train_size_ = max_train_size
        self.batch_size_ = batch_size
        self.check_numerics = check_numerics
        self.debug = debug
        self.X_train = None
        self.y_train = None
        self.K = None
        self.K_inv = None
        self.y_best = None

    def __repr__(self):
        rep = ""
        for k, v in sorted(self.__dict__.items()):
            rep += "{} = {}\n".format(k, v)
        return rep

    def __str__(self):
        return self.__repr__()

    def _reset(self):
        self.X_train = None
        self.y_train = None
        self.K = None
        self.K_inv = None
        self.y_best = None

    def check_X_y(self, X, y):
        from sklearn.utils.validation import check_X_y

        if X.shape[0] > self.max_train_size_:
            raise Exception("X_train size cannot exceed {} ({})"
                            .format(self.max_train_size_, X.shape[0]))
        return check_X_y(X, y, multi_output=True,
                         allow_nd=True, y_numeric=True,
                         estimator="GPRNP")

    def check_fitted(self):
        if self.X_train is None or self.y_train is None \
                or self.K is None:
            raise Exception("The model must be trained before making predictions!")

    @staticmethod
    def check_array(X):
        from sklearn.utils.validation import check_array
        return check_array(X, allow_nd=True, estimator="GPRNP")

    @staticmethod
    def check_output(X):
        finite_els = np.isfinite(X)
        if not np.all(finite_els):
            raise Exception("Input contains non-finite values: {}"
                            .format(X[~finite_els]))

    def fit(self, X_train, y_train, ridge=0.01):
        self._reset()
        X_train, y_train = self.check_X_y(X_train, y_train)
        if X_train.ndim != 2 or y_train.ndim != 2:
            raise Exception("X_train or y_train should have 2 dimensions! X_dim:{}, y_dim:{}"
                            .format(X_train.ndim, y_train.ndim))
        self.X_train = np.float32(X_train)
        self.y_train = np.float32(y_train)
        sample_size = self.X_train.shape[0]
        if np.isscalar(ridge):
            ridge = np.ones(sample_size) * ridge
        assert isinstance(ridge, np.ndarray)
        assert ridge.ndim == 1
        K = self.magnitude * np.exp(-ed(self.X_train, self.X_train) / self.length_scale) \
            + np.diag(ridge)
        K_inv = np.linalg.inv(K)
        self.K = K
        self.K_inv = K_inv
        self.y_best = np.min(y_train)
        return self

    def predict(self, X_test):
        self.check_fitted()
        if X_test.ndim != 2:
            raise Exception("X_test should have 2 dimensions! X_dim:{}"
                            .format(X_test.ndim))
        X_test = np.float32(GPRNP.check_array(X_test))
        test_size = X_test.shape[0]
        arr_offset = 0
        length_scale = self.length_scale
        yhats = np.zeros([test_size, 1])
        sigmas = np.zeros([test_size, 1])
        eips = np.zeros([test_size, 1])
        while arr_offset < test_size:
            if arr_offset + self.batch_size_ > test_size:
                end_offset = test_size
            else:
                end_offset = arr_offset + self.batch_size_
            xt_ = X_test[arr_offset:end_offset]
            K2 = self.magnitude * np.exp(-ed(self.X_train, xt_) / length_scale)#样本点与预测点的方差
            K3 = self.magnitude * np.exp(-ed(xt_, xt_) / length_scale)#预测点的方差
            K2_trans = np.transpose(K2)
            yhat = np.matmul(K2_trans, np.matmul(self.K_inv, self.y_train))#预测值
            sigma = np.sqrt(np.diag(K3 - np.matmul(K2_trans, np.matmul(self.K_inv, K2)))) \
                .reshape(xt_.shape[0], 1)#预测的标准差
            u = (self.y_best - yhat) / sigma
            phi1 = 0.5 * special.erf(u / np.sqrt(2.0)) + 0.5
            phi2 = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(np.square(u) * (-0.5))
            eip = sigma * (u * phi1 + phi2)
            yhats[arr_offset:end_offset] = yhat
            sigmas[arr_offset:end_offset] = sigma
            eips[arr_offset:end_offset] = eip
            arr_offset = end_offset
        GPRNP.check_output(yhats)
        GPRNP.check_output(sigmas)
        return GPRResult(yhats, sigmas)

    def get_params(self, deep=True):
        return {"length_scale": self.length_scale,
                "magnitude": self.magnitude,
                "X_train": self.X_train,
                "y_train": self.y_train,
                "K": self.K,
                "K_inv": self.K_inv}

    def set_params(self, **parameters):
        for param, val in list(parameters.items()):
            setattr(self, param, val)
        return self



def gp_predict(X, y, X_target):
    model = None

    model = GPRNP(length_scale=DEFAULT_LENGTH_SCALE,
                magnitude=DEFAULT_MAGNITUDE,
                max_train_size=MAX_TRAIN_SIZE,
                batch_size=BATCH_SIZE)
    model.fit(X, y, ridge=DEFAULT_RIDGE)

    return model.predict(X_target).ypreds.ravel()


def map_workload(target_X_scaled, y_target, workload_list):
    best_workload = None
    best_score = float('inf')
    scores = {}

    for name, workload_entry in workload_list.items():
        from sklearn.preprocessing import MinMaxScaler

        X_source = np.array(workload_entry['X'])
        y_source = np.array(workload_entry['y'])

        x_scaler = MinMaxScaler()
        X_source_norm = x_scaler.fit_transform(X_source)

        y_scaler = MinMaxScaler()
        y_source_norm = y_scaler.fit_transform(y_source)

        predictions = np.zeros_like(y_target)
        num_cols = y_source.shape[1] if y_source.ndim > 1 else 1
        
        for j in range(num_cols):
            y_col = y_source[:, j].reshape(-1, 1) if num_cols > 1 else y_source_norm.reshape(-1, 1)

            preds = gp_predict(X_source_norm, y_col, target_X_scaled)
            if num_cols > 1:
                predictions[:, j] = preds
            else:
                predictions[:, 0] = preds

        mse = mean_squared_error(y_target, predictions)
        scores[workload_entry.get('workload_id', 'unknown')] = mse
        if mse < best_score:
            best_score = mse
            best_workload = workload_entry.get('workload_id', workload_entry)

    logger.info('[Workload Mapping] MSE scores: {}'.format(scores))
    logger.info('[Workload Mapping] Matched Workload: {}'.format(best_workload))

    return best_workload


def propose_next_botorch(X_np, Y_np):
    X_np = np.asarray(X_np, dtype=np.float64)
    Y_np = np.asarray(Y_np, dtype=np.float64).reshape(-1, 1)

    train_x = torch.tensor(X_np, dtype=torch.double, device=device)
    train_y = torch.tensor(Y_np, dtype=torch.double, device=device)

    if train_x.ndim == 1:
        train_x = train_x.unsqueeze(0)
    if train_y.ndim == 1:
        train_y = train_y.unsqueeze(-1)

    try:
        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model = model.to(device)
        with warnings.catch_warnings():
            # 避免库抛出的无害警告打断
            warnings.simplefilter("ignore")
            mll.train()
            fit_gpytorch_model(mll)
    except Exception as e:
        # 如果拟合失败，退回随机选择
        # print("GP fit error, fallback to random:", e)
        d = X_np.shape[1]
        return np.random.rand(d).astype(np.float64)

    # 设定 best_f（用于最小化问题）
    try:
        best_f = train_y.min().item()
    except Exception:
        best_f = float(train_y[0].item())

    ei = ExpectedImprovement(model=model, best_f=best_f, maximize=False)

    d = X_np.shape[1]
    bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double, device=device)

    try:
        candidates, acq_value = optimize_acqf(
            acq_function=ei,
            bounds=bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
        )
        new_x = candidates.detach().cpu().numpy().reshape(-1)
        # clip 防止数值溢出
        new_x = np.clip(new_x, 0.0, 1.0).astype(np.float64)
        return new_x
    except Exception as e:
        # 优化失败则退化为随机搜索（但尝试避免已存在点）
        # print("Acquisition optimization failed, fallback to random:", e)
        d = X_np.shape[1]
        for _ in range(10):
            cand = np.random.rand(d)
            # 若不与已有点重复则返回
            if not any(np.allclose(cand, xx) for xx in X_np):
                return cand.astype(np.float64)
        return cand.astype(np.float64)


if __name__ == '__main__':
    
    method = 'GP_BOTORCH'
    benchmark = 'tpcc'
    tps_constraint = 700
    gp_cpu_model_dir = 'gp_model_cpu'
    gp_tps_model_dir = 'gp_model_tps'
    
    # Experiment settings
    config = read_config()
    seeds = [int(s.strip()) for s in config['seeds'].split(',')]

    # Create results directory
    results_dir = f"results/restune_{config['experimentName']}"
    os.makedirs(results_dir, exist_ok=True)

    # Run experiments
    all_results = {}
    ini_num = 18
    rnum0 = int(os.environ.get('RNUM', 2 ** 8))
    
    # Load existing workload_list if available
    workload_list_file = f"{results_dir}/workload_list.json"
    if os.path.exists(workload_list_file):
        with open(workload_list_file, 'r') as f:
            workload_list = json.load(f)
        print(f"Loaded existing workload_list with {len(workload_list)} entries from {workload_list_file}")
    else:
        workload_list = {}
        print("No existing workload_list found, starting with empty list")

    for task_info in config['tasks']:
        task_name = task_info['name']
        task_results = {}
        workloads = [int(w.strip()) for w in task_info['workloads'].split(',')]

        for workload in workloads:
            workload_results = []

            for seed in seeds:
                print(f"Running {task_name} with workload {workload}, seed {seed}")
                task = task_class_dict[task_name](
                    task_name=task_name,
                    budget_type=task_info['budget_type'],
                    budget=task_info['budget'],
                    seed=seed,
                    workload=workload,
                    description=task_info['description'],
                    params={'input_dim': int(task_info['num_vars'])}
                )
                ts = []
                start_time = time.time()
                search_space = task.get_configuration_space()
                sampler = RandomSampler(init_num=ini_num, config = {})
                ini_X = sampler.sample(search_space)
                query_datasets = [search_space.map_to_design_space(sample) for sample in ini_X]
                ini_Y = np.array([task.f(query)['f1'] for query in query_datasets]).reshape(-1, 1)

                d = len(search_space.variables_order)

                X_np = np.vstack(ini_X)          # n x d
                Y_np = np.array(ini_Y).reshape(-1, 1)  # n x 1

                # 初始化归一化器
                X_scaler = MinMaxScaler()
                Y_scaler = MinMaxScaler()
                
                # 归一化初始数据
                X_np_scaled = X_scaler.fit_transform(X_np)
                Y_np_scaled = Y_scaler.fit_transform(Y_np)

                budget = int(task_info['budget'])
                step = 0
                result = float('inf')
                best = None
                
                if workload_list:
                    print(f"[Workload Mapping] Finding best workload from {len(workload_list)} available workloads...")
                    best_workload_id = map_workload(X_np_scaled, Y_np_scaled, workload_list)
                    print(f"[Workload Mapping] Selected workload: {best_workload_id}")
                
                while len(X_np) < budget:
                    step += 1
                    step_start = time.time()
                    next_x = propose_next_botorch(X_np_scaled, Y_np_scaled)
                    
                    if any(np.allclose(next_x, x, atol=1e-6) for x in X_np_scaled):
                        next_x = np.clip(next_x + np.random.normal(scale=1e-3, size=next_x.shape), 0.0, 1.0)
                    ts.append(time.time() - start_time)
                    query = search_space.map_to_design_space(np.array(next_x))
                    eval_res = task.f(query)
                    y_val = eval_res['f1'] if isinstance(eval_res, dict) and 'f1' in eval_res else float(eval_res)

                    # 添加新点到原始数据
                    X_np = np.vstack([X_np, next_x.reshape(1, -1)])
                    Y_np = np.vstack([Y_np, np.array([[y_val]])])
                    
                    # 重新归一化所有数据（包括新点）
                    X_np_scaled = X_scaler.fit_transform(X_np)
                    Y_np_scaled = Y_scaler.fit_transform(Y_np)
            
                    if y_val < result:
                        result = y_val
                        best = next_x

                history = []
                for i, trial in enumerate(X_np):
                    history.append({
                        'iteration': i,
                        'params': trial.tolist() if trial is not None else None,
                        'loss': Y_np[i].tolist() if Y_np[i] is not None else None
                    })

                result =  {
                    'best_params': best.tolist() if best is not None else None,
                    'best_value': float(min(Y_np)) if best is not None else None,
                    'history': history,
                    'optimization_time': sum(ts)
                }

                # Save result immediately after each task completion
                task_dir = os.path.join(results_dir, task_info['name'])
                os.makedirs(task_dir, exist_ok=True)
                workload_dir = os.path.join(task_dir, f"workload_{workload}")
                os.makedirs(workload_dir, exist_ok=True)

                filename = f"{task_info['name']}_workload_{workload}_seed_{seed}.json"
                filepath = os.path.join(workload_dir, filename)

                with open(filepath, 'w') as f:
                    json.dump({
                        'task_name': task_info['name'],
                        'workload': workload,
                        'seed': seed,
                        'result': result
                    }, f, indent=2)
                
                print(f"Saved result to {filepath}")
                
                # 更新workload_list，将当前优化的结果添加到workload_list中
                workload_key = f"{task_name}_workload_{workload}_seed_{seed}"
                workload_list[workload_key] = {
                    'workload_id': workload_key,
                    'task_name': task_name,
                    'workload': workload,
                    'seed': seed,
                    'X': X_np_scaled.tolist(),
                    'y': Y_np_scaled.tolist(),
                    'X_original': X_np.tolist(),
                    'y_original': Y_np.tolist(),
                    'X_scaler_params': {
                        'scale_': X_scaler.scale_.tolist(),
                        'min_': X_scaler.min_.tolist(),
                        'data_min_': X_scaler.data_min_.tolist(),
                        'data_max_': X_scaler.data_max_.tolist(),
                        'data_range_': X_scaler.data_range_.tolist()
                    },
                    'Y_scaler_params': {
                        'scale_': Y_scaler.scale_.tolist(),
                        'min_': Y_scaler.min_.tolist(),
                        'data_min_': Y_scaler.data_min_.tolist(),
                        'data_max_': Y_scaler.data_max_.tolist(),
                        'data_range_': Y_scaler.data_range_.tolist()
                    },
                    'best_value': result,
                    'best_params': best.tolist() if best is not None else None,
                    'optimization_time': sum(ts)
                }
                print(f"[Workload Update] Added {workload_key} to workload_list")
                
                # 每次优化完成后立即保存workload_list
                workload_list_file = f"{results_dir}/workload_list.json"
                with open(workload_list_file, 'w') as f:
                    json.dump(workload_list, f, indent=2)
                print(f"[Workload Save] Saved workload_list with {len(workload_list)} entries to {workload_list_file}")
                
                workload_results.append({
                    'seed': seed,
                    'result': result
                })
            task_results[f'workload_{workload}'] = workload_results
        all_results[task_info['name']] = task_results
    
    # Save results
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save workload_list for future use
    workload_list_file = f"{results_dir}/workload_list.json"
    with open(workload_list_file, 'w') as f:
        json.dump(workload_list, f, indent=2)
    print(f"Saved workload_list with {len(workload_list)} entries to {workload_list_file}")
    
    # Print summary
    for task_name in all_results:
        print(f"\nResults for {task_name}:")
        for workload in workloads:
            workload_results = all_results[task_name][f'workload_{workload}']
            best_values = [r['result']['best_value'] for r in workload_results]
            print(f"Workload {workload}:")
            print(f"  Mean best value: {np.mean(best_values):.4f}")
            print(f"  Std best value: {np.std(best_values):.4f}")
            

    
    
    
