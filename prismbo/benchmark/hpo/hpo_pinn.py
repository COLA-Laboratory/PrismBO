import collections
import os
import random
import time
import json
from typing import Dict, Union, Optional, Any

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod, ABC

import matplotlib.pyplot as plt
from torch.optim import Optimizer, Adam, SGD


from prismbo.agent.registry import problem_registry
from prismbo.benchmark.problem_base.non_tab_problem import NonTabularProblem
from prismbo.space.fidelity_space import FidelitySpace
from prismbo.space.search_space import SearchSpace
from prismbo.space.variable import *
from prismbo.benchmark.hpo.pde import *
from prismbo.benchmark.hpo.pde.datasets import Datasets
from prismbo.benchmark.hpo.hpo_base import HPO_base
from prismbo.benchmark.hpo.pde import PROBLEMS, pde_classes

def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class DeviceAware(ABC):
    device: str = 'cpu'

    def __init__(self, device: Optional[str]):
        if device is None:
            device = default_device()
        self.device = device
    


class Serializable:
    """Interface for an arbitrary class being able to serialize/deserialize to/from python dictionary
    using objects that can be pickled/depickled"""

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """
        Returns: dictionary of objects to be serialized
        """
        raise NotImplementedError()

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]):
        """ Given the dictionary, write 'object' data to the appropriate places
        """
        raise NotImplementedError()


class DeviceAwareModule(nn.Module):
    device: str = 'cpu'

    def __init__(self, device: Optional[str]):
        super().__init__()
        if device is None:
            device = default_device()
        self.device = device

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        result.device = args[0]  # hack to extract device
        return result
    

class SerializableModule(DeviceAwareModule, Serializable):
    def serialize(self) -> Dict[str, Any]:
        return {
            'model': self.state_dict(),
        }

    def deserialize(self, data: Dict[str, Any]):
        self.load_state_dict(data['model'])


class PINNBase(DeviceAwareModule):
    def __init__(self, device: Optional[str] = None):
        super().__init__(device)

    @abstractmethod
    def reset(self):
        pass
    
class LearnerConfig:
    input_size: int = 2
    hidden_size: int = 128
    output_size: int = 1
    num_layer: int = 2
    activation: str = 'tanh'
    

class PINN(PINNBase):
    def __init__(self, params: LearnerConfig, device: Optional[str] = None):
        super().__init__(device)
        self.input_size = params.input_size
        self.hidden_size = params.hidden_size
        self.output_size = params.output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='linear')
        self.to(self.device)

    def forward(self, x):
        # x: (batch_size, input_size)
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.fc3(x)  # 回归输出，不加激活
        return x

    def reset(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        self.apply(weight_reset)



@problem_registry.register("HPO_PINN")
class HPO_PINN(NonTabularProblem):
    problem_type = 'hpo'
    num_variables = 4
    num_objectives = 1
    workloads = []
    fidelity = None
    
    def __init__(
        self, task_name, budget_type, budget, seed, workload, description, **kwargs
        ):            
        algorithm = 'MSE'
        architecture ='MLP'
        model_size = 2
        optimizer = kwargs.pop('optimizer', 'random')
        base_dir = kwargs.pop('base_dir', os.path.expanduser('~'))
        
        
        self.pde_name = PROBLEMS[workload]

        super(HPO_PINN, self).__init__(
            task_name=task_name,
            budget=budget,
            budget_type=budget_type,
            seed=seed,
            workload=workload,
            description=description,
        )
        
        self.query_counter = kwargs.get('query_counter', 0)
        self.hpo_optimizer = kwargs.get('optimizer', 'random')

        self.trial_seed = seed
        self.hparams = {}
        
        
        self.verbose = True
        
        self.data_dir = os.path.join(base_dir, 'prismbo_tmp/data/')
        self.model_save_dir  = os.path.join(base_dir, f'prismbo_tmp/output/models/{self.hpo_optimizer}_PINN_{self.pde_name}_{seed}/')
        self.results_save_dir  = os.path.join(base_dir, f'prismbo_tmp/output/results/{self.hpo_optimizer}_PINN_{self.pde_name}_{seed}/')
        
        print(f"Selected algorithm: {self.hpo_optimizer}, dataset: {self.pde_name}")
        
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_save_dir, exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Get the GPU ID from hparams, default to 0 if not specified
        gpu_id = kwargs.get('gpu_id', 1)
        if torch.cuda.is_available():
            # Check if the specified GPU exists
            if gpu_id < torch.cuda.device_count():
                self.device = torch.device(f"cuda:{gpu_id}")
            else:
                print(f"Warning: GPU {gpu_id} not found. Defaulting to CPU.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        self.hparams['device'] = str(self.device)
        
        print(f"Using device: {self.device}")
        
        if self.pde_name in pde_classes:
            self.pde = pde_classes[self.pde_name]()
        else:
            raise NotImplementedError(f"PDE {self.pde_name} not found")
        
        range_mu = self.pde.mu_range
        self.mu = np.random.uniform(range_mu[0], range_mu[1])
                

        self.checkpoint_vals = collections.defaultdict(lambda: [])

    # Initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    
    def save_checkpoint(self, filename):
        save_dict = {
            "model_hparams": self.hparams,
            "model_dict": self.pinn.state_dict()
        }
        torch.save(save_dict, os.path.join(self.model_save_dir, filename))


    def get_configuration_space(
        self, seed: Union[int, None] = None):
        
        hparam_space = {}
        hparam_space['lr'] = ('log', (-6, -2))
        hparam_space['weight_decay'] = ('log', (-7, -4))
        # hparam_space['momentum'] = ('float', (0.5, 0.999))
        hparam_space['batch_size'] = ('categorical', [10, 20, 40, 100])
        
        variables = []

        for name, (hparam_type, range) in hparam_space.items():
            if hparam_type == 'categorical':
                variables.append(Categorical(name, range))
            elif hparam_type == 'float':
                variables.append(Continuous(name, range))
            elif hparam_type == 'int':
                variables.append(Integer(name, range))
            elif hparam_type == 'log':
                variables.append(LogContinuous(name, range))

        ss = SearchSpace(variables)
        return ss
    
    
    def get_fidelity_space(
        self, seed: Union[int, None] = None):

        fs = FidelitySpace([
            Integer("epoch", [1, 1000])  # Adjust the range as needed
        ])
        return fs    

    def train(self, configuration: dict):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.epoches = configuration['epoch']
        self.batch_size = configuration['batch_size']
        verbose = self.verbose
        if verbose:
            print(f"Total epochs: {self.epoches}")
                
        self.train_loader = Datasets.pde_dataloader_with_mu(self.pde_name, mu=self.mu, batch_size=self.batch_size, train=True)
            
        best_epoch = 0
        # Lists to store metrics for plotting
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        
        for epoch in range(self.epoches):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            total_batches = len(self.train_loader)
            for batch in tqdm(self.train_loader, total=total_batches, desc=f"Epoch {epoch+1}/{self.epoches}", unit="batch"):
                X_f, mu_f, X_b, mu_b, X_i, mu_i = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
                
                X_f_i = X_f[0].to(self.device)
                X_b_i = X_b[0].to(self.device)
                X_i_i = X_i[0].to(self.device)
                mu = torch.tensor(self.mu).to(self.device)

                X_f_i = X_f_i.requires_grad_(True)
                X_b_i = X_b_i.requires_grad_(True)
                X_i_i = X_i_i.requires_grad_(True)
                
                
                pred_points = self.pinn(X_f_i)
                pded_residual = self.pde.pde(X_f_i, pred_points, mu)
                pde_loss = torch.mean(pded_residual ** 2)

                pred_bc = self.pinn(X_b_i)
                pred_ic = self.pinn(X_i_i)
                
                bc_loss = self.pde.bc(X_b_i, pred_bc, mu)
                ic_loss = self.pde.ic(X_i_i, pred_ic, mu)
                
                data_loss = torch.mean((pred_points - self.pde.analytic_func(X_f_i, mu).to(self.device)) ** 2)

                loss = pde_loss + bc_loss + ic_loss + data_loss

                epoch_loss += loss
                self.optim.zero_grad()
                loss.backward()
                
                self.optim.step()

            self.checkpoint_vals['step_time'].append(time.time() - epoch_start_time)

            for key, val in self.checkpoint_vals.items():
                self.checkpoint_vals[key].append(val)
                        # Store metrics for plotting
            test_loss = self.test()

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
            
                self.save_checkpoint(f"{self.filename}_model.pkl")

            # Update epoch statistics
            epoch_loss /= len(self.train_loader)
            if verbose:
                print(f"Epoch {epoch+1}/{self.epoches} - Loss: {epoch_loss:.4f}")

            # Store metrics for plotting
            train_losses.append(epoch_loss.detach().cpu().numpy())
            test_losses.append(test_loss.detach().cpu().numpy())
            
        if verbose:
            # Plot training curves
            plt.figure(figsize=(12, 5))
            
            # Plot training loss
            plt.plot(train_losses, 'b-', label='Training Loss')
            plt.plot(test_losses, 'r-', label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss vs. Epoch')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_save_dir, f"{self.filename}_training_curves.png"))
            plt.close()

        # checkpoint = torch.load(os.path.join(self.model_save_dir, f"{self.filename}_model.pkl"))
        # self.pinn.load_state_dict(checkpoint['model_dict'])

        # Calculate final results after all epochs
        results = {
            'epoch': self.epoches,
            'best_epoch': best_epoch + 1,
            'epoch_time': time.time() - epoch_start_time,
            'train_loss': float(epoch_loss.detach().cpu()),
            'test_loss': float(test_loss.detach().cpu()),
        }

        # Calculate memory usage
        results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.**3)

        results['hparams'] = self.hparams
        
        return results


    def save_epoch_results(self, results):
        epoch_path = os.path.join(self.results_save_dir, f"epoch_{results['epoch']}.json")
        with open(epoch_path, 'w') as f:
            json.dump(results, f, indent=2)
        

    def test(self):
        self.test_loader = Datasets.pde_dataloader_with_mu(self.pde_name, mu=self.mu, batch_size=2500, train=False)
        self.pinn.eval()
        test_loss = 0.0
        for batch in tqdm(self.test_loader, total=len(self.test_loader), desc=f"Testing", unit="batch"):
            X_f, mu_f, X_b, mu_b, X_i, mu_i = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
            X_f_i = X_f[0].to(self.device)

            mu = torch.tensor(self.mu).to(self.device)
            
            pred_points = self.pinn(X_f_i)
            data_loss = torch.mean((pred_points - self.pde.analytic_func(X_f_i, mu).to(self.device)) ** 2) 
            test_loss += data_loss
            
        test_loss /= len(self.test_loader)
        return test_loss
            


    def get_score(self, configuration: dict):
        for key, value in configuration.items():
            self.hparams[key] = value
        
        # Construct filename with query and all hyperparameters
        filename_parts = [f"{self.query_counter}"]
        for key, value in configuration.items():
            filename_parts.append(f"{key}_{value}")
        self.filename = "_".join(filename_parts)
        
        self.pinn = PINN(LearnerConfig(), device=self.device)
        self.pinn.to(self.device)
        self.pinn.train()
        self.optim = Adam(self.pinn.parameters(), lr=configuration['lr'], weight_decay=configuration['weight_decay'])


        self.query_counter += 1
        results = self.train(configuration)
        
        # Save results
        epochs_path = os.path.join(self.results_save_dir, f"{self.filename}.jsonl")
        with open(epochs_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save final checkpoint and mark as done
        self.save_checkpoint(f"{self.filename}_model.pkl")
        with open(os.path.join(self.model_save_dir, 'done'), 'w') as f:
            f.write('done')

        test_loss = results['test_loss']
        
        return test_loss, results
    
    

    def objective_function(
        self,
        configuration,
        fidelity = None,
        seed = None,
        **kwargs
    ) -> Dict:

        if fidelity is None:
            fidelity = {"epoch": 10}
        
        print(f'fidelity:{fidelity}')
        
        # Convert log scale values back to normal scale
        # c = self.configuration_space.map_to_design_space(configuration)
        
        # Add fidelity (epoch) to the configuration
        configuration["epoch"] = fidelity["epoch"]        
        test_loss, results = self.get_score(configuration)

        loss = {list(self.objective_info.keys())[0]: float(test_loss)}
        
        return loss
    
    def get_objectives(self) -> Dict:
        return {'function_value': 'minimize'}
    
    def get_problem_type(self):
        return "hpo"
    
    
if __name__ == "__main__":
    import torch
    import numpy as np

    # Set random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    
    # Define hyperparameters
    config = {
        'task_name': 'hpo_erm_test',
        'budget_type': 'epoch',
        'budget': 100,
        'seed': 42,
        'workload': 0,  # RobCifar10
        'architecture': 'wideresnet',
        'model_size': 28,
        'gpu_id': 0,
        'optimizer': 'none',
    }
    
    # Create HPO_ERM instance
    try:
        hpo = HPO_PINN(**config)
        
        # Get configuration space
        cs = hpo.get_configuration_space()
        
        # Example configuration (you should adjust these values based on your needs)
        test_config = {
            'lr': 0.01,
            'weight_decay': 0.0001,
            'momentum': 0.9,
            'dropout_rate': 0.3,
            'batch_size': 32,
            'epoch': 200,
            'class_balanced': True,
            'nonlinear_classifier': True
        }
        
        # Train the model and get results
        print("Starting training...")
        val_acc, results = hpo.get_score(test_config)
        
        print("\nTraining completed!")
        print(f"Validation accuracy: {val_acc:.4f}")
        print("\nTest set results:")
        for key, value in results.items():
            if key.startswith('test_'):
                print(f"{key}: {value:.4f}")
        
        print("\nModel saved at:", hpo.model_save_dir)
        print("Results saved at:", hpo.results_save_dir)
        
    except Exception as e:
        print(f"Error occurred during HPO_ERM execution: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()  # Clear GPU cache after training



