from __future__ import annotations
import numpy as np
import itertools
import time
import yaml
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from contextlib import nullcontext
from tqdm import tqdm
from ConfigSpace import hyperparameters as CSH


import torch
from torch.cuda.amp import autocast, GradScaler
from prismbo.agent.registry import pretrain_registry
from prismbo.optimizer.pretrain.pretrain_base import PretrainBase
from torch import nn
from external.pfns4bo.train import train
from external.pfns4bo.priors.utils import get_batch_to_dataloader
from external.pfns4bo import encoders
from external.pfns4bo import utils
from external.pfns4bo import bar_distribution

from external.pfns4bo import priors 
from external.pfns4bo.priors import Batch



@pretrain_registry.register("TransformerPretrain")
class TransformerPretrain(PretrainBase):
    def __init__(self, config = {}) -> None:
        super().__init__(config)
        self.train_data = None

        self.config_heboplus = {
            'priordataloader_class': priors.get_batch_to_dataloader(
                priors.get_batch_sequence(
                    priors.hebo_prior.get_batch,
                    priors.utils.sample_num_feaetures_get_batch,
                )
            ),
            'encoder_generator': encoders.get_normalized_uniform_encoder(encoders. get_variable_num_features_encoder(encoders.Linear)),
            'emsize': 512,
            'nhead': 4,
            'warmup_epochs': 5,
            'y_encoder_generator': encoders.Linear,
            'batch_size': 128,
            'scheduler': utils.get_cosine_schedule_with_warmup,
            'extra_prior_kwargs_dict': {'num_features': 18,
                        'hyperparameters': {
                        'lengthscale_concentration': 1.2106559584074301,
                        'lengthscale_rate': 1.5212245992840594,
                        'outputscale_concentration': 0.8452312502679863,
                        'outputscale_rate': 0.3993553245745406,
                        'add_linear_kernel': False,
                        'power_normalization': False,
                        'hebo_warping': False,
                        'unused_feature_likelihood': 0.3,
                        'observation_noise': True}},
            'epochs': 50,
            'lr': 0.0001,
            'bptt': 60,
            'single_eval_pos_gen': utils.get_uniform_single_eval_pos_sampler(50, min_len=1), #<function utils.get_uniform_single_eval_pos_sampler.<locals>.<lambda>()>,
            'aggregate_k_gradients': 2,
            'nhid': 1024,
            'steps_per_epoch': 1024,
            'weight_decay': 0.0,
            'train_mixed_precision': False,
            'efficient_eval_masking': True,
            'nlayers': 12}


        self.config_heboplus_userpriors = {**self.config_heboplus,
            'priordataloader_class': priors.get_batch_to_dataloader(
                                priors.get_batch_sequence(
                                    priors.hebo_prior.get_batch,
                                  priors.condition_on_area_of_opt.get_batch,
                                  priors.utils.sample_num_feaetures_get_batch
                              )),
            'style_encoder_generator': encoders.get_normalized_uniform_encoder(encoders.get_variable_num_features_encoder(encoders.Linear))
                }

        self.config_bnn = {'priordataloader_class': priors.get_batch_to_dataloader(
                priors.get_batch_sequence(
                    priors.simple_mlp.get_batch,
                priors.input_warping.get_batch,
                priors.utils.sample_num_feaetures_get_batch,
                    )
                ),
                'encoder_generator': encoders.get_normalized_uniform_encoder(encoders.get_variable_num_features_encoder(encoders.Linear)),
                'emsize': 512,
                'nhead': 4,
                'warmup_epochs': 5,
                'y_encoder_generator': encoders.Linear,
                'batch_size': 128,
                'scheduler': utils.get_cosine_schedule_with_warmup,
                'extra_prior_kwargs_dict': {'num_features': 18,
                'hyperparameters': {'mlp_num_layers': CSH.UniformIntegerHyperparameter('mlp_num_layers', 8, 15),
                'mlp_num_hidden': CSH.UniformIntegerHyperparameter('mlp_num_hidden', 36, 150),
                'mlp_init_std': CSH.UniformFloatHyperparameter('mlp_init_std',0.08896049884896237, 0.1928554813280186),
                'mlp_sparseness': 0.1449806273312999,
                'mlp_input_sampling': 'uniform',
                'mlp_output_noise': CSH.UniformFloatHyperparameter('mlp_output_noise', 0.00035983014290491186, 0.0013416342770574585),
                'mlp_noisy_targets': True,
                'mlp_preactivation_noise_std': CSH.UniformFloatHyperparameter('mlp_preactivation_noise_std',0.0003145707276259681, 0.0013753183831259406),
                'input_warping_c1_std': 0.9759720822120248,
                'input_warping_c0_std': 0.8002534583197192,
                'num_hyperparameter_samples_per_batch': 16}
                                            },
                'epochs': 50,
                'lr': 0.0001,
                # 'bptt': 60,
                'single_eval_pos_gen': utils.get_uniform_single_eval_pos_sampler(50, min_len=1), 
                'aggregate_k_gradients': 1,
                'nhid': 1024,
                'steps_per_epoch': 1024,
                'weight_decay': 0.0,
                'train_mixed_precision': True,
                'efficient_eval_masking': True,
            }
        

        
    def set_data(self, metadata, metadata_info= None):
        train_data = {}
        input_sizes = set()
        max_datasize = 0
        
        # First pass to check input sizes and find max dataset size
        for dataset_name, data in metadata.items():
            input_sizes.add(metadata_info[dataset_name]['num_variables'])
            max_datasize = max(max_datasize, len(data))
            
        # Verify all datasets have same input size
        if len(input_sizes) != 1:
            raise ValueError(f"Inconsistent input sizes across datasets: {input_sizes}")
        
        self.input_size = input_sizes.pop()
        self.max_datasize = max_datasize
        
        # Second pass to construct training data
        for dataset_name, data in metadata.items():
            objectives = metadata_info[dataset_name]["objectives"]
            obj = objectives[0]["name"]

            obj_data = [d[obj] for d in data]
            var_data = [[d[var["name"]] for var in metadata_info[dataset_name]["variables"]] for d in data]
            train_data[dataset_name] = {'X':np.array(var_data), 'y':np.array(obj_data)[:, np.newaxis]}
            
        self.train_data = train_data


    def get_ys(self, config, device='cuda:0'):
        bs = 128
        all_targets = []
        for num_hps in [2,8,12]:
            # a few different samples in case the number of features makes a difference in y dist
            b = config['priordataloader_class'].get_batch_method(bs,1000,num_hps,epoch=0,device=device,
                                                                hyperparameters=
                                                                {**config['extra_prior_kwargs_dict']['hyperparameters'],
                                                                'num_hyperparameter_samples_per_batch': -1,})




            # dummy_dl = config['priordataloader_class'](
            #     num_steps=1024,
            #     batch_size=bs,
            #     seq_len_maximum=10,  # 给个默认值
            #     device=device,
            #     **config.get('extra_prior_kwargs_dict', {})
            # )
            # b = dummy_dl.get_batch_method(bs,1000,num_hps,epoch=0,device=device,
            #                                                     hyperparameters=
            #                                                     {**config['extra_prior_kwargs_dict']['hyperparameters'],
            #                                                     'num_hyperparameter_samples_per_batch': -1,})
            
            
            
            all_targets.append(b.target_y.flatten())
            
            
            
            
        return torch.cat(all_targets,0)

    def add_criterion(self, config, device='cuda:0'):
        return {**config, 'criterion': bar_distribution.FullSupportBarDistribution(
            bar_distribution.get_bucket_limits(1000,ys=self.get_ys(config, device).cpu())
        )}


    # def get_batch_for_regression(self, hyperparameters=None, **kwargs):
        
    #     batch_size = len(self.train_data)
    #     seq_len = max([len(data['X']) for data in self.train_data.values()])
    #     num_features = self.train_data[0]['X'].shape[1]
        
    #     if hyperparameters is None:
    #         hyperparameters = {'a': 0.1, 'b': 1.0}
    #     ws = torch.distributions.Normal(torch.zeros(num_features+1), hyperparameters['b']).sample((batch_size,))

    #     xs = torch.rand(batch_size, seq_len, num_features)
    #     ys = torch.distributions.Normal(
    #         torch.einsum('nmf, nf -> nm',
    #                     torch.cat([xs,torch.ones(batch_size, seq_len,1)],2),
    #                     ws
    #                     ),
    #         hyperparameters['a']
    #     ).sample()
        
    #     # get_batch functions return two different ys, let's come back to this later, though.
    #     return Batch(x=xs.transpose(0,1), y=ys.transpose(0,1), target_y=ys.transpose(0,1))


    def train_a_pfn(self, get_batch_function,  epochs=20,):
    
        max_dataset_size = self.max_datasize
        num_features = self.input_size
        hps = self.hyperparameters
        # define a bar distribution (riemann distribution) criterion with 1000 bars
        ys = get_batch_function(max_dataset_size,20,num_features, hyperparameters=hps).target_y
        # we define our bar distribution adaptively with respect to the above sample of target ys from our prior
        criterion = bar_distribution.FullSupportBarDistribution(bar_distribution.get_bucket_limits(num_outputs=1000, ys=ys))

        # now train
        train_result = train(# the prior is the key. It defines what we train on.
                            get_batch_function, criterion=criterion,
                            # define the transformer size
                            emsize=128, nhead=4, nhid=256, nlayers=4, 
                            # how to encode the x and y inputs to the transformer
                            encoder_generator=encoders.get_normalized_uniform_encoder(encoders.Linear),
                            y_encoder_generator=encoders.Linear, 
                            # these are given to the prior, which needs to know how many features we have etc
                            extra_prior_kwargs_dict=\
                                {'num_features': num_features, 'fuse_x_y': False, 'hyperparameters': hps},
                            # change the number of epochs to put more compute into a training
                            # an epoch length is defined by `steps_per_epoch`
                            # the below means we do 10 epochs, with 100 batches per epoch and 4 datasets per batch
                            # that means we look at 10*1000*4 = 4000 datasets. Considerably less than in the demo.
                            epochs=epochs, warmup_epochs=epochs//4, steps_per_epoch=100,batch_size=8,
                            # the lr is what you want to tune! usually something in [.00005,.0001,.0003,.001] works best
                            # the lr interacts heavily with `batch_size` (smaller `batch_size` -> smaller best `lr`)
                            lr=.001,
                            # seq_len defines the size of your datasets (including the test set)
                            seq_len=max_dataset_size,
                            # single_eval_pos_gen defines where to cut off between train and test set
                            # a function that (randomly) returns lengths of the training set
                            # the below definition, will just choose the size uniformly at random up to `max_dataset_size`
                            single_eval_pos_gen=utils.get_uniform_single_eval_pos_sampler(max_dataset_size))
        return train_result 
    
    
    
    
    
    def meta_train(self):
        train(**self.add_criterion(self.config_heboplus,device='cuda:0'))
        
        










