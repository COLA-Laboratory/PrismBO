from __future__ import annotations

import itertools
import time
import yaml
from contextlib import nullcontext
from tqdm import tqdm

from prismbo.optimizer.model.pfns4bo import utils
from prismbo.optimizer.model.pfns4bo.priors import prior
from prismbo.optimizer.model.pfns4bo.transformer import TransformerModel
from prismbo.optimizer.model.pfns4bo.bar_distribution import BarDistribution, FullSupportBarDistribution, get_bucket_limits, get_custom_bar_dist


from prismbo.optimizer.model.pfns4bo.utils import get_cosine_schedule_with_warmup, get_openai_lr, StoreDictKeyPair, get_weighted_single_eval_pos_sampler, get_uniform_single_eval_pos_sampler

from prismbo.optimizer.model.pfns4bo import positional_encodings
from prismbo.optimizer.model.pfns4bo.utils import init_dist

from prismbo.optimizer.model.pfns4bo.encoders import Linear, get_normalized_uniform_encoder, get_variable_num_features_encoder

from prismbo.optimizer.model.pfns4bo.priors import get_batch_to_dataloader, get_batch_sequence, hebo_prior
from prismbo.optimizer.model.pfns4bo.priors.utils import sample_num_feaetures_get_batch


import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from prismbo.agent.registry import pretrain_registry
from prismbo.optimizer.pretrain.pretrain_base import PretrainBase

class Losses():
    gaussian = nn.GaussianNLLLoss(full=True, reduction='none')
    mse = nn.MSELoss(reduction='none')
    ce = lambda num_classes: nn.CrossEntropyLoss(reduction='none', weight=torch.ones(num_classes))
    bce = nn.BCEWithLogitsLoss(reduction='none')
    get_BarDistribution = BarDistribution
    
    
def _parse_args(config_parser, parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text



class NeuralProcessPretrain(PretrainBase):
    def __init__(self, config = {}) -> None:
        super().__init__(config)
        
        self.emsize = 512
        self.nhid = 1024
        self.nlayers = 12
        self.nhead = 4
        self.dropout = 0.0
        self.epochs = 50
        self.steps_per_epoch = 1024
        self.batch_size = 128
        self.seq_len = 10
        self.lr = 0.0001
        self.weight_decay = 0.0
        self.warmup_epochs = 5
        self.input_normalization = False
        self.y_encoder_generator = Linear
        self.pos_encoder_generator = None
        self.decoder_dict = {}
        self.extra_prior_kwargs_dict = {
            'num_features': 18,
            'hyperparameters': {
                'lengthscale_concentration': 1.2106559584074301,
                'lengthscale_rate': 1.5212245992840594,
                'outputscale_concentration': 0.8452312502679863,
                'outputscale_rate': 0.3993553245745406,
                'add_linear_kernel': False,
                'power_normalization': False,
                'hebo_warping': False,
                'unused_feature_likelihood': 0.3,
                'observation_noise': True
            }
        }
        self.scheduler = get_cosine_schedule_with_warmup
        self.load_weights_from_this_state_dict = None
        self.validation_period = 10
        self.single_eval_pos_gen = utils.get_uniform_single_eval_pos_sampler(50, min_len=1)
        self.gpu_device = 'cuda:0'
        self.aggregate_k_gradients = 2
        self.verbose = True
        self.style_encoder_generator = None
        self.epoch_callback = None
        self.step_callback = None
        self.continue_model = None
        self.initializer = None
        self.initialize_with_model = None
        self.train_mixed_precision = False
        self.efficient_eval_masking = True
        self.border_decoder = None
        self.num_global_att_tokens = 0
        self.progress_bar = False
        
        self.encoder_generator = get_normalized_uniform_encoder(get_variable_num_features_encoder(Linear))
        self.priordataloader_class = get_batch_to_dataloader(
            get_batch_sequence(
                hebo_prior.get_batch,
                sample_num_feaetures_get_batch,
            )
        )
        self.criterion = None
        
        single_eval_pos_gen = utils.get_uniform_single_eval_pos_sampler(50, min_len=1)
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
        print(f'Using {self.device} device')
        self.using_dist, self.rank, self.device = init_dist(self.device)
        single_eval_pos_gen = single_eval_pos_gen if callable(single_eval_pos_gen) else lambda: single_eval_pos_gen
        
        
        def eval_pos_seq_len_sampler():
            single_eval_pos = single_eval_pos_gen()
            return single_eval_pos, self.seq_len
        
        
        self.dl = self.priordataloader_class(num_steps=self.steps_per_epoch,
                            batch_size=self.batch_size,
                            eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
                            seq_len_maximum=self.seq_len,
                            device=self.device,
                            **self.extra_prior_kwargs_dict)
        
        
        test_batch: prior.Batch = self.dl.get_test_batch()
        style_def = test_batch.style    
        print(f'Style definition of first 3 examples: {style_def[:3] if style_def is not None else None}')
        style_encoder = self.style_encoder_generator(style_def.shape[1], self.emsize) if (style_def is not None) else None
        pos_encoder = (self.pos_encoder_generator or positional_encodings.NoPositionalEncoding)(self.emsize, self.seq_len * 2)
        if isinstance(self.criterion, nn.GaussianNLLLoss):
            self.n_out = 2
        elif isinstance(self.criterion, BarDistribution) or "BarDistribution" in self.criterion.__class__.__name__: # TODO remove this fix (only for dev)
            self.n_out = self.criterion.num_bars
        elif isinstance(self.criterion, nn.CrossEntropyLoss):
            self.n_out = self.criterion.weight.shape[0]
        else:
            self.n_out = 1
            
        if self.continue_model:
            self.model = self.continue_model
        else:
            decoder_dict = self.decoder_dict if self.decoder_dict else {'standard': (None, self.n_out)}

            decoder_once_dict = {}
            if test_batch.mean_prediction is not None:
                decoder_once_dict['mean_prediction'] = decoder_dict['standard']

            self.encoder = self.encoder_generator(self.dl.num_features, self.emsize)
            self.model = TransformerModel(encoder=self.encoder
                                    , nhead=self.nhead
                                    , ninp=self.emsize
                                    , nhid=self.nhid
                                    , nlayers=self.nlayers
                                    , dropout=self.dropout
                                    , style_encoder=style_encoder
                                    , y_encoder=self.y_encoder_generator(1, self.emsize)
                                    , input_normalization=self.input_normalization
                                    , pos_encoder=pos_encoder
                                    , decoder_dict=decoder_dict
                                    , init_method=self.initializer
                                    , efficient_eval_masking=self.efficient_eval_masking
                                    , decoder_once_dict=decoder_once_dict
                                    , num_global_att_tokens=self.num_global_att_tokens
                                    , **self.model_extra_args
                                    )
        self.model.criterion = self.criterion
        
            # learning rate
        if self.lr is None:
            self.lr = get_openai_lr(self.model)
            print(f"Using OpenAI max lr of {self.lr}.")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = self.scheduler(optimizer, self.warmup_epochs, self.epochs if self.epochs is not None else 100) # when training for fixed time lr schedule takes 100 steps

    def training_step(self):
        self.model.train()  # Turn on the train mode
        total_loss = 0.
        total_positional_losses = 0.
        total_positional_losses_recorded = 0
        nan_steps = 0
        ignore_steps = 0
        before_get_batch = time.time()
        assert len(self.dl) % self.aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
        tqdm_iter = tqdm(range(len(self.dl)), desc='Training Epoch') if self.rank==0 and self.progress_bar else None # , disable=not verbose

        for batch, full_data in enumerate(self.dl):
            data, targets, single_eval_pos = (full_data.style, full_data.x, full_data.y), full_data.target_y, full_data.single_eval_pos

            def get_metrics():
                return total_loss / self.steps_per_epoch, (
                        total_positional_losses / total_positional_losses_recorded).tolist(), \
                       time_to_get_batch, forward_time, step_time, nan_steps.cpu().item() / (batch + 1), \
                       ignore_steps.cpu().item() / (batch + 1)

            tqdm_iter.update() if tqdm_iter is not None else None
            if self.using_dist and not (batch % self.aggregate_k_gradients == self.aggregate_k_gradients - 1):
                cm = self.model.no_sync()
            else:
                cm = nullcontext()
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()
                try:
                    metrics_to_log = {}
                    with autocast(enabled=self.scaler is not None):
                        # If style is set to None, it should not be transferred to device
                        out = self.model(tuple(e.to(self.device) if torch.is_tensor(e) else e for e in data),
                                    single_eval_pos=single_eval_pos, only_return_standard_out=False)

                        # this handling is for training old models only, this can be deleted soon(ish)
                        # to only support models that return a tuple of dicts
                        out, output_once = out if isinstance(out, tuple) else (out, None)
                        output = out['standard'] if isinstance(out, dict) else out

                        forward_time = time.time() - before_forward

                        if single_eval_pos is not None:
                            targets = targets[single_eval_pos:]

                        if len(targets.shape) == len(output.shape):
                            # this implies the prior uses a trailing 1 dimesnion
                            # below we assume this not to be the case
                            targets = targets.squeeze(-1)
                        assert targets.shape == output.shape[:-1], f"Target shape {targets.shape} " \
                                                                   "does not match output shape {output.shape}"
                        if isinstance(self.criterion, nn.GaussianNLLLoss):
                            assert output.shape[-1] == 2, \
                                'need to write a little bit of code to handle multiple regression targets at once'

                            mean_pred = output[..., 0]
                            var_pred = output[..., 1].abs()
                            losses = self.criterion(mean_pred.flatten(), targets.flatten(), var=var_pred.flatten())
                        elif isinstance(self.criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                            targets[torch.isnan(targets)] = -100
                            losses = self.criterion(output.flatten(), targets.flatten())
                        elif isinstance(self.criterion, nn.CrossEntropyLoss):
                            targets[torch.isnan(targets)] = -100
                            print(f"{targets.min()=}, {targets.max()=}")
                            losses = self.criterion(output.reshape(-1, self.n_out), targets.long().flatten())
                        elif self.border_decoder is not None:
                            def apply_batch_wise_criterion(i):
                                output_, targets_, borders_ = output_adaptive[:, i], targets[:, i], borders[i]
                                criterion_ = get_custom_bar_dist(borders_, self.criterion).to(self.device)
                                return criterion_(output_, targets_)
                            output_adaptive, borders = out['adaptive_bar'], output_once['borders']
                            losses_adaptive_bar = torch.stack([apply_batch_wise_criterion(i) for i in range(output_adaptive.shape[1])], 1)
                            losses_fixed_bar = self.criterion(output, targets)
                            losses = (losses_adaptive_bar + losses_fixed_bar) / 2

                            metrics_to_log = {**metrics_to_log,
                                              **{'loss_fixed_bar': losses_fixed_bar.mean().cpu().detach().item(),
                                                 'loss_adaptive_bar': losses_adaptive_bar.mean().cpu().detach().item()}}
                        elif isinstance(self.criterion, BarDistribution) and full_data.mean_prediction:
                            assert 'mean_prediction' in output_once
                            utils.print_once('Using mean prediction for loss')
                            losses = self.criterion(output, targets, mean_prediction_logits=output_once['mean_prediction'])
                            # the mean pred loss appears as the last per sequence
                        else:
                            losses = self.criterion(output, targets)
                        losses = losses.view(-1, output.shape[1]) # sometimes the seq length can be one off
                                                                  # that is because bar dist appends the mean
                        loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
                        loss_scaled = loss / self.aggregate_k_gradients

                    if self.scaler: loss_scaled = self.scaler.scale(loss_scaled)
                    loss_scaled.backward()

                    if batch % self.aggregate_k_gradients == self.aggregate_k_gradients - 1:
                        if self.scaler: self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                        if self.scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad()

                    step_time = time.time() - before_forward

                    if not torch.isnan(loss):
                        total_loss += loss.cpu().detach().item()
                        total_positional_losses += losses.mean(1).cpu().detach() if single_eval_pos is None else \
                            nn.functional.one_hot(torch.tensor(single_eval_pos), self.seq_len)*\
                            utils.torch_nanmean(losses[:self.seq_len-single_eval_pos].mean(0)).cpu().detach()

                        total_positional_losses_recorded += torch.ones(self.seq_len) if single_eval_pos is None else \
                            nn.functional.one_hot(torch.tensor(single_eval_pos), self.seq_len)

                        metrics_to_log = {**metrics_to_log, **{f"loss": loss, "single_eval_pos": single_eval_pos}}
                        if self.step_callback is not None and self.rank == 0:
                            self.step_callback(metrics_to_log)
                        nan_steps += nan_share
                        ignore_steps += (targets == -100).float().mean()
                except Exception as e:
                    print("Invalid step encountered, skipping...")
                    print(e)
                    raise(e)

            #total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time, nan_share, ignore_share = get_metrics()
            if tqdm_iter:
                tqdm_iter.set_postfix({'data_time': time_to_get_batch, 'step_time': step_time, 'mean_loss': total_loss / (batch+1)})

            before_get_batch = time.time()
        return get_metrics()
    
    def meta_train(self):
        total_loss = float('inf')
        total_positional_losses = float('inf')
        try:
            # Initially test the epoch callback function
            if self.epoch_callback is not None and self.rank == 0:
                self.epoch_callback(self.model, 1, data_loader=self.dl, scheduler=self.scheduler)
            for epoch in (range(1, self.epochs + 1) if self.epochs is not None else itertools.count(1)):
                epoch_start_time = time.time()
                try:
                    total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time, nan_share, ignore_share =\
                        self.pretrain()
                except Exception as e:
                    print("Invalid epoch encountered, skipping...")
                    print(e)
                    raise (e)
                if hasattr(self.dl, 'validate') and epoch % self.validation_period == 0:
                    with torch.no_grad():
                        val_score = self.dl.validate(self.model)
                
                else:
                    val_score = None

                if self.verbose:
                    print('-' * 89)
                    print(
                        f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.2f} | '
                        f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, lr {self.scheduler.get_last_lr()[0]}"
                        f' data time {time_to_get_batch:5.2f} step time {step_time:5.2f}'
                        f' forward time {forward_time:5.2f}' 
                        f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
                        + (f'val score {val_score}' if val_score is not None else ''))
                    print('-' * 89)

                # stepping with wallclock time based scheduler
                if self.epoch_callback is not None and self.rank == 0:
                    self.epoch_callback(self.model, epoch, data_loader=self.dl, scheduler=self.scheduler)
                self.scheduler.step()
        except KeyboardInterrupt:
            pass

        if self.rank == 0: # trivially true for non-parallel training
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model = self.model.module
                self.dl = None
            return total_loss, total_positional_losses, self.model.to('cpu'), self.dl











