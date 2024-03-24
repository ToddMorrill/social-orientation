"""Trainer class for PyTorch models.
"""
import time
import logconfig
import logging
import os
import json
from args import parse_args

import numpy as np
import torch

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Trainer as HFTrainer
from tqdm import tqdm
import wandb

from utils import get_optimizer, log, get_checkpoint


class Trainer():

    def __init__(self,
                 args,
                 model,
                 train_loader,
                 val_loader=None,
                 test_loader=None,
                 callbacks=[],
                 metrics=[],
                 custom_loss_fn=None,
                 tokenizer=None):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.callbacks = callbacks
        self.metric_objects = metrics
        self.custom_loss_fn = custom_loss_fn
        self.tokenizer = tokenizer # not used in class but useful to include in trainer for saving to disk
        self.should_stop = False
        self.gradient_accumulation_steps = 1
        if 'gradient_accumulation_steps' in args and isinstance(args.gradient_accumulation_steps, int):
            self.gradient_accumulation_steps = args.gradient_accumulation_steps
        
        # use a fresh scaler for mixed precision training for each training run
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-gradscaler
        # if args.fp16 == False, self.scaler will be a no-op 
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        
        # record what splits are available
        self.splits = ['train']
        if self.val_loader is not None:
            self.splits.append('val')
        self.device = args.device
        self.epochs = args.epochs

        self.configure_optimizer()
        # global_step is the number of optimization steps
        self.global_step = 0 
        # start at 1 to avoid evaluating and saving the model before training
        # training_step is the number of training batches seen
        self.training_step = 1
        self.epoch = 0
        self.metrics = {}

        # TODO: should the following live in a class factory method?
        # if resume training and checkpoint exists, load it
        # get latest checkpoint
        checkpoint = None
        if args.checkpoint is not None:
            # if args.checkpoint is a directory that exists
            if os.path.isdir(args.checkpoint):
                checkpoint = args.checkpoint
            # otherwise, use the resume strategy to get the checkpoint
            else:                
                checkpoint = get_checkpoint(model_dir=args.model_dir, strategy=args.checkpoint)
            if checkpoint is not None:
                log(f'Resuming from checkpoint: {checkpoint}')
                self.load(checkpoint)
        
        if checkpoint is None:
            if args.checkpoint is not None:
                logging.warning(f'--resume value of {args.checkpoint} specified, but no checkpoint found. Starting training from scratch.')
            args.checkpoint = None
        
            if args.distributed:
                self.model.to(self.device)
                self.model = DDP(self.model,
                                 device_ids=[args.local_rank],
                                 output_device=args.local_rank)
            else:
                self.model.to(self.device)

        # distributed training attributes
        self.rank_0 = True
        if self.args.distributed:
            self.rank_0 = self.args.local_rank == 0
        # variable to determine if we're rank_0 in distributed mode or not in distributed mode at all
        self.main_process = self.rank_0 or not self.args.distributed

        # wandb logging (will resume from checkpoint if exists)
        self.args.wandb = False
        if 'wandb_project' in args and args.wandb_project is not None:
            if self.main_process:
                self._init_wandb(checkpoint)

    def _init_wandb(self, checkpoint=None):
        """Initializes wandb logging."""
        # create a run name based on the folder name of the model directory
        self.wandb_run_name = None
        if 'model_dir' in self.args and self.args.model_dir is not None:
            self.wandb_run_name = os.path.basename(
                os.path.normpath(self.args.model_dir))
        self.args.wandb = True
        # if we're resuming a run, specify the run_id loaded from the checkpoint
        if checkpoint is not None:
            self.wandb_run = wandb.init(
                project=self.args.wandb_project,
                config=self.args,
                id=self.wandb_run_id,  # loaded from checkpoint
                resume='must',
                name=self.wandb_run_name)
        else:
            self.wandb_run = wandb.init(project=self.args.wandb_project,
                                        config=self.args,
                                        name=self.wandb_run_name)
            self.wandb_run_id = self.wandb_run.id

    def run_callbacks(self, method_name):
        for cb in self.callbacks:
            getattr(cb, method_name)(self)

    def configure_optimizer(self):
        """Creates an optimizer (and potentially a learning rate scheduler) for the model."""
        self.optimizer, self.lr_scheduler = get_optimizer(
            self.args, self.model, **self.args.optimizer_kwargs)

    def log(self, key, value):
        """Logs a message to internal key-value store, including step information."""
        if key not in self.metrics:
            self.metrics[key] = {'steps': [], 'values': []}
        self.metrics[key]['steps'].append(self.global_step)
        self.metrics[key]['values'].append(value)
        # TODO: log to wandb and/or tensorboard

    def _report(self, prefix='train'):
        """Reports all keys starting with the specified prefix."""
        ignore_keys = [
            'train_loss_running', 'val_loss_running', 'train_duration_running',
            'val_duration_running', 'test_loss_running', 'test_duration_running',
            'test_loss', 'test_accuracy'
        ]
        # report metrics
        for key in self.metrics:
            # don't report certain keys
            if key in ignore_keys:
                continue
            if key.startswith(prefix):
                # get last value for metric and its corresponding step
                value = self.metrics[key]['values'][-1]
                step = self.metrics[key]['steps'][-1]
                # report to stdout
                log(f'{key} at step {step:,} - {value:.4f}')
        
        # report learning rate
        if prefix == 'train':
            log(f'learning rate at step {self.global_step:,} - {self.lr_scheduler.get_last_lr()[0]}')


    def _average_metric(self,
                        key='train_loss_running',
                        summary_key='train_loss'):
        """Computes the average over batch-level statistics for the specified
        key and save the result to summary_key. NOTE: this will pop the running
        values from the list of metrics, which will reset the running statistics."""
        if key not in self.metrics:
            return None

        if len(self.metrics[key]['values']) == 0:
            avg = None
        else:
            avg = sum(self.metrics[key]['values']) / len(
                self.metrics[key]['values'])

        # record this average in the list of summary values
        self.log(summary_key, avg)

        # pop the running values, which will reset the metric
        self.metrics.pop(key)

    def load(self, checkpoint=None, update_metrics=True):
        """Loads the model from disk."""
        # load trainer state
        # TODO: determine if this is what we want to do wrt the global step and running metrics
        with open(os.path.join(checkpoint, 'trainer_state.json'), 'r') as f:
            trainer_state = json.load(f)
            # if we're reloading a model, we may not want to clobber the metrics of the current model
            if update_metrics:
                self.training_step = trainer_state['training_step']
                self.global_step = trainer_state['global_step']
                self.epoch = trainer_state['epoch']
                self.metrics = trainer_state['metrics']
                # self.args = argparse.Namespace(**trainer_state['args'])
                self.wandb_run_id = trainer_state['wandb_run_id']

        # define device map so we load on rank 0 and broadcast to other ranks
        # https://discuss.pytorch.org/t/checkpoint-in-multi-gpu/97852/11
        map_location = None
        if self.args.distributed:
            map_location = f'cuda:{self.args.local_rank}'
            self.model.load_state_dict(
                torch.load(os.path.join(checkpoint, 'model.pt'),
                           map_location=map_location))
            self.model.to(self.args.device)
            logging.info(
                f'Model device {self.model.device} on rank {self.args.local_rank}'
            )
            # TODO: this is redundant with __init__, remove this
            self.model = DDP(
                self.model,
                device_ids=[self.args.device],
                output_device=self.args.device,
            )
        else:
            self.model.load_state_dict(
                torch.load(os.path.join(checkpoint, 'model.pt'),
                           map_location=map_location))
            self.model.to(self.args.device)

        logging.info(f'Loaded model on {self.args.device}...')
        self.optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint, 'optimizer.pt'),
                       map_location=map_location))
        # make sure to zero out gradients, in case any were accumulated
        self.optimizer.zero_grad()
        logging.info(f'Loaded optimizer on {self.args.device}...')
        self.lr_scheduler.load_state_dict(
            torch.load(os.path.join(checkpoint, 'lr_scheduler.pt'),
                       map_location=map_location))
        # TODO: load mixed precision scaler state

    def train_step(self, batch):
        """Runs a single training step."""
        start = time.perf_counter()
        self.model.train()            

        # forward pass
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # mixed precision training
        device_type = 'cuda' if 'cuda' in str(self.args.device) else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=self.args.fp16):
            outputs = self.model(**batch)
            loss = outputs.loss
            if self.custom_loss_fn is not None:
                loss = self.custom_loss_fn(outputs.logits, batch['labels'])

        # log training step loss
        self.log(
            'train_loss_running',
            loss.detach().item(),
        )

        self.run_callbacks('on_backward_start')
        # ensure we're scaling the loss appropriately for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        self.scaler.scale(loss).backward() # mixed precision training
        self.run_callbacks('on_backward_end')

        # optimization step
        if self.training_step % self.gradient_accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.lr_scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()
            # TODO: we're off by one again, probably can global_step at 0 to fix
            self.global_step += 1

        # only log learning rate every reporting step to avoid spamming
        if self._should_report():
            self.log('learning_rate', self.lr_scheduler.get_last_lr()[0])

        logits = outputs.logits
        labels = batch['labels']
        # update metrics
        for metric in self.metric_objects:
            metric.update(logits, labels)

        end = time.perf_counter()
        duration = end - start
        self.log('train_duration_running', duration)

    def eval_step(self, batch, subset='val'):
        """Runs a single evaluation step."""
        self.model.eval()

        # forward pass
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            if self.args.distributed:
                # need to call module to prevent hanging
                # https://discuss.pytorch.org/t/distributeddataparallel-barrier-doesnt-work-as-expected-during-evaluation/99867/7
                outputs = self.model.module(**batch)
            else:
                outputs = self.model(**batch)
            loss = outputs[0]
            if self.custom_loss_fn is not None:
                loss = self.custom_loss_fn(outputs.logits, batch['labels'])

            # log step loss
            self.log(
                f'{subset}_loss_running',
                loss.detach().item(),
            )

            preds = outputs[1]
            labels = batch['labels']
            # update metrics
            for metric in self.metric_objects:
                metric.update(preds, labels)

    def run_eval(self, subset='val'):
        loader = self.val_loader if subset == 'val' else self.test_loader
        # manually control progress bar to avoid interleaved print statements
        pbar = tqdm(total=len(loader), disable=not self.main_process, desc=f'{subset} epoch {self.epoch}')
        for idx, batch in enumerate(loader):
            self.eval_step(batch, subset=subset)
            pbar.update()
        pbar.close()

        # report metrics and reset running metrics
        self._average_metric(key=f'{subset}_loss_running', summary_key=f'{subset}_loss')
        
        # calculate all metrics
        for metric in self.metric_objects:
            metric_score = metric.compute()
            metric_name = f'{subset}_{metric.name}'
            self.log(metric_name, metric_score)
            metric.reset()

        # report validation metrics to stdout
        self._report(prefix=subset)

    def predict(self, loader):
        """Runs a prediction loop on a given loader."""
        self.model.eval()
        predictions = []
        for batch in tqdm(loader, desc='Predicting'):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits
                predictions.append(logits.detach().cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        return predictions
    
    def _should_evaluate(self):
        """Determines if we should evaluate the model. If val_steps == 0, we
        default to evaluating at the end of each epoch. Otherwise, we evaluate
        every val_steps steps."""
        # only evaluate on rank 0
        if not self.main_process:
            return False
        
        # if validation loader is None, don't evaluate
        if self.val_loader is None:
            return False

        # if we've reached the end of the epoch and val_steps == 0, evaluate
        end_of_epoch = self.idx == len(self.train_loader) - 1
        if (self.args.val_steps == 0) and end_of_epoch:
            return True

        # val_steps > 0 and the current step is a multiple of val_steps, evaluate
        # val_steps is based on the number of optimization steps, so we need to multiply
        # by gradient_accumulation_steps
        val_steps = self.args.val_steps * self.gradient_accumulation_steps
        if (self.args.val_steps > 0) and (self.training_step % val_steps == 0):
            return True

        # otherwise, don't evaluate
        return False
    
    def _should_report(self):
        """Determines if we should report to stdout. If reporting_steps == 0, we
        default to reporting at the end of each epoch. Otherwise, we report
        every reporting_steps steps."""
        # only report on rank 0
        if not self.main_process:
            return False
        
        # if we've reached the end of the epoch and reporting_steps == 0, report
        end_of_epoch = self.idx == len(self.train_loader) - 1
        if (self.args.reporting_steps == 0) and end_of_epoch:
            return True

        # reporting_steps > 0 and the current step is a multiple of reporting_steps, report
        # reporting_steps is based on the number of optimization steps, so we need to multiply
        # by gradient_accumulation_steps
        reporting_steps = self.args.reporting_steps * self.gradient_accumulation_steps
        if (self.args.reporting_steps > 0) and (self.training_step % reporting_steps == 0):
            return True

        # otherwise, don't report
        return False

    def _train(self):
        """Runs training and evaluation for the specified number of epochs."""
        self.run_callbacks('on_training_start')

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch

            self.run_callbacks('on_epoch_start')

            # set the epoch for the sampler so it shuffles appropriately
            # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
                logging.debug(
                    f'Setting epoch to {str(self.train_loader.sampler.epoch)} on rank {str(self.args.local_rank)}'
                )

            pbar = tqdm(initial=self.training_step, total=len(self.train_loader), disable=not self.main_process, desc=f'Train Epoch {epoch}')
            pbar_is_closed = False
            for idx, batch in enumerate(self.train_loader, start=1):
                self.idx = idx # used to determine if we should evaluate or report
                
                self.run_callbacks('on_batch_start')
                self.train_step(batch)

                # potentially report metrics and reset running training metrics
                if self._should_report():
                    if (self.idx == (len(self.train_loader) - 1)) and (not pbar_is_closed):
                        # manually finish the progress bar
                        pbar.update()
                        pbar.close()
                        pbar_is_closed = True
                    self._average_metric(key='train_loss_running',
                                        summary_key='train_loss')
                    self._average_metric(key='train_duration_running',
                                        summary_key='train_duration')
                    # calculate all training metrics
                    for metric in self.metric_objects:
                        metric_score = metric.compute()
                        metric_name = f'train_{metric.name}'
                        self.log(metric_name, metric_score)
                        metric.reset()

                    # report training metrics to stdout
                    self._report(prefix='train')

                # potentially evaluate on validation set
                # NB: we don't evaluate on the last batch of the epoch because
                # we want to avoid breaking the tqdm progress bar
                if self._should_evaluate():
                    if (self.idx == (len(self.train_loader) - 1)) and (not pbar_is_closed):
                        # manually finish the progress bar
                        pbar.update()
                        pbar.close()
                        pbar_is_closed = True
                    self.run_eval(subset='val')
                    # optionally evaluate on test set
                    if self.test_loader is not None:
                        self.run_eval(subset='test')


                # TODO: make a distinction between train and val batch/epoch_end
                # early stopping, model checkpointing, etc. are really things
                # that happen at the end of a val_epoch, not at the end of a
                # train_batch
                self.run_callbacks('on_batch_end')

                if not pbar_is_closed:
                    pbar.update()
                
                if self.should_stop:
                    pbar.close()
                    self.training_step += 1
                    break

                self.training_step += 1
            
            # if pbar is still open, close it
            if not pbar_is_closed:
                pbar.close()

            self.run_callbacks('on_epoch_end')

            if self.should_stop:
                logging.info('Early stopping...')
                break

        self.run_callbacks('on_training_end')
        return None

    def train(self):
        # allow for variable number of batches per epoch
        if self.args.distributed:
            # https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=join#torch.nn.parallel.DistributedDataParallel.join
            # throws an exception if any process terminates early, which is useful for maintaining consistent state
            # and early stopping
            try:
                with self.model.join(throw_on_early_termination=True):
                    self._train()
            except RuntimeError as e:
                # if uneven number of batches or early stopping, terminate training
                logging.debug(
                    f'Exception thrown on rank {self.args.local_rank}')
                logging.debug(e)
                logging.debug(
                    f'Terminating training on {self.args.local_rank}')
                return None
        else:
            self._train()
        
        # load best model, if load_best_model_at_end is True
        if self.args.load_best_model_at_end:
            checkpoint = get_checkpoint(model_dir=self.args.model_dir, strategy=self.args.checkpoint)
            if checkpoint is not None:
                # logging.warning('--load-best-model-at-end is specified, but no checkpoint found. Consider setting --save-steps')
                logging.warning(f'Loading best model from {checkpoint}')
                # NB: update_metrics=False so we don't clobber the metrics of the current model
                self.load(checkpoint, update_metrics=False)


class CustomHFTrainer(HFTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        if self.args.weighted_loss:
            # compute custom loss
            self.args.class_weights = self.args.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=self.args.class_weights)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        else:
            # compute default loss
            loss = super().compute_loss(model, inputs, return_outputs=False)
        return (loss, outputs) if return_outputs else loss

if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)