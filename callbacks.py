"""This module implements commonly used callbacks. I should really be using
PyTorch lightning..."""
import copy
import json
import os
import logconfig
import logging

import torch
import torch.distributed as dist

from utils import log


class Callback(object):

    def on_training_start(self, trainer):
        pass

    def on_epoch_start(self, trainer):
        pass

    def on_batch_start(self, trainer):
        pass

    def on_backward_start(self, trainer):
        pass

    def on_backward_end(self, trainer):
        pass

    def on_batch_end(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_training_end(self, trainer):
        pass

    def on_validation_start(self, trainer):
        pass

    def on_validation_end(self, trainer):
        pass


class ModelSaver(Callback):

    def __init__(self,
                 save_dir,
                 save_every_n_steps=None,
                 num_checkpoints=2,
                 monitor='val_loss',
                 minimize=True):
        """Defaults to saving at the end of the epoch if no step interval is
        provided."""
        self.name = 'ModelSaver'
        self.save_dir = save_dir
        self.save_every_n_steps = save_every_n_steps
        if self.save_every_n_steps == 0:
            self.save_every_n_steps = None
        self.num_checkpoints = num_checkpoints
        self.monitor = monitor
        self.minimize = minimize
        self.current_step = 0
        self.current_train_step = 0
        self.best_checkpoint = None

    def should_save(self, trainer):
        """Called to determine if we're on a save step."""
        # only save if num_checkpoints > 0
        save_checkpoints = self.num_checkpoints > 0
        # if we're saving every n steps, check if we're on a save step
        if self.save_every_n_steps is not None:
            # only save every n steps
            # if we're using gradient accumulation, we need to multiply the step interval by the number of gradient accumulation steps
            save_steps = self.save_every_n_steps * trainer.gradient_accumulation_steps
            save_step = self.current_train_step % save_steps == 0
            if save_step and save_checkpoints:
                return True
        else:
            # otherwise, we're saving at the end of the epoch
            if save_checkpoints:
                return True

        return False

    def _remove_old_checkpoints(self):
        """Removes old checkpoints."""
        # get all checkpoint directories
        checkpoints = [
            f for f in os.listdir(self.save_dir) if 'checkpoint' in f
            and os.path.isdir(os.path.join(self.save_dir, f))
        ]
        # sort checkpoints by step number (descending)
        checkpoints = sorted(checkpoints,
                             key=lambda x: int(x.split('-')[1]),
                             reverse=True)
        checkpoints = [
            os.path.join(self.save_dir, checkpoint)
            for checkpoint in checkpoints
        ]

        # remove the best checkpoint from the list (so it doesn't get deleted)
        checkpoints.remove(self.best_checkpoint)

        # remove all checkpoints from old runs (i.e. greater than the current global step)
        # TODO: this is an opinionated move, maybe expose a parameter to control this?
        final_checkpoints = []
        for checkpoint in checkpoints:
            checkpoint_number = int(os.path.basename(checkpoint).split('-')[1])
            if checkpoint_number > self.current_step:
                log(f'Removing old checkpoint: {checkpoint}')
                os.system(f'rm -rf {checkpoint}')
            else:
                final_checkpoints.append(checkpoint)

        # remove all extra checkpoints (those that are less than the current global step)
        # two cases to consider:
        # 1. current_step is the best checkpoint, in which case we subtract 1 from the number of checkpoints to keep
        # 2. current_step is not the best checkpoint, in which case another checkpoint is using one of the slots, so again we subtract 1
        keep_idx = self.num_checkpoints - 1
        if keep_idx >= 0 and keep_idx < len(final_checkpoints):
            # NB: final_checkpoints is sorted in descending order by step number
            for checkpoint in final_checkpoints[keep_idx:]:
                checkpoint_number = int(
                    os.path.basename(checkpoint).split('-')[1])
                # the following should always be true, but just in case...
                if checkpoint_number < self.current_step:
                    log(f'Removing old checkpoint: {checkpoint}')
                    os.system(f'rm -rf {checkpoint}')

    def save(self, trainer):
        """Saves the model to disk."""
        # in distributed mode, only save from rank 0
        if not trainer.main_process:
            return

        save_dir = os.path.join(self.save_dir,
                                f'checkpoint-{trainer.global_step}')
        # if the directory exists, first remove it to avoid cross-contaminating with old runs
        if os.path.exists(save_dir):
            os.system(f'rm -rf {save_dir}')
        os.makedirs(save_dir)

        logging.info(f'Saving model to {save_dir}...')
        if trainer.args.distributed:
            model_obj = trainer.model.module.state_dict()
        else:
            model_obj = trainer.model.state_dict()
        torch.save(model_obj, os.path.join(save_dir, 'model.pt'))
        # save optimizer and lr scheduler state
        torch.save(trainer.optimizer.state_dict(),
                   os.path.join(save_dir, 'optimizer.pt'))
        torch.save(trainer.lr_scheduler.state_dict(),
                   os.path.join(save_dir, 'lr_scheduler.pt'))

        # save args
        args = copy.deepcopy(trainer.args)

        # remove attributes that can't be serialized
        # TODO: generalize this to any attribute that can't be serialized
        del args.device

        # if logging with wandb
        wandb_run_id = None
        if 'wandb_project' in args and args.wandb_project is not None:
            wandb_run_id = trainer.wandb_run_id

        # record the trainer state
        trainer_state = {
            'training_step': trainer.training_step,
            'global_step': trainer.global_step,
            'epoch': trainer.epoch,
            'metrics': trainer.metrics,
            'args': vars(args),
            'wandb_run_id': wandb_run_id
        }
        with open(os.path.join(save_dir, 'trainer_state.json'), 'w') as f:
            json.dump(trainer_state, f)

        # save the tokenizer
        if trainer.tokenizer is not None:
            trainer.tokenizer.save_pretrained(save_dir)

        # if best == True, write a text file to args.model_dir to denote which checkpoint has the best metric
        # get best checkpoint
        if self.minimize:
            best_idx = min(
                range(len(trainer.metrics[self.monitor]['values'])),
                key=lambda idx: trainer.metrics[self.monitor]['values'][idx])
        else:
            best_idx = max(
                range(len(trainer.metrics[self.monitor]['values'])),
                key=lambda idx: trainer.metrics[self.monitor]['values'][idx])
        best_checkpoint_step = trainer.metrics[self.monitor]['steps'][best_idx]
        best = False
        if best_checkpoint_step == self.current_step:
            best = True

        if best:
            with open(
                    os.path.join(self.save_dir,
                                 'best_checkpoint.txt'), 'w') as f:
                f.write(save_dir)
            # update best checkpoint
            self.best_checkpoint = save_dir
        
        # remove old checkpoints
        self._remove_old_checkpoints()

    def on_epoch_end(self, trainer):
        """If no step interval is provided, save at the end of the epoch."""
        self.current_step = trainer.global_step
        self.current_train_step = trainer.training_step
        if self.should_save(trainer):
            self.save(trainer)

    def on_batch_end(self, trainer):
        """If a step interval is provided, save at the end of the step."""
        # use trainer's global step, which is useful when resuming training
        self.current_step = trainer.global_step
        self.current_train_step = trainer.training_step
        if self.should_save(trainer):
            self.save(trainer)


class EarlyStopping(Callback):

    def __init__(self,
                 patience,
                 monitor='val_loss',
                 min_delta=0,
                 eval_every_n_steps=None,
                 minimize=True):
        """Defaults to evaluating at the end of the epoch if no step interval is
        provided."""
        self.name = 'EarlyStopping'
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.eval_every_n_steps = eval_every_n_steps
        if self.eval_every_n_steps == 0:
            self.eval_every_n_steps = None
        self.current_train_step = 0
        self.best_score = None
        self.wait = 0  # the current number of epochs/steps without improvement
        self.minimize = minimize

    def should_stop(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if ((score < (self.best_score - self.min_delta)) and self.minimize) or \
            ((score > (self.best_score + self.min_delta)) and not self.minimize):
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            logging.info(
                f'No improvement in {self.monitor} for {self.wait} evaluation steps.')

        if self.wait >= self.patience:
            return True

        return False

    def on_epoch_end(self, trainer):
        # validation is only performed on the main process
        if not trainer.main_process:
            return

        if self.eval_every_n_steps is None:
            score = trainer.metrics[self.monitor]['values'][-1]
            if self.should_stop(score):
                trainer.should_stop = True

    def on_batch_end(self, trainer):
        # validation is only performed on the main process
        if not trainer.main_process:
            return

        if self.eval_every_n_steps is None:
            return

        self.current_train_step = trainer.training_step
        # if we're using gradient accumulation, we need to multiply the step interval by the number of gradient accumulation steps
        val_steps = self.eval_every_n_steps * trainer.gradient_accumulation_steps
        if self.current_train_step % val_steps == 0:
            score = trainer.metrics[self.monitor]['values'][-1]
            if self.should_stop(score):
                trainer.should_stop = True


class Accuracy(object):

    def __init__(self, tokenizer=None, label2id=None):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.name = 'accuracy'
        self.num_correct = 0
        self.num_total = 0

    def update(self, y_pred, y_true):
        with torch.no_grad():
            # if using a tokenizer, convert token ids to labels
            if self.tokenizer is not None and self.label2id is not None:
                y_pred = self.tokenizer.batch_decode(y_pred.data.max(-1)[1], skip_special_tokens=True)
                y_true = self.tokenizer.batch_decode(y_true, skip_special_tokens=True)
                y_pred = [self.label2id[label] if label in self.label2id else -1 for label in y_pred]
                y_true = [self.label2id[label] if label in self.label2id else -1 for label in y_true]
                y_pred = torch.tensor(y_pred, dtype=torch.long)
                y_true = torch.tensor(y_true, dtype=torch.long)
            else:
                _, y_pred = torch.max(y_pred.data, 1)
            
            self.num_total += y_true.size(0)
            self.num_correct += (y_pred == y_true).sum().item()

    def compute(self):
        return self.num_correct / self.num_total if self.num_total != 0 else 0

    def reset(self):
        self.num_correct = 0
        self.num_total = 0
