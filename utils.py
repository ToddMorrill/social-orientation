import logging
import os
import sys
import random

import numpy as np
import pandas as pd
import tiktoken
import torch
import torch.distributed as dist
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

def load_prompt(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def create_line(row):
    """Creates conversation row in markdown format."""
    content = f"| {row['utterance_id']} | {row['speaker']} | {row['text'].strip()} |"
    return content


def log(message):
    # determine if we are running in distributed mode
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    # determine if we are the master process
    is_master = not is_distributed or int(os.environ['RANK']) == 0
    if is_master:
        logging.info(message)

def set_random_seed(seed):
    """Sets the random seed for torch, numpy, and python."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def dist_setup(args):
    # if RANK and WORLD_SIZE are set, then we are running in distributed mode
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_distributed:
        args.distributed = True
        # announce that rank k is starting
        rank = int(os.environ['RANK'])
        logging.debug(f'Rank {rank} is starting.')
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        args.local_rank = rank
        args.main_process = rank == 0
        world_size = dist.get_world_size()
        args.world_size = world_size
        device_id = rank % torch.cuda.device_count()
        args.device = torch.device(f'cuda:{device_id}')
        log('Distributed training enabled.')
        logging.info(f'Rank: {rank}, world size: {world_size}')
        return args
    else:
        args.distributed = False
        args.local_rank = None
        args.main_process = True
        args.world_size = None
        log('Distributed training disabled.')
        return args

def dist_cleanup(args):
    if dist.is_initialized():
        logging.debug(
            f'Shutting down distributed training on rank: {args.local_rank}.')
        dist.barrier()
        logging.debug(
            f'Shutting down distributed training on rank: {args.device}.')
        dist.destroy_process_group()

# def shutdown_handler(signum, frame):
#     """Shuts down the distributed training when ctrl+c signal is detected."""
#     logging.info('Ctrl+C detected, Shutting down distributed training.')
#     dist.destroy_process_group()
#     sys.exit(0)

def get_optimizer(args, model, **kwargs):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            **kwargs,
        )
    elif args.optimizer == 'SGD-Nesterov':
        optimizer = optim.SGD(
            model.parameters(),
            **kwargs,
        )
    elif args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(
            model.parameters(),
            **kwargs,
        )
    elif args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(
            model.parameters(),
            **kwargs,
        )
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            **kwargs,
        )
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            **kwargs,
        )
    # get learning rate scheduler
    scheduler = None
    if args.lr_scheduler == 'linear-with-warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.num_train_steps,
        )
    return optimizer, scheduler

# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def add_utterance_id(group_df):
    """Add a unique identifier for each utterance."""
    group_df['utterance_id'] = list(range(len(group_df)))
    # add 1 to the utterance id so that it starts at 1
    group_df['utterance_id'] = group_df['utterance_id'] + 1
    return group_df

def merge_social_orientation_labels(df, social_orientation_df, drop_missing=False):
    # create utterance ids per file_id
    if 'utterance_id' not in df.columns:
        df = df.groupby('conversation_id', group_keys=False).apply(add_utterance_id)
    
    # merge
    merged_df = pd.merge(df, social_orientation_df, on=['conversation_id', 'utterance_id', 'speaker'], how='left')
    
    # drop conversations that are entirely missing labels
    if drop_missing:
        # identify conversations where all utterances are missing labels
        missing_conversations = merged_df.groupby('conversation_id')['social_orientation'].apply(lambda x: x.isna().all())
        missing_conversations = missing_conversations[missing_conversations].index
        # remove these conversations
        merged_df = merged_df[~merged_df['conversation_id'].isin(set(missing_conversations))]
        # merged_df = merged_df[merged_df['social_orientation'].notna()]

    return merged_df

def get_checkpoint(model_dir, strategy='best'):
    """Returns the checkpoint directory for the specified strategy, which can be
    'best' or 'last'."""
    if strategy is None:
        # if no strategy is specified, default to best
        strategy = 'best'
    # get the best checkpoint per best_checkpoint.txt
    if strategy == 'best':
        if not os.path.exists(os.path.join(model_dir, 'best_checkpoint.txt')):
            logging.warning(f'No best checkpoint found in {model_dir}.')
            return None
        with open(os.path.join(model_dir, 'best_checkpoint.txt'), 'r') as f:
            checkpoint = f.read()
        return checkpoint
    elif strategy == 'last':
        # get the last checkpoint
        checkpoints = [
            os.path.join(model_dir, f) for f in os.listdir(model_dir) if 'checkpoint' in f
            and os.path.isdir(os.path.join(model_dir, f))
        ]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
        checkpoint = None
        if len(checkpoints) > 0:
            checkpoint = checkpoints[-1]
        return checkpoint
    else:
        raise ValueError(f'Invalid checkpoint strategy {strategy}. Must be "best" or "last".')