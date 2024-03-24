import argparse
import logging
import tempfile
import logconfig  # import default logger setup
import os
import random
import sys

import numpy as np
import torch

from utils import dist_setup, log, set_random_seed

def parse_window_size(values):
    """Parse window size argument(s)."""
    if values is None:
        return None
    original_values_is_list = isinstance(values, list)
    if not isinstance(values, list):
        values = [values]
    final_values = []
    for value in values:
        if value == 'all':
            final_values.append(None)
        else:
            final_values.append(int(value))
    if original_values_is_list:
        return final_values
    else:
        return final_values[0]


def parse_args(args=None):
    """Parse command line arguments."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()

    # train.py related arguments
    parser.add_argument('--experiment',
                        default=None,
                        type=str,
                        help='Which experiment to run (e.g. subset, etc.).')
    parser.add_argument(
        '--window-sizes',
        nargs='+',
        type=parse_window_size,
        default=None,
        help=
        'The window sizes to use for the window-size experiment. If None, then default to the window sizes specified in train.py.'
    )
    parser.add_argument(
        '--dont-compare-social-orientation',
        action='store_true',
        default=False,
        help=
        'If passed, the code will not compare social orientations and instead will default to the --include-social-orientation argument. This is useful for the window-size experiment, where we want to compare the effect of window size while holding social orientation constant (e.g. when evaluating the effectiveness of GPT annotations).'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=1,
        help=
        'The number of runs to perform. If > 1, the experiment will be repeated with random seeds --seed through --seed + --num-runs - 1.'
    )
    parser.add_argument(
        '--use-multiprocessing',
        action='store_true',
        default=False,
        help=
        'If passed, the code will use multiprocessing to parallelize experiment runs.'
    )
    parser.add_argument(
        '--use-cache',
        action='store_true',
        default=False,
        help=
        'If passed, the code will use cached experimental results, where possible.'
    )
    parser.add_argument('--exp-dir',
                        type=str,
                        default='./logs/experiments',
                        help='The directory to save experimental results to.')
    parser.add_argument('--model-dir', type=str, default=None)
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help=
        'Resume training from a checkpoint. Optionally specify, "best", "last", or a path to a checkpoint.'
    )
    parser.add_argument('--hf-cache-dir', type=str, default=None)
    parser.add_argument(
        '--fast-dev-run',
        action='store_true',
        default=False,
        help='Run 1 batch of train, val, test for debugging purposes.')
    parser.add_argument('--model-name-or-path',
                        type=str,
                        default='distilbert-base-uncased',
                        help='The name or path of the model to use.')
    parser.add_argument(
        '--disable-cuda',
        action='store_true',
        help='If passed, the code will use the CPU instead of the GPU.')
    parser.add_argument(
        '--num-dataloader-workers',
        default=0,
        type=int,
        help='The number of workers for the train and test dataloaders.')
    parser.add_argument(
        '--optimizer',
        default='AdamW',
        choices=[
            'SGD', 'SGD-Nesterov', 'Adagrad', 'Adadelta', 'Adam', 'AdamW'
        ],
        help=
        'The optimizer used to train the network. Must be one of {SGD, SGD-Nesterov, Adagrad, Adadelta, Adam, AdamW}.'
    )
    parser.add_argument(
        '--lr-scheduler',
        default='linear-with-warmup',
        choices=['linear-with-warmup'],
        help=
        'The learning rate scheduler to employ. Must be one of {linear-with-warmup}.'
    )
    parser.add_argument('--num-warmup-steps',
                        type=float,
                        default=100,
                        help='The number of training steps to warmup for.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,  # trust early stopping will take over
        help='The number of training epochs.')
    parser.add_argument('--wandb-project',
                        type=str,
                        help='The name of the wandb project to log to.')
    parser.add_argument('--lr',
                        type=float,
                        default=5e-5,
                        help='The learning rate.')
    parser.add_argument(
        '--scale-lr',
        action='store_true',
        default=False,
        help=
        'If passed, the learning rate will be scaled by the number of GPUs.')
    parser.add_argument(
        '--reporting-steps',
        type=int,
        default=100,
        help='The number of steps between each reporting step.')
    parser.add_argument('--val-steps',
                        type=int,
                        default=100,
                        help='The number of steps between evaluation.')
    parser.add_argument('--save-steps',
                        type=int,
                        default=0,
                        help='The number of steps between saving the model.')
    parser.add_argument(
        '--num-checkpoints',
        type=int,
        default=0,
        help=
        'The number of checkpoints to keep. This will always attempt to keep the best checkpoint first.'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help=
        'The logging level to use. Must be one of {DEBUG, INFO, WARNING, ERROR, CRITICAL}.'
    )
    parser.add_argument(
        '--monitored-metric',
        type=str,
        default='val_loss',
        help=
        'The criterion to use for early stopping and deleting old checkpoints.'
    )
    parser.add_argument(
        '--maximize-metric',
        action='store_true',
        default=False,
        help=
        'If passed, the monitored metric will be maximized instead of minimized.'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=3,
        help=
        'The number of evaluations without improvement to wait before early stopping.'
    )
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='The random seed to use.')
    parser.add_argument('--triplet-loss',
                        action='store_true',
                        default=False,
                        help='If passed, the model will use a triplet loss.')
    parser.add_argument('--weight-decay',
                        type=float,
                        default=0.0,
                        help='The weight decay to use for the optimizer.')
    parser.add_argument(
        '--window-size',
        type=parse_window_size,
        default=2,
        help=
        'Determines the number of utterances considered when predicting a change point or social orientation tag. Must be an integer or "all".'
    )
    parser.add_argument(
        '--weighted-loss',
        action='store_true',
        default=False,
        help='Weights classes by their frequency and normalizes the weights.')
    parser.add_argument('--trainer',
                        type=str,
                        default='torch',
                        help='The trainer to use. Must be one of {torch, hf}.')
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help=
        'The number of gradient accumulation steps before applying gradients to update the weights.'
    )
    parser.add_argument(
        '--add-tokens',
        action='store_true',
        default=False,
        help=
        'If passed, a <sep> special token and class label tokens (e.g. Gregarious-Extraverted, Unassuming-Ingenuous, etc.) will be added to the tokenizer.'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=512,
        help=
        'The maximum sequence length to use for the model, particularly for the encoder.'
    )
    parser.add_argument('--fp16',
                        action='store_true',
                        default=False,
                        help='If passed, the model will mixed precision.')
    parser.add_argument(
        '--jobs-per-gpu',
        type=int,
        default=1,
        help=
        'The number of jobs to run per GPU. If you know the memory footprint of your model, you can stack multiple jobs per GPU.'
    )
    parser.add_argument(
        f'--eval-test',
        action='store_true',
        default=False,
        help=
        'If passed, the model will evaluate on the test set at each evaluation step.'
    )

    # data.py related arguments
    parser.add_argument(
        '--tokenizer-name-or-path',
        type=str,
        default=None,
        help=
        'The name or path of the tokenizer to use. If not passed, will default to --model-name-or-path.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='cga',
        help='Which dataset to use. Options: [cga, social-orientation]')
    parser.add_argument(
        '--data-dir',
        # type=os.path.expanduser,
        default='data/convokit/conversations-gone-awry-corpus')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='The batch size to use for training.')
    parser.add_argument(
        '--prepare-data-splits',
        action='store_true',
        default=False,
        help='If passed, train/val/test splits will be defined in data.py.')
    parser.add_argument('--social-orientation-filepaths',
                        type=os.path.expanduser,
                        nargs='+',
                        default=None,
                        help='The filepaths to the social orientation data.')
    parser.add_argument(
        '--predicted-social-orientation-filepaths',
        type=os.path.expanduser,
        nargs='+',
        default=None,
        help=
        'The filepaths to the predicted social orientation data. Note that this can be specified if --social-orientation-filepaths is already being used by say GPT-4 annotations and we want to run a window-size or subset experiment.'
    )
    parser.add_argument(
        '--include-speakers',
        action='store_true',
        default=False,
        help=
        'If passed, speaker information will be prepended to text utterances.')
    parser.add_argument(
        '--include-social-orientation',
        action='store_true',
        default=False,
        help=
        'If passed, the model will use utterance social orientation tags when training the model.'
    )
    parser.add_argument(
        '--drop-missing',
        action='store_true',
        default=False,
        help=
        'If passed, utterances with missing social orientation tags will be dropped.'
    )
    parser.add_argument(
        '--subset-pct',
        type=float,
        default=1.0,
        help=
        'Subsets the dataset by the specified percent (e.g. 0.1) for rapid development. Defaults to 1.0.'
    )
    parser.add_argument(
        '--subset-pcts',
        type=float,
        nargs='+',
        default=None,
        help=
        'Specifies a range of subsets to explore for a data ablation study. If passed, --subset-pct will be ignored. E.g. --subset-pcts 0.1 0.2 0.3 0.4 0.5 will train on 10%, 20%, 30%, 40%, and 50% of the data.'
    )
    parser.add_argument(
        '--disable-train-shuffle',
        action='store_true',
        default=False,
        help='If passed, the training data loader will not shuffle data.')
    parser.add_argument(
        '--casino-speaker',
        type=str,
        default='mturk_agent_1',
        help='Which speaker to model: {mturk_agent_1, mturk_agent_2, both}.')
    parser.add_argument(
        '--return-utterances',
        action='store_true',
        default=False,
        help=
        'If passed, the (CaSiNo) datasets will return utterances instead of entire conversations. This is useful when predicting social orientation tags for utterances in a conversation.'
    )
    parser.add_argument(
        '--load-best-model-at-end',
        action='store_true',
        default=False,
        help='If passed, the best model will be loaded at the end of training.'
    )
    parser.add_argument(
        '--make-predictions',
        action='store_true',
        default=False,
        help=
        'If passed, the model will make predictions on the train and val sets at the end of training.'
    )
    parser.add_argument(
        '--dont-return-labels',
        action='store_true',
        default=False,
        help='If passed, the data loader will not return labels.')
    parser.add_argument(
        '--disable-prepared-inputs',
        action='store_true',
        default=False,
        help=
        'If passed, we don\'t cache any of the prepared examples in the dataset class. This is useful when you want to modify the data on the fly (e.g., for the explainability/corruption experiment).'
    )

    # gpt_prompts.py related arguments
    parser.add_argument(
        '--gpt-data-dir',
        type=str,
        default='./data',
        help='Path to the output directory where JSONL files will be stored.')
    parser.add_argument('--prompt-filepath',
                        type=str,
                        default='./prompt.txt',
                        help='Path to the GPT instruction prompt file.')
    parser.add_argument('--gpt-model',
                        type=str,
                        default='gpt-3.5-turbo',
                        help='Which GPT model to use.')
    parser.add_argument(
        '--data-token-pct',
        type=float,
        default=0.8,
        help=
        'What percentage of the token limit to (after removing the prompt count) to use for data. The remainder will be available to the GPT model for generation. This may be need to set through a bit of trial and error by 0.5 is a good starting point.'
    )
    parser.add_argument(
        '--gpt-mode',
        default='short',
        choices=[
            'sample', 'short', 'long-remainder', 'all', 'val', 'train', 'test',
            'train-long', 'val-long', 'test-long'
        ],
        help=
        'sample corresponds to a sample of 10 conversations that can be used for rapid development. short corresponds to all conversations where [data_tokens <= (model_token_limit - prompt_tokens) * data_token_pct] (i.e. the conversations don\'t need to be split). long-remainder corresponds to all long conversations not included in short. all corresponds to all conversations, where conversations have been broken into chunks to respect the model\'s token limit.'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=1,
        help='Number of overlapping utterances to include in GPT calls.')
    parser.add_argument(
        '--min-utterances',
        type=int,
        default=2,
        help='Minimum number of utterances to include in GPT calls.')
    parser.add_argument(
        '--calculate-cost',
        action='store_true',
        default=False,
        help='If passed, the total cost GPT calls will be calculated.')

    # parse.py related arguments
    parser.add_argument('--gpt-outputs-filepaths',
                        type=os.path.expanduser,
                        nargs='+',
                        default=None,
                        help='Paths to the GPT output files to be parsed.')
    parser.add_argument(
        '--parse-output-dir',
        type=os.path.expanduser,
        default='data/gpt-4-cga-social-orientation-labels',
        help='Path to the output directory where parsed files will be stored.')

    # annotation.py related arguments
    parser.add_argument('--annotation-samples',
                        type=int,
                        default=30,
                        help='The number of samples to use for annotation.')

    # predict.py related arguments
    parser.add_argument('--prediction-dir',
                        type=str,
                        help='Directory where predictions will be saved.')
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=False,
        help=
        'If passed, predictions will be made in parallel on all available devices using torch\'s DataParallel.'
    )

    # analyze.py related arguments
    parser.add_argument(
        '--analysis-dir',
        type=str,
        default='logs/analysis',
        help='Directory where tables, figures, and data will be saved.')
    parser.add_argument(
        '--analysis-mode',
        type=str,
        default=None,
        help='Which analysis to perform. See analyze.py for options.')
    parser.add_argument('--experiment-filepath',
                        type=str,
                        nargs='+',
                        default=None,
                        help='Path to the experiment file to be analyzed.')
    parser.add_argument('--prediction-filepaths',
                        type=str,
                        nargs='+',
                        default=None,
                        help='Paths to the prediction files to be analyzed.')
    parser.add_argument(
        '--human-annotation-filepaths',
        type=str,
        nargs='+',
        default=None,
        help='Paths to the human annotation files to be analyzed.')

    # classic.py related arguments
    parser.add_argument('--features',
                        type=str,
                        nargs='+',
                        default=['social_counts', 'distilbert', 'tfidf'],
                        help='The feature sets to use for the classic model.'
                        )  # sentiment, VAD
    # parse arguments
    args = parser.parse_args(args)

    # print arguments
    command_line_args = []
    for arg in vars(args):
        command_line_args.append(f'--{arg}')
        command_line_args.append(str(getattr(args, arg)))
    # TODO: clean up formatting and handle case where sys.argv[0] might not be present
    # python_executable_name = os.path.basename(sys.executable)
    # print(python_executable_name)
    # print(sys.argv[0], end=' \\\n    ')
    # print(' \\\n    '.join(command_line_args))

    # if tokenizer-name-or-path is not passed, set it to the model-name-or-path
    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path

    # if val-steps and save-steps are both passed, make sure save-steps is a multiple of val-steps
    if args.val_steps != 0 and args.save_steps != 0:
        if args.save_steps % args.val_steps != 0:
            raise ValueError('--save-steps must be a multiple of --val-steps')

    # if args.model_dir is None, set it to a tmp directory
    if args.model_dir is None:
        args.model_dir = tempfile.mkdtemp()

    # TODO: do we want to adjust this by the effective batch size?
    args.optimizer_kwargs = {'lr': args.lr, 'weight_decay': args.weight_decay}
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    # set log level based on command line argument
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.loglevel}')
    logging.getLogger().setLevel(numeric_level)

    # set up distributed training if we've launched with torchrun
    args = dist_setup(args)

    log(f'Running with:\n{args}')

    # set random seed
    log(f'Setting random seed to {args.seed}')
    set_random_seed(args.seed)
    return args
