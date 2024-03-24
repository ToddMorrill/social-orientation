"""This module implements data loaders for the models.

Examples:    
    $ python -m data \
        --dataset social-orientation \
        --social-orientation-filepaths \
            data/gpt-4-cga-social-orientation-labels/train_results_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/val_results_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/train-long_results_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/val-long_results_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/test_results_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/test-long_results_parsed.csv \
        --include-speakers \
        --drop-missing \
        --log-level DEBUG \
        --window-size 4 \
        --subset-pct 0.1
"""
import logging
from args import parse_args
import random
import json
import os
from collections import Counter
from tqdm import tqdm

from convokit import Corpus, download
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from utils import merge_social_orientation_labels, log

CGA_LABEL2ID = {'Civil': 0, 'Uncivil': 1}
CGA_ID2LABEL = {v: k for k, v in CGA_LABEL2ID.items()}

SOCIAL_ORIENTATION_LABEL2ID = {
    'Cold': 0,
    'Arrogant-Calculating': 1,
    'Aloof-Introverted': 2,
    'Assured-Dominant': 3,
    'Unassuming-Ingenuous': 4,
    'Unassured-Submissive': 5,
    'Warm-Agreeable': 6,
    'Gregarious-Extraverted': 7,
    'Not Available': 8
}
SOCIAL_ORIENTATION_ID2LABEL = {
    v: k
    for k, v in SOCIAL_ORIENTATION_LABEL2ID.items()
}


def get_labels(args):
    if args.dataset == 'cga' or args.dataset == 'cga-cmv':
        label2id = CGA_LABEL2ID
        id2label = CGA_ID2LABEL
    elif args.dataset == 'social-orientation':
        label2id = SOCIAL_ORIENTATION_LABEL2ID
        id2label = SOCIAL_ORIENTATION_ID2LABEL
    else:
        logging.warning(
            f'Dataset {args.dataset} has no label mappings. This might either be because the dataset is not supported, or because the dataset has continuous labels like a dollar amount.'
        )
        return None, None
    return label2id, id2label


class DialogueDataset(Dataset):
    """Base class for all dialogue datasets, which implements the
    _combine_utterances method and has a prepared_inputs attribute (empty
    dict)."""

    def __init__(self,
                 df,
                 tokenizer,
                 window_size=2,
                 default_label=None,
                 max_len=None):
        self.df = df
        self.window_size = window_size
        self.default_label = default_label
        self.max_len = max_len
        self.tokenizer = tokenizer
        if max_len is None:
            self.max_len = tokenizer.model_max_length
        self.convo_ids = df['conversation_id'].unique().tolist()

        # pretokenize the text
        self.df.loc[:, 'input_ids'] = self.tokenizer(
            self.df['text'].values.tolist(),
            add_special_tokens=False,
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=False)['input_ids']

        # cache the prepared inputs
        self.prepared_inputs = {}

    def __len__(self):
        # length is likely the number of conversations in the dataset
        # but this can be easily overriden in the subclass
        return len(self.convo_ids)

    def _combine_utterances(self, input_ids_list):
        """Combine all utterances into a single input with <sep> tokens in
        between. If the sum of the lengths of the utterances is greater than
        max_len, then trim from the longest utterance until the total length
        is less than the max_len by a window_size amount, which will account
        for the CLS token (or EOS token) and the <sep> tokens. In particular,
        there are window_size - 1 <sep> tokens, and 1 CLS token (or EOS token).
        """
        # TODO: it might be nice to add some sort of ellipsis token to indicate
        # to the model that the input has been truncated
        total_len = sum([len(utterance) for utterance in input_ids_list])
        dialog_capacity = self.max_len - len(input_ids_list)
        while total_len > dialog_capacity:
            # find the longest utterance
            longest_utterance_idx = max(
                range(len(input_ids_list)),
                key=lambda idx: len(input_ids_list[idx]))
            # remove the last token from the longest utterance
            input_ids_list[longest_utterance_idx] = input_ids_list[
                longest_utterance_idx][:-1]
            # update the total length
            total_len -= 1

        # combine the utterances into a single input
        tokens = []
        tokens = [self.tokenizer.cls_token_id]
        for idx, utterance in enumerate(input_ids_list):
            # add a sep token between utterances
            tokens.extend(utterance)
            tokens.append(self.tokenizer.sep_token_id)
        # remove the last sep token
        tokens.pop()
        return tokens


class CGADataset(DialogueDataset):
    """Dataset for the Conversations Gone Awry corpus."""

    def __init__(
        self,
        df,
        tokenizer,
        window_size=2,
        max_len=None,
        disable_prepared_inputs=False,
        default_label=None,
    ):
        super().__init__(df, tokenizer, window_size, default_label, max_len,)
        self.target = 'meta.comment_has_personal_attack'
        self.disable_prepared_inputs = disable_prepared_inputs

    def __getitem__(self, idx):
        # if we've already prepared idx, retrieve from cache, and return it
        if idx in self.prepared_inputs and not self.disable_prepared_inputs:
            return self.prepared_inputs[idx]

        # otherwise prepare the input and add it to the cache
        # we only have one training example per conversation
        convo_id = self.convo_ids[idx]
        convo_df = self.df[self.df['conversation_id'] == convo_id]
        # combine the first window_size utterances into a single input
        # but ensure we don't include the last utterance (the awry utterance)
        # NB: assuming attack is always the last utterance
        utterances = convo_df['input_ids'].values.tolist()[:-1]
        # default to using all utterances, but if window_size is not None,
        # then only use the first window_size utterances
        if self.window_size is not None:
            utterances = utterances[:self.window_size]

        input_ids = self._combine_utterances(utterances)
        # if meta.has_personal_attack is True for any row, then the label is 1
        label = convo_df[self.target].any().astype(int)
        input_dict = {'input_ids': input_ids, 'label': label}
        if not self.disable_prepared_inputs:
            self.prepared_inputs[idx] = input_dict
        return input_dict


class SocialOrientationDataset(DialogueDataset):
    """Dataset for social orientation tagging."""

    def __init__(self,
                 df,
                 tokenizer,
                 window_size=2,
                 default_label='Not Available',
                 max_len=None,):
        super().__init__(df, tokenizer, window_size, default_label, max_len,)

    def __len__(self):
        # length is the number of utterances in the dataset
        return len(self.df)

    def __getitem__(self, idx):
        # if we've already prepared idx, retrieve from cache, and return it
        if idx in self.prepared_inputs:
            return self.prepared_inputs[idx]

        # otherwise prepare the input and add it to the cache
        utterance = self.df.iloc[idx]
        convo_id = utterance['conversation_id']
        convo_df = self.df[self.df['conversation_id'] == convo_id]
        # combine the first window_size utterances into a single input
        # where the last utterance is the one we're predicting social
        # orientation for
        first_loc = convo_df.index[0]
        end_loc = utterance.name
        if self.window_size is None:
            start_loc = first_loc
        else:
            start_loc = max(first_loc, end_loc - self.window_size + 1)
        utterances = convo_df.loc[start_loc:end_loc,
                                  'input_ids'].values.tolist()
        input_ids = self._combine_utterances(utterances)
        if 'social_orientation' in utterance:
            label = utterance['social_orientation']
            if pd.isna(label) or (label not in SOCIAL_ORIENTATION_LABEL2ID):
                # if label is nan, or not a valid label, then set it to default_label
                label = self.default_label
            label_id = SOCIAL_ORIENTATION_LABEL2ID[label]
            input_dict = {'input_ids': input_ids, 'label': label_id}
        else:
            # inference mode
            input_dict = {'input_ids': input_ids}
        self.prepared_inputs[idx] = input_dict
        return input_dict


class CGACMVDataset(DialogueDataset):
    """Dataset for the Conversations Gone Awry - Change My View corpus."""

    def __init__(self,
                 df,
                 tokenizer,
                 window_size=2,
                 default_label=None,
                 max_len=None,
                 return_utterances=False,
                 return_labels=True,
                 disable_prepared_inputs=False):
        super().__init__(df, tokenizer, window_size, default_label, max_len,)
        self.target = 'meta.has_removed_comment'
        # if true, will return a single data example for each utterance
        self.return_utterances = return_utterances
        self.return_labels = return_labels
        self.disable_prepared_inputs = disable_prepared_inputs
        self.convo_ids = df['conversation_id'].unique().tolist()

        self.tokenized_prompt = self.tokenizer(
            self.prompt, add_special_tokens=False)['input_ids']

    def __len__(self):
        if self.return_utterances:
            return len(self.df)
        else:
            return len(self.convo_ids)

    def __getitem__(self, idx):
        # if we've already prepared idx, retrieve from cache, and return it
        if idx in self.prepared_inputs and not self.disable_prepared_inputs:
            return self.prepared_inputs[idx]

        # otherwise prepare the input and add it to the cache
        if self.return_utterances:
            utterance = self.df.iloc[idx]
            convo_id = utterance['conversation_id']
            convo_df = self.df[self.df['conversation_id'] == convo_id]
            # combine previous window_size utterances into a single input
            # where the last utterance is the one we're predicting social
            # orientation for
            first_loc = convo_df.index[0]
            end_loc = utterance.name
            if self.window_size is None:
                start_loc = first_loc
            else:
                start_loc = max(first_loc, end_loc - self.window_size + 1)
            utterances = convo_df.loc[start_loc:end_loc,
                                      'input_ids'].values.tolist()
        else:
            # otherwise prepare the input and add it to the cache
            # we only have one training example per conversation
            convo_id = self.convo_ids[idx]
            convo_df = self.df[self.df['conversation_id'] == convo_id]

            # combine the first window_size utterances into a single input
            utterances = convo_df['input_ids'].values.tolist()
            # default to including all utterances if window_size is None
            if self.window_size is not None:
                utterances = utterances[:self.window_size]

        input_ids = self._combine_utterances(utterances)

        # replace dash with underscore in target
        # iloc[0] the right thing to do because meta.has_comment_removed is repeated in each row, so any row is valid
        label = convo_df.iloc[0][self.target]

        # TODO: consider how to predict both speakers' labels
        # maybe return a vector of values and treat it as a multilabel problem
        if self.return_labels:
            input_dict = {'input_ids': input_ids, 'label': int(label)}
        else:
            input_dict = {'input_ids': input_ids}
        if not self.disable_prepared_inputs:
            self.prepared_inputs[idx] = input_dict
        return input_dict


def prepare_example_dict(example_dict,
                         include_speakers=True,
                         include_social_orientation=False,):
    """Prepare an example dict for the model. If include_speakers is True, then
    prepend the speaker to the utterance."""
    # strip out extra whitespace
    example_dict['text'] = example_dict['text'].strip()
    example_dict['speaker'] = example_dict['speaker'].strip()
    if (not include_speakers) and (not include_social_orientation):
        return example_dict

    updated_string = ''

    if include_speakers:
        updated_string += example_dict['speaker'] + ': '

    if include_social_orientation:
        # add the social orientation label
        # if key not present, value is nan, or value is not a valid label, then
        # add Not Available
        if ('social_orientation' not in example_dict) or (pd.isna(
                example_dict['social_orientation'])) or (
                    example_dict['social_orientation']
                    not in SOCIAL_ORIENTATION_LABEL2ID):
            updated_string += '(Not Available) '
        else:
            updated_string += f"({example_dict['social_orientation']}) "

    # add the utterance
    updated_string += example_dict['text']

    example_dict['text'] = updated_string
    return example_dict


def load_social_orientation_data(filepaths):
    dfs = []
    for filepath in filepaths:
        dfs.append(pd.read_csv(filepath))
    df = pd.concat(dfs)

    # TODO: this might break other parts of the pipeline
    # WARNING, this is slightly untested
    # TLDR; I started making social orientation predictions at the end of training runs and saved the result to the prediction column.
    # the default social orientation schema is conversation_id,chunk_id,utterance_id,speaker,social_orientation
    # so, we'll keep conversation_id,chunk_id,utterance_id,speaker and
    # if 'prediction' is a column, we'll set it to social_orientation
    if 'prediction' in df.columns:
        cols = [
            'conversation_id', 'chunk_id', 'utterance_id', 'speaker',
            'prediction'
        ]
        df = df[cols]
        df['social_orientation'] = df['prediction']
        df.drop(columns=['prediction'], inplace=True)

    # drop duplicates
    df.drop_duplicates(subset=['conversation_id', 'utterance_id', 'speaker'],
                       inplace=True,
                       keep='last')

    # sort by conversation_id and utterance_id
    df.sort_values(by=['conversation_id', 'utterance_id'], inplace=True)
    return df


def load_cga_data(corpus):
    df = corpus.get_utterances_dataframe()
    # filter out section headers for now but need to generalize this
    # to include section header in the first utterance of a conversation
    df = df[df['meta.is_section_header'] == False]
    # strip out extra whitespace before dropping nan utterances
    df['text'] = df['text'].apply(lambda x: x.strip())
    # filter out nan utterances
    df = df[df['text'].notna()]

    # reset df index to ensure we don't lose that information
    df = df.reset_index()
    # sort conversations by id and timestamp
    df.sort_values(by=['conversation_id', 'timestamp'], inplace=True)
    return df


def load_cga_cmv_data(corpus):
    """Load the conversations-gone-awry-cmv-corpus data."""
    # load the data
    df = corpus.get_utterances_dataframe()
    df_len = len(df)
    convo_df = corpus.get_conversations_dataframe()
    # merge convo_df with df, with left_on='conversation_id', right on convo_df.index
    df = df.merge(
        convo_df[['meta.pair_id', 'meta.has_removed_comment', 'meta.split']],
        left_on='conversation_id',
        right_index=True,
        how='left')
    assert len(df) == df_len

    # strip out extra whitespace
    df['text'] = df['text'].apply(lambda x: x.strip())

    # reset the index so we retain the utterance ids
    df.reset_index(inplace=True)
    return df


def load_data(data_dir,
              include_speakers=True,
              social_orientation_filepaths=None,
              include_social_orientation=False,
              drop_missing=False):
    """Load the data from disk. If include_speakers is True, then
    prepend the speaker to the utterance."""
    _, corpus_name = os.path.split(data_dir)
    log(f'Loading {corpus_name} data...')
    try:
        corpus = Corpus(data_dir)
    except FileNotFoundError:
        # download the data
        log('Downloading data...')
        data_dir, corpus_name = os.path.split(data_dir)
        data_dir = download(corpus_name, data_dir=data_dir)
        corpus = Corpus(data_dir)
        log(f'Downloaded data to: {data_dir}')

    if corpus_name == 'conversations-gone-awry-corpus':
        df = load_cga_data(corpus)
    elif corpus_name == 'conversations-gone-awry-cmv-corpus':
        df = load_cga_cmv_data(corpus)

    # if social orientation filepath is passed, add social orientation labels
    if social_orientation_filepaths is not None:
        social_orientation_df = load_social_orientation_data(
            social_orientation_filepaths)
        df = merge_social_orientation_labels(df,
                                             social_orientation_df,
                                             drop_missing=drop_missing)

    # this will clobber the original text field, so retain it
    df['original_text'] = df['text']
    # prepare rows of data
    df = df.apply(prepare_example_dict,
                  axis=1,
                  include_speakers=include_speakers,
                  include_social_orientation=include_social_orientation)
    return df, corpus


def prepare_data_splits(df, corpus, data_dir):
    """Carve out conversation IDs for the train/val/test splits and save them to
    disk for a fixed train/val/test split.
    """
    _, corpus_name = os.path.split(data_dir)
    logging.info(f'Preparing data splits for {corpus_name}...')

    if corpus_name == 'conversations-gone-awry-corpus':
        convo_df = corpus.get_conversations_dataframe()
        # use official train, val, test splits
        train_convo_ids = convo_df[convo_df['meta.split'] ==
                                   'train'].index.tolist()
        val_convo_ids = convo_df[convo_df['meta.split'] ==
                                 'val'].index.tolist()
        test_convo_ids = convo_df[convo_df['meta.split'] ==
                                  'test'].index.tolist()
    elif corpus_name == 'conversations-gone-awry-cmv-corpus':
        # cga_cmv_corpus comes with its own splits that we can use
        train_convo_ids = df[df['meta.split'] ==
                             'train']['conversation_id'].unique().tolist()
        val_convo_ids = df[df['meta.split'] ==
                           'val']['conversation_id'].unique().tolist()
        test_convo_ids = df[df['meta.split'] ==
                            'test']['conversation_id'].unique().tolist()
    # shuffle the training split again
    random.shuffle(train_convo_ids)

    # save the splits to disk
    splits = {
        'train': train_convo_ids,
        'val': val_convo_ids,
        'test': test_convo_ids
    }
    with open(os.path.join(data_dir, 'splits.json'), 'w') as f:
        json.dump(splits, f)
    return splits


def get_data_splits(df, data_dir, subset_pct=1.0, corpus=None):
    try:
        # load the splits from disk
        with open(os.path.join(data_dir, 'splits.json'), 'r') as f:
            splits = json.load(f)
    except FileNotFoundError:
        if corpus is None:
            raise ValueError(
                'If splits.json does not exist, a corpus must be passed or you must call prepare_data_splits().'
            )
        # if the splits don't exist, create them
        splits = prepare_data_splits(df, corpus, data_dir)
    train_convo_ids = splits['train']
    val_convo_ids = splits['val']
    test_convo_ids = splits['test']
    # subset set dataset, if specified, for rapid development
    if subset_pct < 1.0:
        logging.info(
            f'Subsetting training dataset to {subset_pct*100:.2f}% of original size.'
        )
        # select subset documents
        train_convo_ids = random.sample(train_convo_ids,
                                        int(len(train_convo_ids) * subset_pct))
    train_df = df[df['conversation_id'].isin(train_convo_ids)]
    val_df = df[df['conversation_id'].isin(val_convo_ids)]
    test_df = df[df['conversation_id'].isin(test_convo_ids)]
    return train_df, val_df, test_df, splits


def get_tokenizer(args, labels=[]):
    # specifying model_max_length=args.max_seq_length to avoid HF warning about T5's max length
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path, model_max_length=args.max_seq_length)
    # if we load an existing tokenizer, return it
    if os.path.exists(args.tokenizer_name_or_path):
        return tokenizer

    # otherwise, we might be modifying a pretrained tokenizer

    # if specified, add special tokens to the tokenizer
    # retain the idxs to make up the newly added tokens so that we can take
    # their average when initializing the embeddings in the model
    tokens_to_ids = {}
    if args.add_tokens:
        special_tokens = {}
        # add <sep> or <eos> tokens if they don't exist
        for token_name, token in zip(('sep_token', 'eos_token'),
                                     ('<sep>', '<eos>')):
            if token_name not in tokenizer.special_tokens_map:
                special_tokens[token_name] = token
                # init to average of currently tokenized <sep> or <eos>
                idxs = tokenizer(token,
                                 add_special_tokens=False,
                                 return_attention_mask=False)['input_ids']
                tokens_to_ids[token] = idxs
        tokenizer.add_special_tokens(special_tokens)

        # add labels tokens, where they don't already exist
        label_tokens = []
        for label in labels:
            # if the token isn't already as single token, add it to the tokenizer
            if len(tokenizer.tokenize(label, add_special_tokens=False)) > 1:
                # TODO: determine if there's a way to add '_' prefix to start of tokens
                # init to average of tokenized label
                idxs = tokenizer(label,
                                 add_special_tokens=False,
                                 return_attention_mask=False)['input_ids']
                tokens_to_ids[label] = idxs
                label_tokens.append(label)
        tokenizer.add_tokens(label_tokens)
    return tokenizer, tokens_to_ids


def create_datasets(df, args, tokenizer, corpus=None):
    train_df, val_df, test_df, splits = get_data_splits(df,
                                                        args.data_dir,
                                                        args.subset_pct,
                                                        corpus=corpus)
    if args.dataset == 'cga':
        train_dataset = CGADataset(train_df,
                                   tokenizer,
                                   window_size=args.window_size,
                                   max_len=args.max_seq_length,)
        val_dataset = CGADataset(val_df,
                                 tokenizer,
                                 window_size=args.window_size,
                                 max_len=args.max_seq_length,)
        test_dataset = None
        if len(test_df) > 0:
            test_dataset = CGADataset(test_df,
                                      tokenizer,
                                      window_size=args.window_size,
                                      max_len=args.max_seq_length,)
    elif args.dataset == 'social-orientation':
        train_dataset = SocialOrientationDataset(train_df,
                                                 tokenizer,
                                                 window_size=args.window_size,
                                                 max_len=args.max_seq_length,)
        val_dataset = SocialOrientationDataset(val_df,
                                               tokenizer,
                                               window_size=args.window_size,
                                               max_len=args.max_seq_length,)
        test_dataset = None
        if len(test_df) > 0:
            test_dataset = SocialOrientationDataset(
                test_df,
                tokenizer,
                window_size=args.window_size,
                max_len=args.max_seq_length,)
    elif args.dataset == 'cga-cmv':
        train_dataset = CGACMVDataset(
            train_df,
            tokenizer,
            window_size=args.window_size,
            max_len=args.max_seq_length,
            return_utterances=args.return_utterances,
            return_labels=not args.dont_return_labels,
            disable_prepared_inputs=args.disable_prepared_inputs)
        val_dataset = CGACMVDataset(
            val_df,
            tokenizer,
            window_size=args.window_size,
            max_len=args.max_seq_length,
            return_utterances=args.return_utterances,
            return_labels=not args.dont_return_labels,
            disable_prepared_inputs=args.disable_prepared_inputs)
        test_dataset = None
        if len(test_df) > 0:
            test_dataset = CGACMVDataset(
                test_df,
                tokenizer,
                window_size=args.window_size,
                max_len=args.max_seq_length,
                return_utterances=args.return_utterances,
                return_labels=not args.dont_return_labels,
                disable_prepared_inputs=args.disable_prepared_inputs)

    # sanity check distributions of labels and lengths of inputs
    if args.log_level == 'DEBUG':
        for split, dataset in [('Train', train_dataset),
                               ('Validation', val_dataset),
                               ('Test', test_dataset)]:
            if dataset is None:
                continue
            # get labels
            logging.info(f'Retrieving all examples from {split} dataset...')
            labels = [dataset[i]['label'] for i in range(len(dataset))]
            # get class counts
            counts = Counter(labels)
            logging.debug(
                f'{split} dataset class counts: {counts.most_common()}')
            lengths = [
                len(dataset[i]['input_ids']) for i in range(len(dataset))
            ]
            logging.debug(f'{split} dataset input length statistics:')
            logging.debug(pd.DataFrame(lengths).describe())
        # decode a couple of inputs to make sure everything looks good
        logging.debug('Decoding a couple of inputs...')
        for i in range(2):
            logging.debug(
                f"{tokenizer.decode(train_dataset[i]['input_ids'])}\n")
    return train_dataset, val_dataset, test_dataset


def get_data_loaders(args, tokenizer):
    """Get the data loaders for the model."""
    # load the data
    df, corpus = load_data(
        args.data_dir,
        include_speakers=args.include_speakers,
        social_orientation_filepaths=args.social_orientation_filepaths,
        include_social_orientation=args.include_social_orientation,
        drop_missing=args.drop_missing)
    # create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(df,
                                                               args,
                                                               tokenizer,
                                                               corpus=corpus)

    train_sampler = None
    if args.distributed:
        # only use distributed sampler for training data because this sampler
        # pads the dataset to be divisible by the number of processes
        # which will mess up the validation and test data
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=not args.disable_train_shuffle,
            seed=args.seed)

    # create data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=train_dataset.tokenizer,
        padding='longest',
        max_length=args.max_seq_length,
        return_tensors='pt')
    # create data loaders
    # shuffle train if no sampler is used and train shuffle is not disabled
    shuffle = train_sampler is None and not args.disable_train_shuffle
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=shuffle,
                              num_workers=args.num_dataloader_workers,
                              collate_fn=data_collator,
                              sampler=train_sampler)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_dataloader_workers,
                            collate_fn=data_collator)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_dataloader_workers,
                             collate_fn=data_collator)
    return train_loader, val_loader, test_loader


def main(args):
    df, corpus = load_data(
        args.data_dir,
        include_speakers=args.include_speakers,
        social_orientation_filepaths=args.social_orientation_filepaths,
        include_social_orientation=args.include_social_orientation)

    # optionally define train/val/test splits
    if args.prepare_data_splits:
        prepare_data_splits(df, corpus, args.data_dir)

    # create datasets
    label2id, id2label = get_labels(args)
    tokenizer, tokens2ids = get_tokenizer(args, label2id.keys())

    train_dataset, val_dataset, test_dataset = create_datasets(df,
                                                               args,
                                                               tokenizer,
                                                               corpus=corpus)

    # get data loaders
    # load the tokenizer
    train_loader, val_loader, test_loader = get_data_loaders(args, tokenizer)

    # get a sample batch
    batch = next(iter(train_loader))

    # decode a couple of inputs to make sure everything looks good
    if args.log_level == 'DEBUG':
        logging.debug('Decoding a couple of inputs...')
        for i in range(2):
            logging.debug(
                f"{train_loader.dataset.tokenizer.decode(batch['input_ids'][i])}\n"
            )


if __name__ == '__main__':
    args = parse_args()
    main(args)
