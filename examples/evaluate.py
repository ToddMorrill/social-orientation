"""Loads social orientation labels from HF datasets, loads model from HF, and
evaluates test set accuracy. This is a full-blown example showing how to prepare
a real dataset for the social orientation model. This example would be pretty
straightforward to adapt for training."""
import os
import logging

from convokit import Corpus, download
from datasets import load_dataset, load_metric
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from tqdm import tqdm

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


def load_cga_data(corpus):
    df = corpus.get_utterances_dataframe()
    # filter out section headers
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


def prepare_example_dict(
    example_dict,
    include_speakers=True,
):
    """Prepare an example dict for the model. If include_speakers is True, then
    prepend the speaker to the utterance."""
    # strip out extra whitespace
    example_dict['text'] = example_dict['text'].strip()
    example_dict['speaker'] = example_dict['speaker'].strip()

    updated_string = ''
    if include_speakers:
        updated_string += example_dict['speaker'] + ': '
    # add the utterance
    updated_string += example_dict['text']
    example_dict['text'] = updated_string
    return example_dict


def load_data(
    data_dir,
    include_speakers=True,
):
    """Load the CGA data and corresponing social orientation labels. If
    include_speakers=True, then prepend the speaker to the utterance."""
    _, corpus_name = os.path.split(data_dir)
    logging.info(f'Loading {corpus_name} data...')
    try:
        corpus = Corpus(data_dir)
    except FileNotFoundError:
        # download the data
        logging.info('Downloading data...')
        data_dir, corpus_name = os.path.split(data_dir)
        data_dir = download(corpus_name, data_dir=data_dir)
        corpus = Corpus(data_dir)
        logging.info(f'Downloaded data to: {data_dir}')

    df = load_cga_data(corpus)

    # load social orientation labels
    social_labels = load_dataset('tee-oh-double-dee/social-orientation')
    social_labels_df = social_labels['train'].to_pandas()
    # merge in social orientation labels on id
    df = df.merge(social_labels_df, on='id')

    # prepend text utterances with speaker information
    # this will clobber the original text field, so retain it
    df['original_text'] = df['text']
    # prepare rows of data
    df = df.apply(prepare_example_dict,
                  axis=1,
                  include_speakers=include_speakers)
    return df, corpus


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


class SocialOrientationDataset(DialogueDataset):
    """Dataset for social orientation tagging."""

    def __init__(
        self,
        df,
        tokenizer,
        window_size=2,
        default_label='Not Available',
        max_len=None,
    ):
        super().__init__(
            df,
            tokenizer,
            window_size,
            default_label,
            max_len,
        )

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
            breakpoint()
            # inference mode
            input_dict = {'input_ids': input_ids}
        self.prepared_inputs[idx] = input_dict
        return input_dict


def main():
    data_dir = '../data/convokit/conversations-gone-awry-corpus'  # cache directory for convokit data
    df, corpus = load_data(data_dir)
    convo_df = corpus.get_conversations_dataframe()
    # use official train, val, test splits
    train_convo_ids = convo_df[convo_df['meta.split'] ==
                               'train'].index.tolist()
    val_convo_ids = convo_df[convo_df['meta.split'] == 'val'].index.tolist()
    test_convo_ids = convo_df[convo_df['meta.split'] == 'test'].index.tolist()
    train_df = df[df['conversation_id'].isin(train_convo_ids)]
    val_df = df[df['conversation_id'].isin(val_convo_ids)]
    test_df = df[df['conversation_id'].isin(test_convo_ids)]
    tokenizer = AutoTokenizer.from_pretrained(
        'tee-oh-double-dee/social-orientation')
    train_dataset = SocialOrientationDataset(train_df, tokenizer)
    val_dataset = SocialOrientationDataset(val_df, tokenizer)
    test_dataset = SocialOrientationDataset(test_df, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=train_dataset.tokenizer,
                                            padding='longest',
                                            max_length=512,
                                            return_tensors='pt')
    train_loader = DataLoader(train_dataset,
                              batch_size=64,
                              shuffle=False,
                              collate_fn=data_collator)
    val_loader = DataLoader(val_dataset,
                            batch_size=64,
                            shuffle=False,
                            collate_fn=data_collator)
    test_loader = DataLoader(test_dataset,
                             batch_size=64,
                             shuffle=False,
                             collate_fn=data_collator)
    model = AutoModelForSequenceClassification.from_pretrained(
        'tee-oh-double-dee/social-orientation')
    # push to GPU, if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # train eval
    metric = load_metric('accuracy')
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)
    acc = metric.compute()['accuracy']
    print(f'Training set accuracy: {acc*100:.2f}%')

    # val eval
    metric = load_metric('accuracy')
    for batch in tqdm(val_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)
    acc = metric.compute()['accuracy']
    print(f'Validation set accuracy: {acc*100:.2f}%')

    # test eval
    metric = load_metric('accuracy')
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)
    acc = metric.compute()['accuracy']
    print(f'Test set accuracy: {acc*100:.2f}%')


if __name__ == '__main__':
    main()
