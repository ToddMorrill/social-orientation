"""This module is responsible for generating prompts for GPT models.

Examples:    
    $ python gpt_prompts.py \
        --gpt-mode train \
        --gpt-model gpt-4 \
        --gpt-data-dir data/gpt-4-input \
        --calculate-cost
"""
from args import parse_args
import os
import logconfig
import logging
from tqdm import tqdm
import warnings

import pandas as pd
import tiktoken

import utils
from data import load_data, get_data_splits
from utils import add_utterance_id, create_line, load_prompt

# filter pandas warning
warnings.simplefilter('ignore', DeprecationWarning)
# ignore pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None

GPT_3_5_TURBO_TOKEN_LIMIT = 4096
GPT_4_TURBO_TOKEN_LIMIT = 8192 / 2 # 8192 is the max token limit for GPT-4, but can't ever use it based on experience
GPT_3_5_PRICE = 0.002
GPT_4_PRICE = 0.03


def assign_chunks(df, max_input_length, encoding, overlap=2, min_utterances=2):
    """Assigns each row to a contiguous chunk of text that is less than the max_input_length. Just before a chunk will exceed the max_input_length, a new chunk is created.
    """
    if overlap >= min_utterances:
        raise ValueError('overlap must be less than min_utterances')
    start = 0
    end = max_input_length
    idx_start = df.iloc[0].name
    idx_end = None
    chunk_id = 0
    dfs = []
    last_chunk_was_long = False
    while idx_end != df.iloc[-1].name:
        # retrieve slice that's between the current start and current end
        chunk_df = df[(df['length_cumsum'] > start)
                      & (df['length_cumsum'] <= end)]
        
        # if len(chunk_df) is less than min_utterances, then start trimming longest utterances
        # until the length is less than min_utterances
        if len(chunk_df) < min_utterances:
            last_chunk_was_long = True
            encoded_utterances = df.loc[idx_start:idx_start + min_utterances]['gpt_line_encoded'].values.tolist()
            total_len = sum([len(utterance) for utterance in encoded_utterances])
            
            # subtract off extra capacity for each utterance and add back the space and | at the end
            while total_len > (max_input_length - (min_utterances*2)):
                # find the longest utterance
                longest_utterance_idx = max(range(len(encoded_utterances)), key=lambda idx: len(encoded_utterances[idx]))
                # remove the last token from the longest utterance
                encoded_utterances[longest_utterance_idx] = encoded_utterances[longest_utterance_idx][:-1]
                # update the total length
                total_len -= 1
            
            ending = encoding.encode(' |')
            # for any encoded utterance that doesn't end with ending, add ending
            final_encoded_utterances = []
            for utterance in encoded_utterances:
                utterance_ending = utterance[-len(ending):]
                if utterance_ending != ending:
                    utterance += ending
                final_encoded_utterances.append(utterance)
            encoded_utterances = final_encoded_utterances

            # update the chunk_df. copy to avoid modifying the original df
            chunk_df = df.loc[idx_start:idx_start + min_utterances].copy(deep=True)
            chunk_df['gpt_line_encoded'] = encoded_utterances
            chunk_df['gpt_line'] = [encoding.decode(utterance) for utterance in encoded_utterances]
        else:
            last_chunk_was_long = False
        idx_start = chunk_df.iloc[0].name
        idx_end = chunk_df.iloc[-1].name

        # assign chunk_id to the slice
        chunk_df.loc[idx_start:idx_end, 'chunk_id'] = chunk_id
        dfs.append(chunk_df)
        chunk_id += 1

        # attempt to overlap the next chunk by overlap utterances if 
        # it's not the last chunk and the last chunk wasn't long, where
        # the latter condition is to avoid an infinite loop, where we rewind
        # idx_end back to the same value as idx_start (i.e. the start of the 
        # long chunk)
        if (idx_end != df.iloc[-1].name) and (not last_chunk_was_long):
            idx_end -= overlap

        # update start and end
        start = df.loc[idx_end]['length_cumsum']
        end = start + max_input_length
        idx_start = idx_end

    # return df
    return pd.concat(dfs)


def group_cumsum(group_df):
    group_df['length_cumsum'] = group_df['length'].cumsum()
    return group_df



def create_gpt_data(conversation, system_prompt, prompt):
    final_prompt = prompt + conversation
    # format GPT messages
    messages = [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': final_prompt
        },
    ]
    return messages


def create_convos_df(df, include_chunk_id=False):
    group_cols = ['conversation_id']
    if include_chunk_id:
        # this is used when we have to split the conversation into chunks
        group_cols.append('chunk_id')
    convos_df = df.groupby(group_cols)['gpt_line'].apply(lambda x: '\n'.join(list(x))).to_frame().reset_index()
    convos_df.rename(columns={'gpt_line': 'conversation'}, inplace=True)
    return convos_df

def main(args):
    # load data
    df, corpus = load_data(args.data_dir, include_speakers=False)

    # reset the index so we have an integer index to work with later
    df = df.reset_index()

    # create utterance ids per file_id
    df = df.groupby('conversation_id', group_keys=False).apply(add_utterance_id)

    # create GPT line
    df['gpt_line'] = df.apply(create_line, axis=1)

    # load prompt
    prompt = load_prompt(args.prompt_filepath)

    # determine max input length
    encoding = tiktoken.encoding_for_model(args.gpt_model)
    prompt_length = len(encoding.encode(prompt))
    logging.info(f'Prompt length: {prompt_length}')
    token_limit = GPT_3_5_TURBO_TOKEN_LIMIT if '3.5' in args.gpt_model else GPT_4_TURBO_TOKEN_LIMIT
    # subtract prompt length from token limit and use specified portion for input tokens
    max_input_length = int(
        (token_limit - prompt_length) * (args.data_token_pct))
    logging.info(
        f'Max conversation token length (excluding prompt): {max_input_length}'
    )
    logging.info(
        f'Max generative length: {token_limit - max_input_length - prompt_length}'
    )

    # get training data information so we're developing on the training set
    _, _, _, splits = get_data_splits(df, args.data_dir)

    # get lengths of conversations
    convos_df = create_convos_df(df)
    convos_df['length'] = convos_df['conversation'].apply(lambda x: len(encoding.encode(x)))
    short_convos_df = convos_df[convos_df['length'] <= max_input_length]
    logging.info(f'{len(short_convos_df):,}/{len(convos_df):,} conversations are short enough to be used with GPT based on a max conversation length of {max_input_length:,}.')

    # if sample mode, just use the first 10 conversations from the training set
    if args.gpt_mode == 'sample':
        # filter short_convos_df to only include conversations in the training set
        train_short_convos_df = short_convos_df[short_convos_df['conversation_id'].isin(set(splits['train']))]
        # get first 10 conversations
        sample_convo_ids = train_short_convos_df['conversation_id'].values.tolist()[:10]
        convos_df = convos_df[convos_df['conversation_id'].isin(sample_convo_ids)] 
        # add in a chunk_id, which will be useful for consistency later
        convos_df['chunk_id'] = 0
        filename = 'sample.jsonl'
    elif args.gpt_mode == 'short':
        # add in a chunk_id, which will be useful for consistency later
        convos_df['chunk_id'] = 0
    elif args.gpt_mode == 'long-remainder':
        pass
    elif args.gpt_mode == 'all':
        pass
    elif args.gpt_mode in {'train', 'val', 'test'}:
        # filter short_convos_df to only include conversations in the specified set
        short_convos_df = short_convos_df[short_convos_df['conversation_id'].isin(set(splits[args.gpt_mode]))]
        convo_ids = set(short_convos_df['conversation_id'].values.tolist())
        convos_df = convos_df[convos_df['conversation_id'].isin(convo_ids)] 
        # add in a chunk_id, which will be useful for consistency later
        convos_df['chunk_id'] = 0
        filename = f'{args.gpt_mode}.jsonl'
    elif args.gpt_mode in {'train-long', 'val-long', 'test-long'}:
        # filter short_convos_df to only include conversations in the specified set
        split = args.gpt_mode.split('-')[0]
        short_convos_df = short_convos_df[short_convos_df['conversation_id'].isin(set(splits[split]))]
        convo_ids = set(splits[split]) - set(short_convos_df['conversation_id'].values.tolist())
        # get conversations that are not short conversations (i.e. long conversations) 
        long_convos_df = convos_df[convos_df['conversation_id'].isin(convo_ids)]

        # filter df to only include long conversations
        df = df[df['conversation_id'].isin(set(long_convos_df['conversation_id'].values.tolist()))]
        
        # create cumulative sum of encoding lengths to easier chunking
        df['gpt_line_encoded'] = df['gpt_line'].apply(lambda x: encoding.encode(x))
        df['length'] = df['gpt_line_encoded'].apply(lambda x: len(x))
        df = df.groupby('conversation_id', group_keys=False).apply(group_cumsum)
        logging.info('Assigning utterances to chunks..')
        group_dfs = []
        for convo_id, group_df in tqdm(df.groupby('conversation_id', group_keys=False)):
            temp_df = assign_chunks(group_df, max_input_length=max_input_length, encoding=encoding, overlap=args.overlap, min_utterances=args.min_utterances)
            group_dfs.append(temp_df)
        df = pd.concat(group_dfs)

        # create convos_df
        convos_df = create_convos_df(df, include_chunk_id=True)
        filename = f'{args.gpt_mode}.jsonl'
    
    # create GPT messages and save to disk
    # create messages for each conversation
    system_prompt = 'You are a helpful assistant.'
    convos_df['messages'] = convos_df['conversation'].apply(create_gpt_data, system_prompt=system_prompt, prompt=prompt)
    convos_df = convos_df[['conversation_id', 'chunk_id', 'messages']]
    # save to jsonl
    os.makedirs(args.gpt_data_dir, exist_ok=True)
    save_filepath = os.path.join(args.gpt_data_dir, filename)
    logging.info(f'Saving GPT prompt data to {save_filepath}.')
    convos_df.to_json(save_filepath, orient='records', lines=True)

    # optionally calculate cost (only look at input cost for now)
    if args.calculate_cost:
        price_per_thousand = GPT_3_5_PRICE if 'gpt-3.5' in args.gpt_model else GPT_4_PRICE
        logging.info(f'Calculating cost to send tokens to {args.gpt_model}..')
        token_count = 0
        for idx, messages in tqdm(convos_df['messages'].items()):
            token_count += utils.num_tokens_from_messages(messages, model=args.gpt_model)
        logging.info(f'Cost to send {token_count:,} tokens to {args.gpt_model}: ${(token_count/1000) * price_per_thousand:.2f}')

if __name__ == '__main__':
    args = parse_args()
    main(args)