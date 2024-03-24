"""This module is responsible for parsing responses from GPT models.

Examples:
    $ python parse.py \
        --gpt-outputs-filepath \
            data/gpt-4-output/train_results_gpt4.jsonl \
            data/gpt-4-output/train-long_results_gpt4.jsonl \
            data/gpt-4-output/val_results_gpt4.jsonl \
            data/gpt-4-output/val-long_results_gpt4.jsonl \
            data/gpt-4-output/test_results_gpt4.jsonl \
            data/gpt-4-output/test-long_results_gpt4.jsonl \
        --parse-output-dir \
            data/gpt-4-cga-social-orientation-labels
"""
from args import parse_args
import os
import logconfig
import logging
import difflib

import pandas as pd

from data import load_data, SOCIAL_ORIENTATION_LABEL2ID
from utils import merge_social_orientation_labels

def parse_row(row):
    """Parse a row of markdown formatted data.
    | utterance_id | participant_id | text | --> dict
    """
    raw_row = row
    # if the row is malformed, return None
    if (not row.startswith('| ')) or (not row.endswith(' |')):
        return None
    
    # remove the leading and trailing pipes
    row = row[2:-2]
    row = row.split(' | ')
    utterance_id = row[0]
    # try to cast to int
    try:
        utterance_id = int(utterance_id)
    except ValueError:
        return None
    
    try:
        participant_id = row[1]
    except IndexError:
        return None

    # rejoin in case there are ' | ' in the text
    text = ' | '.join(row[2:])
    return {
        'utterance_id': utterance_id,
        'speaker': participant_id,
        'label': text,
    }


def parse_rows(rows):
    """Parse a list of rows."""
    parsed_rows = []
    contains_error = False
    for idx, row in enumerate(rows):
        parsed_row = parse_row(row)
        if parsed_row is not None:
            # if the utterance_id is not an int, then it's an error
            if not isinstance(parsed_row['utterance_id'], int):
                contains_error = True
                continue
            # if the utterance_id is not sequential, then it's an error
            if (idx != 0) and (parsed_row['utterance_id'] != parsed_rows[-1]['utterance_id'] + 1):
                contains_error = True
                continue
            parsed_rows.append(parsed_row)
        else:
            contains_error = True
    if len(parsed_rows) == 0:
        contains_error = True
    # if contains_error:
    #     breakpoint()
    return contains_error, parsed_rows


def closest_label(label):
    """Apply Levenshtein distance to find the closest label."""
    if pd.isna(label):
        return label
    # first check that the label isn't something bogus
    if len(label) > 25:
        return label
    # otherwise, find the closest label
    labels = list(SOCIAL_ORIENTATION_LABEL2ID.keys())
    closest_labels = difflib.get_close_matches(label, labels, n=1, cutoff=0.7)
    if len(closest_labels) == 0:
        # some manual fixes
        if label == 'Uncompromising':
            return 'Assured-Dominant'
        if label == 'Assertive':
            return 'Assured-Dominant'
        if label == 'Uncompromising-Steadfast':
            return 'Assured-Dominant'
        if label == 'Inquisitive':
            return 'Assured-Dominant'
        # otherwise return the original label
        return label
    
    logging.warning(f'Replacing label {label} with {closest_labels[0]}.')
    return closest_labels[0]

def parse_completion(completion):
    """This function is a hodge-podge of string cleanup of GPT responses."""
    header = '| Utterance ID | Speaker ID | Label |\n| --- | --- | --- |\n'
    if completion.startswith(header) or completion.startswith(header[2:]):
        rows = completion.split('\n')[2:]
    else:
        rows = completion.split('\n')
    # remove empty rows
    rows = [row for row in rows if len(row) > 0]

    # remove rows that don't start with a pipe (sometimes GPT wants to continue the conversation)
    rows = [row for row in rows if row.startswith('|')]
    # it's possible that this stripped out preceding text rows, so we need to check again for the header
    rows_string = '\n'.join(rows)
    if rows_string.startswith(header) or rows_string.startswith(header[2:]):
        rows = rows_string.split('\n')[2:]
    else:
        rows = rows_string.split('\n')
    return rows

def patch_rows(row):
    """This function patches any known issue responses from GPT with responses
    that were re-run manually."""
    if row['conversation_id'] == '309820950.149611.149611' and row['chunk_id'] == 0 and row['request']['model'] == 'gpt-4-0314':
        # the original response spewed nonsense
        rows = """| 1 | Jackiespeel | Unassured-Submissive |\n| 2 | LjL | Warm-Agreeable |\n| 3 | Martinevans123 | Unassuming-Ingenuous |\n| 4 | Jackiespeel | Unassured-Submissive |\n| 5 | Mirafra | Warm-Agreeable |\n| 6 | LjL | Arrogant-Calculating |\n| 7 | Martinevans123 | Unassuming-Ingenuous |\n| 8 | LjL | Unassuming-Ingenuous |\n| 9 | Mirafra | Unassured-Submissive |""".split('\n')
        row['rows'] = rows
    elif row['conversation_id'] == '53420222.119328.119328' and row['chunk_id'] == 0 and row['request']['model'] == 'gpt-4-0314':
        # the original response got the speakers out of order
        rows = """| 1 | Tamfang | Unassuming-Ingenuous |\n| 2 | Irgendwer | Unassured-Submissive |\n| 3 | Irgendwer | Unassuming-Ingenuous |\n| 4 | Tamfang | Unassuming-Ingenuous |\n| 5 | Irgendwer | Unassured-Submissive |\n| 6 | Tamfang | Unassuming-Ingenuous |""".split('\n')
        row['rows'] = rows
    elif row['conversation_id'] == '64076409.109440.109440' and row['chunk_id'] == 1 and row['request']['model'] == 'gpt-4-0314':
        # GPT4 didn't include pipes on the left side of the table or the right side of the table
        # then it renumbered the utterances and made up a few utterances
        rows = ['| 3 | 207.207.79.202 | Warm-Agreeable |', '| 4 | Ishu | Aloof-Introverted |', '| 5 | Ogthor | Warm-Agreeable |']
        row['rows'] = rows
    elif row['conversation_id'] == '318749222.42100.42100' and row['chunk_id'] == 1 and row['request']['model'] == 'gpt-4-0314':
        # GPT4 hallucinated 3 utterances at the end of the conversation
        rows = ['| 1 | Blueshirts | Unassuming-Ingenuous |', '| 2 | DCTT | Warm-Agreeable |', '| 3 | Mafia godfather | Assured-Dominant |']
        row['rows'] = rows
    elif row['conversation_id'] == '34364810.2636.2636' and row['chunk_id'] == 0 and row['request']['model'] == 'gpt-4-0314':
        # ignored the first 3 utterances
        rows = ['| 1 | Outbackjack | Gregarious-Extraverted |', '| 2 | Jiminy Krikkitt | Unassuming-Ingenuous |', '| 3 | 192.43.227.18 | Arrogant-Calculating |', '| 4 | Jiminy Krikkitt | Warm-Agreeable |']
        row['rows'] = rows
    elif row['conversation_id'] == '71944556.77482.77482' and row['chunk_id'] == 0 and row['request']['model'] == 'gpt-4-0314':
        # GPT4 hallucinated 3 utterances at the end of the conversation
        row['rows'] = row['rows'][:-3]
    elif row['conversation_id'] == '109626944.13153.13153' and row['chunk_id'] == 0 and row['request']['model'] == 'gpt-4-0314':
        # GPT4 hallucinated 3 utterances at the end of the conversation
        row['rows'] = row['rows'][:-3]
    elif row['conversation_id'] == '1902810.120015.120015' and row['chunk_id'] == 1 and row['request']['model'] == 'gpt-4-0314':
        # lots of hallucinations at the end of the conversation
        row['rows'] = row['rows'][:3]
    elif row['conversation_id'] == '446880892.128069.128069' and row['chunk_id'] == 1 and row['request']['model'] == 'gpt-4-0314':
        # GPT got the speakers out of order
        rows = ['| 2 | Andrew Lancaster | Warm-Agreeable |', '| 3 | Andrew Lancaster | Arrogant-Calculating |', '| 4 | JohnLloydScharf | Cold |', '| 5 | JohnLloydScharf | Arrogant-Calculating |']
        row['rows'] = rows
    elif row['conversation_id'] == '592996737.44960.44960' and row['chunk_id'] == 0 and row['request']['model'] == 'gpt-4-0314':
        # hallucinated an extra reply from Popcornduff at the end: remove it 
        rows = ['| 1 | The Millionth One | Warm-Agreeable |', '| 2 | The Millionth One | Warm-Agreeable |', '| 3 | The Millionth One | Warm-Agreeable |', '| 4 | The Millionth One | Warm-Agreeable |', '| 5 | The Millionth One | Warm-Agreeable |', '| 6 | The Millionth One | Warm-Agreeable |', '| 7 | Popcornduff | Unassured-Submissive |', '| 8 | The Millionth One | Gregarious-Extraverted |',]
        row['rows'] = rows
    return row

def parse_gpt_outputs(gpt_outputs_filepath, replace_labels=True, drop_issues=False):
    """Parse the GPT outputs file into a dataframe."""
    df = pd.read_json(gpt_outputs_filepath, lines=True)
    # determine if any call was successful for a particular conversation_id + chunk_id
    df['success'] = df['response'].apply(lambda x: isinstance(x, dict) and ('choices' in x))
    
    # groupby conversation_id and chunk_id, and if any call was successful then
    # the group is successful, accounting for the fact that we may have retried
    success_df = df.groupby(['conversation_id', 'chunk_id']).agg({'success': 'any'})
    success_counts = success_df['success'].value_counts()
    if False in success_counts.index:
        logging.warning(f'Found {success_counts[False]} unsuccessful conversation chunks; dropping them. You should inspect {gpt_outputs_filepath} to see what went wrong or set a breakpoint here to inspect the errors.')
    df = df[df['success']].reset_index(drop=True)

    # extract the response
    df['completion'] = df['response'].apply(lambda x: x['choices'][0]['message']['content'])

    # assess stop reasons
    df['stop_reason'] = df['response'].apply(lambda x: x['choices'][0]['finish_reason'])
    logging.debug(f"Stop reasons:\n{df['stop_reason'].value_counts()}")
    
    # split the response into rows
    df['rows'] = df['completion'].apply(lambda x: parse_completion(x))

    # patch any known issues
    df = df.apply(lambda x: patch_rows(x), axis=1)
    
    # parse response rows
    parsed_rows = []
    errors = []
    for idx, row in df.iterrows():
        contains_error, parsed_rows_ = parse_rows(row['rows'])
        if contains_error:
            errors.append((idx, row['rows']))
            breakpoint()
        parsed_rows.append(parsed_rows_)
    df['parsed_rows'] = parsed_rows
    
    logging.warning(f'Found parsing errors with {len(errors)}/{len(df)} conversation chunks.')
    
    # explode the lists in the 'parsed_rows' column
    df_exploded = df[['conversation_id', 'chunk_id', 'parsed_rows']].explode('parsed_rows')
    # extract the 'parsed_rows' dict keys into their own columns
    df_parsed_rows = pd.json_normalize(df_exploded['parsed_rows'])
    # drop the old 'parsed_rows' column from the exploded dataframe
    df_exploded = df_exploded.drop('parsed_rows', axis=1)

    # reset the index of the df_parsed_rows dataframe (this is needed for the next step)
    df_parsed_rows = df_parsed_rows.reset_index(drop=True)

    # concatenate the two dataframes
    final_df = pd.concat([df_exploded.reset_index(drop=True), df_parsed_rows], axis=1)
    
    # drop duplicates (in case there was overlap between chunks in the GPT call)
    # first, sort by conversation_id, chunk_id, and utterance_id
    final_df = final_df.sort_values(by=['conversation_id', 'chunk_id', 'utterance_id'])
    final_df = final_df.drop_duplicates(subset=['conversation_id', 'utterance_id'], keep='first')
    # final_df = final_df.drop(columns=['chunk_id'])
    # sort by conversation_id and utterance_id
    final_df = final_df.sort_values(by=['conversation_id', 'utterance_id'])
    final_df = final_df.reset_index(drop=True)

    if replace_labels:
        # determine which labels are valid
        valid_labels = final_df['label'].isin(SOCIAL_ORIENTATION_LABEL2ID)
        # replace invalid labels with the closest valid label
        final_df.loc[~valid_labels, 'label'] = final_df.loc[~valid_labels]['label'].apply(lambda x: closest_label(x))
    
    # determine which labels are valid
    valid_labels = final_df['label'].isin(SOCIAL_ORIENTATION_LABEL2ID)
    logging.debug(f'Valid labels:\n{valid_labels.value_counts()}')
    # drop invalid labels
    if drop_issues:
        logging.info(f'Dropping {len(final_df[~valid_labels])} rows with invalid labels.')
        # NB: many of these were the result of not filtering out empty comments
        # so the labels are often blank
        final_df = final_df[valid_labels]
        final_df = final_df.reset_index(drop=True)
    else:
        # replace invalid labels with Not Available
        final_df.loc[~valid_labels, 'label'] = 'Not Available'
    return final_df


def main(args):
    os.makedirs(args.parse_output_dir, exist_ok=True)

    # load original data so we can merge the parsed data with it
    # most importantly to clean up labels for empty strings
    original_df, corpus = load_data(args.data_dir, include_speakers=False)

    dfs = []
    for filepath in args.gpt_outputs_filepaths:
        logging.info(f'Parsing {filepath}...')
        # drop_issues = False because we can fill these with (Not Available) labels later
        df = parse_gpt_outputs(filepath, replace_labels=True, drop_issues=False)
        df.rename(columns={'label': 'social_orientation'}, inplace=True)
        df = df[['conversation_id', 'chunk_id', 'utterance_id', 'speaker', 'social_orientation']]
        logging.debug(f'Parsed conversations:\n{df.head()}')

        # merge the parsed data with the original data
        merged_df = merge_social_orientation_labels(original_df, df)
        # identify which rows have a social orientation label and 'text' == ''
        # then replace these labels with Not Available in df
        empty_rows_df = merged_df.loc[(merged_df['social_orientation'].notna()) & (merged_df['text'] == '')]
        # where ['conversation_id', 'utterance_id', 'speaker'] in empty_rows_df match df, replace the label with Not Available
        # TODO: refine this logic
        for idx, row in df.iterrows():
            if not empty_rows_df[(empty_rows_df['conversation_id'] == row['conversation_id']) & (empty_rows_df['utterance_id'] == row['utterance_id']) & (empty_rows_df['speaker'] == row['speaker'])].empty:
                df.loc[idx, 'social_orientation'] = 'Not Available'
        
        # print out label distribution
        logging.info(f'Label distribution for {filepath}: {df["social_orientation"].value_counts()}')
        
        # save only the parsed data (not the merged data)
        filename = os.path.basename(filepath).split('.')[0] + '_parsed.csv'
        save_filepath = os.path.join(args.parse_output_dir, filename)
        logging.info(f'Saving parsed conversations to {save_filepath}.')
        df.to_csv(save_filepath, index=False)
        dfs.append(df)
    
    # merge the parsed dataframes
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)

    # merge with the original data
    merged_df = merge_social_orientation_labels(original_df, df)
    merged_df.rename(columns={'label': 'social_orientation'}, inplace=True)
    logging.debug(f'Merged conversations:\n{merged_df.head()}')
    
    # identify the conversations that were part of the GPT call
    # though this isn't perfect because it's possible that some conversations
    # that were part of the GPT call don't wind up in the final df
    # TODO: create a separate EDA script to analyze the GPT outputs
    gpt_conversations = set(df['conversation_id'].unique())
    subset_df = merged_df[merged_df['conversation_id'].isin(gpt_conversations)]
    num_invalid_labels = len(subset_df[~subset_df['social_orientation'].isin(SOCIAL_ORIENTATION_LABEL2ID)])
    logging.warning(f'{num_invalid_labels}/{len(subset_df)} utterances have invalid labels.')
    # temp_df = subset_df[~subset_df['social_orientation'].isin(SOCIAL_ORIENTATION_LABEL2ID)]
    # print(temp_df['conversation_id'].value_counts())

if __name__ == '__main__':
    args = parse_args()
    main(args)