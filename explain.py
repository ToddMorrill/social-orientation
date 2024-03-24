"""This module implements the label-flipping experiment to show that the model
is indeed relying social orientation labels to make predictions.

The experiment gradually increases the number of flipped social orientation
labels in the conversation and measures the effect on the accuracy of the
model's predictions.

Examples:
    $ python explain.py \
        --model-dir model/distilbert-cga-cmv-distilbert-winsize-2 \
        --window-size all \
        --checkpoint best \
        --dataset cga-cmv \
        --data-dir data/convokit/conversations-gone-awry-cmv-corpus \
        --analysis-dir logs/analysis \
        --social-orientation-filepaths \
            data/predictions/cga-cmv-social/distilbert-base-uncased/train_winsize_2_model_distilbert-base-uncased.csv \
            data/predictions/cga-cmv-social/distilbert-base-uncased/val_winsize_2_model_distilbert-base-uncased.csv \
            data/predictions/cga-cmv-social/distilbert-base-uncased/test_winsize_2_model_distilbert-base-uncased.csv \
        --include-speakers \
        --include-social-orientation \
        --batch-size 256 \
        --add-tokens \
        --disable-train-shuffle \
        --disable-prepared-inputs
"""
from collections import Counter
from itertools import product
import math

import pandas as pd
import logconfig
import logging
import os
import random
from args import parse_args
from tqdm import tqdm

from callbacks import Accuracy
from data import get_data_loaders, get_labels, get_tokenizer, SOCIAL_ORIENTATION_LABEL2ID, SOCIAL_ORIENTATION_ID2LABEL, SOCIAL_ORIENTATION2VALENCE, prepare_example_dict
from predict import Predictor

SOCIAL_ORIENTATION_TAGS = list(SOCIAL_ORIENTATION_LABEL2ID.keys())

VALENCE_TAGS = {}
for tag, valence in SOCIAL_ORIENTATION2VALENCE.items():
    if valence not in VALENCE_TAGS:
        VALENCE_TAGS[valence] = []
    VALENCE_TAGS[valence].append(tag)

OTHER_VALENCE_TAGS = {}
for v in VALENCE_TAGS.keys():
    if v not in OTHER_VALENCE_TAGS:
        OTHER_VALENCE_TAGS[v] = []
    for other_v in VALENCE_TAGS.keys():
        if other_v != v:
            OTHER_VALENCE_TAGS[v].extend(VALENCE_TAGS[other_v])

OTHER_OPPOSITE_VALENCE_TAGS = {}
for v in VALENCE_TAGS.keys():
    if v not in OTHER_OPPOSITE_VALENCE_TAGS:
        OTHER_OPPOSITE_VALENCE_TAGS[v] = []
    for other_v in VALENCE_TAGS.keys():
        if v == 'Negative':
            OTHER_OPPOSITE_VALENCE_TAGS[v].extend(VALENCE_TAGS['Positive'])
        elif v == 'Positive':
            OTHER_OPPOSITE_VALENCE_TAGS[v].extend(VALENCE_TAGS['Negative'])
        else:
            # TODO: experiment with this
            OTHER_OPPOSITE_VALENCE_TAGS[v].extend(VALENCE_TAGS['Positive'])

def count_changes(original_results, new_results):
    # compare predictions on the original and corrupted val sets
    changed = {'pos2neg': 0, 'neg2pos': 0, 'same': 0}
    for i, (original, new) in enumerate(zip(original_results['predictions'], new_results['predictions'])):
        if original == new:
            changed['same'] += 1
        elif original == 'Civil' and new == 'Uncivil':
            changed['pos2neg'] += 1
        elif original == 'Uncivil' and new == 'Civil':
            changed['neg2pos'] += 1
    
    logging.info(f'Changed: {changed}')
    # normalize
    total = sum(changed.values())
    changed_norm = {k: v/total for k, v in changed.items()}
    logging.info(f'Changed (normalized): {changed_norm}')
    return changed

def score_preds(results, source='Original'):
    scoring = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for i, (prediction, label) in enumerate(zip(results['predictions'], results['labels'])):
        label = 'Uncivil' if label == 1 else 'Civil'
        if prediction == label:
            if prediction == 'Uncivil':
                scoring['TP'] += 1
            else:
                scoring['TN'] += 1
        else:
            if prediction == 'Uncivil':
                scoring['FP'] += 1
            else:
                scoring['FN'] += 1
    logging.info(f'{source} scoring: {scoring}')
    # normalize
    total = sum(scoring.values())
    scoring_normed = {k: v/total for k, v in scoring.items()}
    logging.info(f'{source} scoring (normalized): {scoring_normed}')
    return scoring

def flip_valence(social_orientation):
    """Flips the valence of a social orientation label by first determining
    the current valence and randomly sampling from the other valences' tags."""
    valence = SOCIAL_ORIENTATION2VALENCE[social_orientation]
    other_valences = OTHER_VALENCE_TAGS[valence]
    # other_valences = OTHER_OPPOSITE_VALENCE_TAGS[valence]
    new_social_orientation = random.choice(other_valences)
    # new_social_orientation = 'Warm-Agreeable'# 'Arrogant-Calculating' # 'Gregarious-Extraverted' # 'Cold'
    return new_social_orientation

def corrupt_labels(convo_df, percent_corrupt, valence=False):
    """Corrupts the social orientation labels in a conversation dataframe. This
    function corrupts in an incremental fashion, so that the % of corrupted
    labels is equal to the specified percent_corrupt and retains the original
    corruption status of the labels.
    """
    convo_len = len(convo_df)
    # identify indexes of rows that are already corrupted
    already_corrupted = convo_df[convo_df['corrupted'] == True].index
    num_already_corrupted = len(already_corrupted)
    # infer the % already corrupted
    percent_already_corrupted = num_already_corrupted / convo_len
    # calculate the number of new rows to corrupt
    if percent_corrupt < percent_already_corrupted:
        raise ValueError(f'percent_corrupt ({percent_corrupt}) must be greater than percent_already_corrupted ({percent_already_corrupted})')
    num_new_corrupted = math.ceil((percent_corrupt - percent_already_corrupted) * convo_len)
    # possible that we don't need to corrupt any new rows
    if num_new_corrupted <= 0:
        return convo_df
    # sample from the uncorrupted rows
    uncorrupted = convo_df[convo_df['corrupted'] == False].index.tolist()
    new_corrupted = random.sample(uncorrupted, num_new_corrupted)
    # corrupt the labels
    convo_df.loc[new_corrupted, 'corrupted'] = True
    if valence:
        convo_df.loc[new_corrupted, 'social_orientation'] = convo_df.loc[new_corrupted, 'social_orientation'].apply(lambda x: flip_valence(x))
    else:        
        # for each row, sample a new label from the remaining labels
        choices = []
        for i, row in convo_df.loc[new_corrupted].iterrows():
            # get the current label
            current_label = row['social_orientation']
            remaining_social_orientations = [s for s in SOCIAL_ORIENTATION_TAGS if s != current_label]
            # sample a new label
            new_label = random.choice(remaining_social_orientations)
            # add to the list of choices
            choices.append(new_label)
        # replace the labels
        convo_df.loc[new_corrupted, 'social_orientation'] = choices
    return convo_df

def count_co_occurrences(group_df):
    # for each speaker, get the set of social orientation labels for all other speakers in the conversation
    counter = Counter()
    idx = 0
    for _, row in group_df.iterrows():
        # select rows excluding idx
        other_rows = list(range(0, idx)) + list(range(idx+1, len(group_df)))
        # get set of social orientation labels for all other speakers in the conversation
        other_speakers_to_labels = set()
        group_df.iloc[other_rows].apply(lambda x: other_speakers_to_labels.update(x['social_orientation']), axis=1)
        # get cartesian product of social orientation labels for current speaker and all other speakers
        cartesian_product = product(row['social_orientation'], other_speakers_to_labels)
        # update counter
        counter.update(cartesian_product)
        idx += 1
    return counter

def get_conversation_co_occurrences(df):
    # get social orientiation tags for each speaker in each conversation
    speakers_to_labels = df.groupby(['conversation_id', 'speaker'])['social_orientation'].apply(set)
    speakers_to_labels = speakers_to_labels.to_frame().reset_index()
    co_occurrence_counts = speakers_to_labels.groupby('conversation_id').apply(count_co_occurrences)
    co_occurrence_counts = co_occurrence_counts.to_frame().reset_index().rename(columns={0: 'co_occurrence_counts'})
    return co_occurrence_counts

def filter_conversations(co_occurrence_counts, search_patterns=[('Assured-Dominant', 'Unassured-Submissive'), ('Unassured-Submissive', 'Assured-Dominant')]):
    # filter conversations that have the specified co-occurrence patterns
    for pattern in search_patterns:
        if pattern in co_occurrence_counts:
            return True
    return False

def intervention(social_orientation, mapping):
    if social_orientation in mapping:
        return mapping[social_orientation]
    return social_orientation

def intervention_experiment(loader, original_results, predictor, args, interaction_patterns=None, mapping=None, corruption_rate=0.0, valence=False):
    # retrieve the df underlying the loader
    df = loader.dataset.df

    # random intervention
    if corruption_rate > 0.0:
        logging.info(f'Corrupting {corruption_rate*100:.2f}% of labels. Valence: {valence}')
        df['corrupted'] = False
        df['original_social_orientation'] = df['social_orientation']
        # val_df = val_df.groupby('conversation_id', group_keys=False).apply(corrupt_labels, percent_corrupt=corruption_rate, valence=valence)
        df = corrupt_labels(df, percent_corrupt=corruption_rate, valence=valence)
    # controlled intervention
    else:
        logging.info(f'Intervening on conversations with the following co-occurrence patterns: {interaction_patterns}')
        logging.info(f'Intervention mapping: {mapping}')
        co_occurrence_counts = get_conversation_co_occurrences(df)
        subset_convos = co_occurrence_counts[co_occurrence_counts['co_occurrence_counts'].apply(lambda x: filter_conversations(x, interaction_patterns))]
        df['social_orientation'] = df['social_orientation'].apply(lambda x: intervention(x, mapping))

    # pull the original text column back in
    df['text'] = df['original_text']
    # prepare rows of data
    df = df.apply(prepare_example_dict,
                    axis=1,
                    include_speakers=args.include_speakers,
                    include_social_orientation=args.include_social_orientation,)
    # retokenize the text
    df['input_ids'] = predictor.tokenizer(
        df['text'].values.tolist(),
        add_special_tokens=False,
        max_length=args.max_seq_length,
        truncation=True,
        return_attention_mask=False)['input_ids']
    loader.dataset.df = df
    predictions, logits, labels = predictor.predict(loader)
    results = {
        'predictions': predictions,
        'logits': logits,
        'labels': labels
    }
    accuracy = Accuracy()
    accuracy.update(logits, labels)
    logging.info(f'Intervened accuracy: {accuracy.compute()*100:.2f}%')

    # filter down to predictions for the conversations that have the specified co-occurrence patterns
    if corruption_rate == 0.0:
        original_results_filt = {
            'predictions': [],
            'logits': [],
            'labels': [],
        }
        new_results_filt = {
            'predictions': [],
            'logits': [],
            'labels': [],
        }
        for idx in subset_convos.index:
            # get the original predictions
            original_results_filt['predictions'].append(original_results['predictions'][idx])
            original_results_filt['logits'].append(original_results['logits'][idx])
            original_results_filt['labels'].append(original_results['labels'][idx])
            # get the new predictions
            new_results_filt['predictions'].append(results['predictions'][idx])
            new_results_filt['logits'].append(results['logits'][idx])
            new_results_filt['labels'].append(results['labels'][idx])
    else:
        # no need to filter if we're corrupting labels
        original_results_filt = original_results
        new_results_filt = results
    
    # changes
    changes = count_changes(original_results_filt, new_results_filt)
    original_scoring = score_preds(original_results_filt, source='Original')
    new_scoring = score_preds(new_results_filt, source='New')
    return changes, original_scoring, new_scoring


def main(args):
    # get the tokenizer
    # TODO: save the tokenizer, especially if we've modified it
    # the current approach is to transform the tokenizer in the exact same way
    # as was done during training, but this this is error-prone
    label2id, id2label = get_labels(args)
    added_tokens = SOCIAL_ORIENTATION_LABEL2ID.keys() if args.add_tokens else []
    tokenizer, tokens2ids = get_tokenizer(args, added_tokens)
     
    # load data
    train_loader, val_loader, test_loader = get_data_loaders(args, tokenizer)
    predictor = Predictor(args, tokenizer, tokens2ids=tokens2ids, id2label=id2label, label2id=label2id)
    sample = 'I am a very social person!'
    predictions = predictor.predict(sample)
    logging.debug(predictions[0])

    # original val results
    predictions, logits, labels = predictor.predict(test_loader)
    test_results = {
        'predictions': predictions,
        'logits': logits,
        'labels': labels
    }
    accuracy = Accuracy()
    accuracy.update(logits, labels)
    logging.info(f'Original accuracy on the test set: {accuracy.compute()*100:.2f}%')

    # make a copy of the social_orientations so we can modify it several times
    test_loader.dataset.df['original_social_orientation'] = test_loader.dataset.df['social_orientation']
    test_loader.dataset.df['corrupted'] = False

    results = []
    # random interventions
    changes, original_scoring, new_scoring = intervention_experiment(test_loader, test_results, predictor, args, interaction_patterns=None, mapping=None, corruption_rate=1.0, valence=False)
    test_loader.dataset.df['social_orientation'] = test_loader.dataset.df['original_social_orientation']
    test_loader.dataset.df['corrupted'] = False

    results.append({'Intervention': 'Random', 'Pos2Neg': changes['pos2neg'], 'Neg2Pos': changes['neg2pos'], 'Same': changes['same']})
    changes, original_scoring, new_scoring = intervention_experiment(test_loader, test_results, predictor, args, interaction_patterns=None, mapping=None, corruption_rate=1.0, valence=True)
    # results.append({'Intervention': 'Random (Valence)', 'Pos2Neg': changes['pos2neg'], 'Neg2Pos': changes['neg2pos'], 'Same': changes['same']})
    test_loader.dataset.df['social_orientation'] = test_loader.dataset.df['original_social_orientation']
    test_loader.dataset.df['corrupted'] = False

    # filter for patterns, make targeted interventions
    interaction_patterns = [('Assured-Dominant', 'Unassured-Submissive'), ('Unassured-Submissive', 'Assured-Dominant')]
    mapping = {'Unassured-Submissive': 'Assured-Dominant'}
    changes, original_scoring, new_scoring = intervention_experiment(test_loader, test_results, predictor, args, interaction_patterns, mapping, corruption_rate=0.0, valence=False)
    # results.append({'Intervention': '(Assured-Dominant, Assured-Dominant)', 'Pos2Neg': changes['pos2neg'], 'Neg2Pos': changes['neg2pos'], 'Same': changes['same']})
    # restore the original social orientation labels
    test_loader.dataset.df['social_orientation'] = test_loader.dataset.df['original_social_orientation']

    interaction_patterns = [('Cold', 'Arrogant-Calculating'), ('Arrogant-Calculating', 'Cold')]
    mapping = {'Cold': 'Unassuming-Ingenuous', 'Arrogant-Calculating': 'Unassured-Submissive'}
    changes, original_scoring, new_scoring = intervention_experiment(test_loader, test_results, predictor, args, interaction_patterns, mapping, corruption_rate=0.0, valence=False)
    results.append({'Intervention': '(Unassuming-Ingenuous, Unassured-Submissive)', 'Pos2Neg': changes['pos2neg'], 'Neg2Pos': changes['neg2pos'], 'Same': changes['same']})
    test_loader.dataset.df['social_orientation'] = test_loader.dataset.df['original_social_orientation']

    interaction_patterns = [('Unassuming-Ingenuous', 'Unassured-Submissive'), ('Unassured-Submissive', 'Unassuming-Ingenuous')]
    mapping = {'Unassuming-Ingenuous': 'Arrogant-Calculating', 'Unassured-Submissive': 'Cold'}
    changes, original_scoring, new_scoring = intervention_experiment(test_loader, test_results, predictor, args, interaction_patterns, mapping, corruption_rate=0.0, valence=False)
    results.append({'Intervention': '(Arrogant-Calculating, Cold)', 'Pos2Neg': changes['pos2neg'], 'Neg2Pos': changes['neg2pos'], 'Same': changes['same']})
    test_loader.dataset.df['social_orientation'] = test_loader.dataset.df['original_social_orientation']

    interaction_patterns = [('Assured-Dominant', 'Unassuming-Ingenuous'), ('Unassuming-Ingenuous', 'Assured-Dominant'), ('Assured-Dominant', 'Unassured-Submissive'), ('Unassured-Submissive', 'Assured-Dominant'), ('Assured-Dominant', 'Warm-Agreeable'), ('Warm-Agreeable', 'Assured-Dominant'), ('Assured-Dominant', 'Gregarious-Extraverted'), ('Gregarious-Extraverted', 'Assured-Dominant')]
    mapping = {'Unassured-Submissive': 'Assured-Dominant', 'Unassuming-Ingenuous': 'Assured-Dominant', 'Warm-Agreeable': 'Assured-Dominant', 'Gregarious-Extraverted': 'Assured-Dominant'}
    changes, original_scoring, new_scoring = intervention_experiment(test_loader, test_results, predictor, args, interaction_patterns, mapping, corruption_rate=0.0, valence=False)
    results.append({'Intervention': '(Assured-Dominant, Assured-Dominant)', 'Pos2Neg': changes['pos2neg'], 'Neg2Pos': changes['neg2pos'], 'Same': changes['same']})
    test_loader.dataset.df['social_orientation'] = test_loader.dataset.df['original_social_orientation']

    # add results to DF and save
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(os.path.join(args.analysis_dir, 'explain_interventions.csv'), index=False)
    # save to LaTeX, use commas
    results_df.to_latex(os.path.join(args.analysis_dir, 'explain_interventions.tex'), index=False, na_rep='', formatters={'Pos2Neg': '{:,.0f}'.format, 'Neg2Pos': '{:,.0f}'.format, 'Same': '{:,.0f}'.format})

if __name__ == '__main__':
    args = parse_args()
    main(args)