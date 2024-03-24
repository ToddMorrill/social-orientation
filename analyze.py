"""Analyze model predictions.

Examples:
    $ python analyze.py \
        --social-orientation-filepaths \
            ~/Documents/data/circumplex/transformed/train_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/val_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/train-long_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/val-long_results_gpt4_parsed.csv
    
    $ python analyze.py \
        --analysis-mode t-test \
        --experiment-filepath logs/experiments/window_size_cga_distilbert-base-uncased.csv

    $ python analyze.py \
        --analysis-mode merge-data \
        --prediction-filepaths \
            ~/Documents/data/circumplex/predictions/social-orientation/train_winsize_2_model_distilbert-base-uncased.csv \
            ~/Documents/data/circumplex/predictions/social-orientation/val_winsize_2_model_distilbert-base-uncased.csv \
        --social-orientation-filepaths \
            ~/Documents/data/circumplex/transformed/train_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/val_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/train-long_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/val-long_results_gpt4_parsed.csv
    
    $ python analyze.py \
        --analysis-mode merge-data \
        --prediction-filepaths \
            ./predictions/social-orientation/train_winsize_6_model_microsoft_deberta-v3-large.csv \
            ./predictions/social-orientation/val_winsize_6_model_microsoft_deberta-v3-large.csv \
        --social-orientation-filepaths \
            ~/Documents/data/circumplex/transformed/train_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/val_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/train-long_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/val-long_results_gpt4_parsed.csv
    
    $ python analyze.py \
        --analysis-mode compare-cga \
        --prediction-filepaths \
            ./predictions/cga/distilbert-winsize-2-predicted-social/val_preds_cga_2_distilbert-base-uncased.csv \
            ./predictions/cga/distilbert-winsize-2-gpt4-social/val_preds_cga_2_distilbert-base-uncased.csv
    
    $ python analyze.py \
        --analysis-mode human-eval \
        --prediction-filepaths \
            ./predictions/cga/distilbert-winsize-2-gpt4-social/val_preds_cga_2_distilbert-base-uncased.csv \
            ./predictions/cga/distilbert-winsize-2-predicted-social/val_preds_cga_2_distilbert-base-uncased.csv \
        --human-annotation-filepaths \
            "./predictions/cga/human-annotations/Circumplex Annotation - Todd.csv" \
            "./predictions/cga/human-annotations/Circumplex Annotation - Amith.csv" \
            "./predictions/cga/human-annotations/Circumplex Annotation - Yanda.csv"
    
    $ python analyze.py \
        --analysis-mode data-ablation \
        --experiment-filepath logs/experiments/subset_cga_distilbert-base-uncased.csv
    
    $ python analyze.py \
        --analysis-mode explainability \
        --social-orientation-filepaths \
            ~/Documents/data/circumplex/transformed/train_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/val_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/train-long_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/val-long_results_gpt4_parsed.csv
    
    $ python analyze.py \
        --analysis-mode explainability \
        --social-orientation-filepaths \
            ./predictions/social-orientation/distilbert-winsize-2/train_preds_social-orientation_2_distilbert-base-uncased.csv \
            ./predictions/social-orientation/distilbert-winsize-2/val_preds_social-orientation_2_distilbert-base-uncased.csv
    
    $ python analyze.py \
        --analysis-mode explainability \
        --dataset casino-satisfaction \
        --data-dir ~/Documents/data/convokit/casino-corpus \
        --social-orientation-filepaths \
            predictions/casino-satisfaction-social/distilbert-base-uncased/train_winsize_2_model_distilbert-base-uncased.csv \
            predictions/casino-satisfaction-social/distilbert-base-uncased/val_winsize_2_model_distilbert-base-uncased.csv
    
    $ python analyze.py \
        --analysis-mode explainability \
        --dataset cga-cmv \
        --data-dir ~/Documents/data/convokit/conversations-gone-awry-cmv-corpus \
        --social-orientation-filepaths \
            predictions/cga-cmv-social/distilbert-base-uncased/train_winsize_2_model_distilbert-base-uncased.csv \
            predictions/cga-cmv-social/distilbert-base-uncased/val_winsize_2_model_distilbert-base-uncased.csv
    
    $ python analyze.py \
        --analysis-mode data-ablation \
        --experiment-filepath logs/experiments/subset_cga-cmv_distilbert-base-uncased.csv
    
    $ python analyze.py \
        --analysis-mode explainability \
        --dataset cga-cmv \
        --data-dir ~/Documents/data/convokit/conversations-gone-awry-cmv-corpus \
        --social-orientation-filepaths \
            predictions/cga-cmv-social/distilbert-base-uncased/train_winsize_2_model_distilbert-base-uncased.csv \
            predictions/cga-cmv-social/distilbert-base-uncased/val_winsize_2_model_distilbert-base-uncased.csv \
        --num-runs 5 \
        --subset-pcts 0.01 0.1 0.2 0.5 1.0
    
    $ python analyze.py \
        --analysis-mode data-ablation \
        --dataset cga-cmv \
        --experiment-filepath \
            logs/experiments/subset_cga-cmv_distilbert-base-uncased.csv \
            logs/experiments/subset_cga-cmv_logistic_clf.csv
    
    $ python analyze.py \
        --analysis-mode explainability \
        --dataset cga \
        --social-orientation-filepaths \
            ./predictions/social-orientation-social/distilbert-base-uncased/train_winsize_2_model_distilbert-base-uncased.csv \
            ./predictions/social-orientation-social/distilbert-base-uncased/val_winsize_2_model_distilbert-base-uncased.csv \
        --num-runs 5 \
        --subset-pcts 0.01 0.1 0.2 0.5 1.0 \
        --seed 43
    
    $ python analyze.py \
        --analysis-mode data-ablation \
        --experiment-filepath \
            logs/experiments/subset_cga_distilbert-base-uncased.csv \
            logs/experiments/subset_cga_logistic_clf.csv
    
    $ python analyze.py \
        --analysis-mode t-test \
        --experiment subset \
        --experiment-filepath \
            logs/experiments/subset_cga_distilbert-base-uncased.csv \
            logs/experiments/subset_cga_logistic_clf.csv
    
    $ python analyze.py \
        --analysis-mode t-test \
        --experiment subset \
        --dataset cga-cmv \
        --data-dir ~/Documents/data/convokit/conversations-gone-awry-cmv-corpus \
        --experiment-filepath \
            logs/experiments/subset_cga-cmv_distilbert-base-uncased.csv \
            logs/experiments/subset_cga-cmv_logistic_clf.csv
    
    $ python analyze.py \
        --analysis-mode gpt-4-preds \
        --dataset cga \
        --social-orientation-filepaths \
            ~/Documents/data/circumplex/transformed/train_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/val_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/train-long_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/val-long_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/test_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/test-long_results_gpt4_parsed.csv
    
    $ python analyze.py \
        --analysis-mode social-eval \
        --predicted-social-orientation-filepaths \
            predictions/social-orientation-social/distilbert-base-uncased/train_winsize_2_model_distilbert-base-uncased.csv \
            predictions/social-orientation-social/distilbert-base-uncased/val_winsize_2_model_distilbert-base-uncased.csv \
            predictions/social-orientation-social/distilbert-base-uncased/test_winsize_2_model_distilbert-base-uncased.csv \
        --social-orientation-filepaths \
            ~/Documents/data/circumplex/transformed/train_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/val_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/train-long_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/val-long_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/test_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/test-long_results_gpt4_parsed.csv
"""
import itertools
import logging
import random

import numpy as np
from args import parse_args
import os

import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.stats.inter_rater import aggregate_raters
from statsmodels.stats.inter_rater import fleiss_kappa
import seaborn as sns

from data import load_data, get_data_splits, SOCIAL_ORIENTATION_LABEL2ID, SocialOrientationDataset, CGA_LABEL2ID
from utils import set_random_seed

def plot_confusion_matrix(y_true, y_preds, labels, model_name=None, output_dir=None, xlabel='Predicted label', ylabel='True label', split=None):
    cm = confusion_matrix(y_true, y_preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    # ensure that full number is used, not scientific notation 
    sns.heatmap(cm_df, annot=True, fmt='g')
    # cm_display = ConfusionMatrixDisplay(
    #     cm, display_labels=labels).plot()
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    filename = f'confusion_matrix'
    title = 'Confusion Matrix'
    if model_name is not None:
        title += f' - {model_name}'
        filename += f'_{model_name}'
    if split is not None:
        title += f' ({split})'
        filename += f'_{split}'
    # plt.title(title)
    filename += '.png'
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename),
                    dpi=300,
                    bbox_inches='tight')
    plt.clf()

def t_test(group_df, experiment='window_size'):
    """Perform t-test on group_df."""
    # unstack by social_orientation
    unstack_df = group_df.pivot(index=['seed', experiment], columns=['social_orientation'], values=['val_acc', 'val_loss'])
    # perform t-test on val_acc, with social_orientation as the grouping variable
    # alternative hypothesis: E[val_acc | social_orientation==False] < E[val_acc | social_orientation==True]
    p_value = stats.ttest_ind(unstack_df['val_acc'][False], unstack_df['val_acc'][True], alternative='less').pvalue
    return p_value


def t_test_a_vs_b(
        group_df,
        method_a='distilbert',
        method_b='logistic_social_counts',
        method_a_social_features='None',
        method_b_social_features='winsize_2_model_distilbert-base-uncased',
        subset='test', alternative='less'):
    # NB: this function is largely geared toward subset pct experiments (as opposed to window size experiments, etc.)
    unstack_df = group_df.pivot(index=['method', 'seed', 'subset_pct'], columns=['social_orientation_prediction'], values=[f'{subset}_acc', f'{subset}_loss'])
    a_results = unstack_df.loc[method_a][(f'{subset}_acc', method_a_social_features)]
    b_results = unstack_df.loc[method_b][(f'{subset}_acc', method_b_social_features)]
    p_value = stats.ttest_ind(a_results, b_results, alternative=alternative).pvalue
    return p_value

def t_test_textpluspreds_vs_text(group_df):
    # TODO: try to generalize this
    unstack_df = group_df.pivot(index=['method', 'seed', 'subset_pct'], columns=['social_orientation_prediction'], values=['val_acc', 'val_loss'])
    text_pred_results = unstack_df.loc['distilbert'][('val_acc', 'winsize_2_model_distilbert-base-uncased')]
    text_results = unstack_df.loc['distilbert'][('val_acc', 'None')]
    p_value = stats.ttest_ind(text_results, text_pred_results, alternative='less').pvalue
    return p_value

def social_orientation_source(x):
    if not isinstance(x, str):
        return 'None'
    if 'gpt4' in x:
        return 'GPT-4'
    # otherwise, try to extract out some information about the source
    # assuming the filepath is of the form: ./dir/{train/val}_preds_{dataset}_{window_size}_{model_name}.csv
    # convert string to list of filepath string
    filepaths = eval(x)
    # get the first filepath
    filepath = filepaths[0]
    # split on underscores
    splits = filepath.split('_')
    # get the last 4 splits
    splits = splits[-4:]
    # join them back together
    splits = '_'.join(splits)
    # drop the .csv extension
    splits = splits.split('.')[0]
    return splits

def get_speakers_first_turn(group_df, speaker='mturk_agent_1'):
    """This function takes a conversation as input and returns the social orientation tag of the first turn by the specified speaker."""
    speaker_social_tag = group_df[group_df['speaker'] == speaker].iloc[0]['social_orientation']
    return speaker_social_tag

def simple_classifier(train_df, val_df, corpus, args, count_vector=True, social_source='gpt4'):
    """Trains a simple logistic classifier using only social orientation tags.
    
    If count_vector=True, uses counts of all social orientation tags in the conversation
    as features. Otherwise, creates a binary feature for each social orientation tag for
    the first 2 utterances in the conversation
    """
    # convert social orientation tags to integer labels
    # train_df['social_orientation'] = train_df['social_orientation'].apply(lambda x: SOCIAL_ORIENTATION_LABEL2ID[x])
    # val_df['social_orientation'] = val_df['social_orientation'].apply(lambda x: SOCIAL_ORIENTATION_LABEL2ID[x])

    if count_vector:
        # get counts of social orientation tags in the conversation
        # include all turns except the last one
        if args.dataset == 'cga':
            train_social_counts_df = train_df.groupby('conversation_id')['social_orientation'].apply(lambda x: x.iloc[:-1].value_counts()).unstack(level=1)
            val_social_counts_df = val_df.groupby('conversation_id')['social_orientation'].apply(lambda x: x.iloc[:-1].value_counts()).unstack(level=1)
        else:
            # other take all turns, even the last one
            train_social_counts_df = train_df.groupby('conversation_id')['social_orientation'].apply(lambda x: x.value_counts()).unstack(level=1)
            val_social_counts_df = val_df.groupby('conversation_id')['social_orientation'].apply(lambda x: x.value_counts()).unstack(level=1)

        # fill in missing values
        train_social_counts_df.fillna(0, inplace=True)
        val_social_counts_df.fillna(0, inplace=True)

        # normalize counts
        train_social_counts_df = train_social_counts_df.div(train_social_counts_df.sum(axis=1), axis=0)
        val_social_counts_df = val_social_counts_df.div(val_social_counts_df.sum(axis=1), axis=0)

        # ensure that all columns are present and if not, fill with 0
        for label in SOCIAL_ORIENTATION_LABEL2ID.keys():
            if label not in train_social_counts_df.columns:
                train_social_counts_df[label] = 0.0
            if label not in val_social_counts_df.columns:
                val_social_counts_df[label] = 0.0
        # arrange columns according to SOCIAL_ORIENTATION_LABEL2ID
        train_social_counts_df = train_social_counts_df[SOCIAL_ORIENTATION_LABEL2ID.keys()]
        val_social_counts_df = val_social_counts_df[SOCIAL_ORIENTATION_LABEL2ID.keys()]
    else:
        col_names = ['social_orientation_1', 'social_orientation_2']
        # NB: there's a subtlety with the casino corpus - there are two different outcomes per conversations, one for each speaker
        # if we're modeling mturk_agent_1, then we need to first get their utterance's social orientation tag
        # then we need to get mturk_agent_2's utterance's social orientation tag
        # crucially, we can't rely on these speakers being in any particular order
        if args.dataset == 'casino-satisfaction' or args.dataset == 'casino-opponent-likeness':
            # get social orientation tag for mturk_agent_1
            train_social_orientation_1 = train_df.groupby('conversation_id').apply(lambda x: get_speakers_first_turn(x, speaker='mturk_agent_1'))
            train_social_orientation_2 = train_df.groupby('conversation_id').apply(lambda x: get_speakers_first_turn(x, speaker='mturk_agent_2'))
            train_social_counts_df = pd.concat([train_social_orientation_1, train_social_orientation_2], axis=1)
            train_social_counts_df.columns = ['social_orientation_1', 'social_orientation_2']

            val_social_orientation_1 = val_df.groupby('conversation_id').apply(lambda x: get_speakers_first_turn(x, speaker='mturk_agent_1'))
            val_social_orientation_2 = val_df.groupby('conversation_id').apply(lambda x: get_speakers_first_turn(x, speaker='mturk_agent_2'))
            val_social_counts_df = pd.concat([val_social_orientation_1, val_social_orientation_2], axis=1)
            val_social_counts_df.columns = ['social_orientation_1', 'social_orientation_2']
        else:
            # create binary features for each social orientation tag for the first 2 utterances in the conversation
            train_first_two = train_df.groupby('conversation_id').apply(lambda x: x['social_orientation'].iloc[:2].values)
            val_first_two = val_df.groupby('conversation_id').apply(lambda x: x['social_orientation'].iloc[:2].values)

            # expand the list of social orientation tags into columns
            train_social_counts_df = pd.DataFrame(train_first_two.tolist(), index=train_first_two.index, columns=col_names)
            val_social_counts_df = pd.DataFrame(val_first_two.tolist(), index=val_first_two.index, columns=col_names)

        # create column ordering
        column_order = []
        for col in col_names:
            col_sequence = []
            for label in SOCIAL_ORIENTATION_LABEL2ID.keys():
                col_sequence.append(f'{col}_{label}')
            column_order.extend(col_sequence)

        # create dummy variables
        train_social_counts_df = pd.get_dummies(train_social_counts_df, columns=['social_orientation_1', 'social_orientation_2']).astype(int)
        val_social_counts_df = pd.get_dummies(val_social_counts_df, columns=['social_orientation_1', 'social_orientation_2']).astype(int)

        # ensure all columns are present
        for col in column_order:
            if col not in train_social_counts_df.columns:
                train_social_counts_df[col] = 0.0
            if col not in val_social_counts_df.columns:
                val_social_counts_df[col] = 0.

        # reorder columns
        train_social_counts_df = train_social_counts_df[column_order]
        val_social_counts_df = val_social_counts_df[column_order]

    # get labels from corpus
    if args.dataset == 'cga':
        train_y = train_df.groupby('conversation_id').apply(lambda x: corpus.get_conversation(x['conversation_id'].iloc[0]).meta['conversation_has_personal_attack']).astype(int)
        val_y = val_df.groupby('conversation_id').apply(lambda x: corpus.get_conversation(x['conversation_id'].iloc[0]).meta['conversation_has_personal_attack']).astype(int)
    elif args.dataset == 'casino-satisfaction':
        # binarize satisfaction labels
        label_map = {
            'Extremely satisfied': True,
            'Slightly satisfied': True,
            'Undecided': True,
            'Slightly dissatisfied': False,
            'Extremely dissatisfied': False,
        }
        # NB: we're hardcoding mturk_agent_1 here, TODO: generalize this
        train_y = train_df.groupby('conversation_id').apply(lambda x: corpus.get_conversation(x['conversation_id'].iloc[0]).meta['participant_info']['mturk_agent_1']['outcomes']['satisfaction']).map(label_map)
        val_y = val_df.groupby('conversation_id').apply(lambda x: corpus.get_conversation(x['conversation_id'].iloc[0]).meta['participant_info']['mturk_agent_1']['outcomes']['satisfaction']).map(label_map)
    elif args.dataset == 'casino-opponent-likeness':
        # binarize opponent likeness labels
        label_map = {
            'Extremely like': True,
            'Slightly like': True,
            'Undecided': True,
            'Slightly dislike': False,
            'Extremely dislike': False
        }
        # NB: we're hardcoding mturk_agent_1 here, TODO: generalize this
        train_y = train_df.groupby('conversation_id').apply(lambda x: corpus.get_conversation(x['conversation_id'].iloc[0]).meta['participant_info']['mturk_agent_1']['outcomes']['opponent_likeness']).map(label_map)
        val_y = val_df.groupby('conversation_id').apply(lambda x: corpus.get_conversation(x['conversation_id'].iloc[0]).meta['participant_info']['mturk_agent_1']['outcomes']['opponent_likeness']).map(label_map)
    elif args.dataset == 'cga-cmv':
        train_y = train_df.groupby('conversation_id').apply(lambda x: corpus.get_conversation(x['conversation_id'].iloc[0]).meta['has_removed_comment']).astype(int)
        val_y = val_df.groupby('conversation_id').apply(lambda x: corpus.get_conversation(x['conversation_id'].iloc[0]).meta['has_removed_comment']).astype(int)
    clf = LogisticRegression(random_state=args.seed, max_iter=1000).fit(train_social_counts_df, train_y)
    val_preds = clf.predict(val_social_counts_df)
    val_acc = accuracy_score(val_y, val_preds)
    logging.info(f'Simple classifier accuracy with {args.subset_pct*100:.2f}% of the data: {val_acc:.4f}')
    logging.info(f'\n{classification_report(val_y, val_preds, zero_division=0)}')
    val_clf_report = classification_report(val_y, val_preds, zero_division=0, output_dict=True)

    # examine coefficients
    coefficients_df = pd.DataFrame({'Feature': train_social_counts_df.columns, 'Coefficient': clf.coef_[0]})
    coefficients_df['abs_coefficient'] = coefficients_df['Coefficient'].abs()
    coefficients_df.sort_values(by='abs_coefficient', ascending=False, inplace=True)
    coefficients_df.drop(columns=['abs_coefficient'], inplace=True)
    logging.info(f'Most important features:\n{coefficients_df.head(10)}')
    # save coefficients to disk
    count_all = 'count_all' if count_vector else 'count_first_two'
    coefficients_df.to_csv(os.path.join(args.analysis_dir, f'{args.dataset}_logistic_classifier_coefficients_{count_all}_{social_source}_{args.subset_pct}.csv'), index=False)
    return clf, val_acc, val_clf_report

def cga_explainability(train_df, val_df, corpus, args, social_source='gpt4'):
    # assign cga labels to conversations
    train_df['cga_label'] = train_df['conversation_id'].apply(lambda x: corpus.get_conversation(x).meta['conversation_has_personal_attack'])
    # group by cga_label and count social_orientation labels
    cga_social_splits = train_df.groupby('cga_label')['social_orientation'].value_counts()

    # get likelihood ratio of social orientation labels for each cga label: P(social_orientation | cga_label==True) / P(social_orientation | cga_label==False)
    cga_social_splits = cga_social_splits.unstack(level=0)
    # probability of each social orientation label within each cga label
    cga_social_splits = cga_social_splits / cga_social_splits.sum()
    cga_social_splits['likelihood_ratio'] = cga_social_splits[True] / cga_social_splits[False]
    # sort by likelihood ratio
    cga_social_splits.sort_values(by='likelihood_ratio', ascending=False, inplace=True)
    # plot likelihood ratio
    cga_social_splits['likelihood_ratio'].plot(kind='bar')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Social Orientation')
    plt.ylabel('Likelihood Ratio')
    plt.title('Likelihood Ratio of Social Orientation Labels for CGA Labels')
    plt.savefig(os.path.join('./logs/analysis', f'cga_social_likelihood_ratio_{social_source}.png'), dpi=300, bbox_inches='tight')
    plt.clf()

    # examine most commonly occuring first two social orientation labels for each cga label
    # groupby conversation_id grab first two utterances for each conversation, and return a tuple
    # of the social orientation labels
    social_orientation_pairs_df = train_df.groupby('conversation_id')['social_orientation'].apply(lambda x: tuple(sorted(list(x[:2])))).to_frame()
    social_orientation_pairs_df.reset_index(inplace=True)
    social_orientation_pairs_df['cga_label'] = social_orientation_pairs_df['conversation_id'].apply(lambda x: corpus.get_conversation(x).meta['conversation_has_personal_attack'])
    # group by cga_label and count social_orientation pairs
    cga_social_pairs = social_orientation_pairs_df.groupby('cga_label')['social_orientation'].value_counts().unstack(level=0).sort_values(by=True, ascending=False)
    # plot
    cga_social_pairs.iloc[:10].plot(kind='bar')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('First 2 Social Orientation Tags')
    plt.ylabel('Count')
    plt.title('Most Common First 2 Social Orientation Tags for CGA Labels')
    plt.savefig(os.path.join('./logs/analysis', f'cga_social_pairs_{social_source}.png'), dpi=300, bbox_inches='tight')
    plt.clf()

    # train a simple classifier to predict cga_label from social_orientation
    logging.info('Training simple classifier to predict CGA label from social orientation tags (all turns)')
    simple_classifier(train_df, val_df, corpus, args, count_vector=True, social_source=social_source)
    logging.info('Training simple classifier to predict CGA label from social orientation tags (first two turns)')
    simple_classifier(train_df, val_df, corpus, args, count_vector=False, social_source=social_source)

def explainability(train_df, val_df, corpus, args, social_source='gpt4'):
    if args.dataset == 'cga':
        target_name = 'cga_label'
    elif args.dataset == 'casino-satisfaction':
        target_name = 'satisfaction_binary'
    elif args.dataset == 'casino-opponent-likeness':
        target_name = 'opponent_likeness_binary'
    elif args.dataset == 'cga-cmv':
        target_name = 'meta.has_removed_comment'
    # assign outcome labels to conversations
    if args.dataset == 'cga':
        train_df[target_name] = train_df['conversation_id'].apply(lambda x: corpus.get_conversation(x).meta['conversation_has_personal_attack'])
    elif args.dataset == 'casino-satisfaction':
        # binarize satisfaction labels
        label_map = {
            'Extremely satisfied': True,
            'Slightly satisfied': True,
            'Undecided': True,
            'Slightly dissatisfied': False,
            'Extremely dissatisfied': False,
        }
        train_df[target_name] = train_df['satisfaction'].map(label_map)
    elif args.dataset == 'casino-opponent-likeness':
        # binarize opponent likeness labels
        label_map = {
            'Extremely like': True,
            'Slightly like': True,
            'Undecided': True,
            'Slightly dislike': False,
            'Extremely dislike': False
        }
        train_df[target_name] = train_df['opponent_likeness'].map(label_map)
    elif args.dataset == 'cga-cmv':
        train_df[target_name] = train_df['conversation_id'].apply(lambda x: corpus.get_conversation(x).meta['has_removed_comment'])

    # NB: there's a subtlety with the casino corpus - there are two different outcomes per conversations, one for each speaker
    # TODO: currently leaving this alone, but should really filter down to 1 of the speakers
    # group by outcome label and count social_orientation labels
    social_splits = train_df.groupby(target_name)['social_orientation'].value_counts()

    # get likelihood ratio of social orientation labels for each outcome label: P(social_orientation | outcome_label==True) / P(social_orientation | outcome_label==False)
    social_splits = social_splits.unstack(level=0)
    # probability of each social orientation label within each outcome label
    social_splits = social_splits / social_splits.sum()
    social_splits['likelihood_ratio'] = social_splits[True] / social_splits[False]
    social_splits.sort_values(by='likelihood_ratio', ascending=False, inplace=True)
    # plot likelihood ratio
    social_splits['likelihood_ratio'].plot(kind='bar')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Social Orientation')
    plt.ylabel('Likelihood Ratio')
    plt.title(f'Likelihood Ratio of Social Orientation Labels for {args.dataset} {target_name}')
    plt.savefig(os.path.join('./logs/analysis', f'{args.dataset}_{target_name}_social_likelihood_ratio_{social_source}.png'), dpi=300, bbox_inches='tight')
    plt.clf()

    # examine most commonly occuring first two social orientation labels for each outcome label
    # groupby conversation_id grab first two utterances for each conversation, and return a tuple
    # of the social orientation labels
    social_orientation_pairs_df = train_df.groupby('conversation_id')['social_orientation'].apply(lambda x: tuple(sorted(list(x[:2])))).to_frame()
    social_orientation_pairs_df.reset_index(inplace=True)
    # need to pull in outcome labels
    if args.dataset == 'casino-satisfaction':
        # NB: we're hardcoding mturk_agent_1 here, TODO: generalize this
        social_orientation_pairs_df['satisfaction'] = social_orientation_pairs_df['conversation_id'].apply(lambda x: corpus.get_conversation(x).meta['participant_info']['mturk_agent_1']['outcomes']['satisfaction'])
        social_orientation_pairs_df[target_name] = social_orientation_pairs_df['satisfaction'].map(label_map)
    elif args.dataset == 'casino-opponent-likeness':
        # NB: we're hardcoding mturk_agent_1 here, TODO: generalize this
        social_orientation_pairs_df['opponent_likeness'] = social_orientation_pairs_df['conversation_id'].apply(lambda x: corpus.get_conversation(x).meta['participant_info']['mturk_agent_1']['outcomes']['opponent_likeness'])
        social_orientation_pairs_df[target_name] = social_orientation_pairs_df['opponent_likeness'].map(label_map)
    elif args.dataset == 'cga':
        social_orientation_pairs_df[target_name] = social_orientation_pairs_df['conversation_id'].apply(lambda x: corpus.get_conversation(x).meta['conversation_has_personal_attack'])
    elif args.dataset == 'cga-cmv':
        social_orientation_pairs_df[target_name] = social_orientation_pairs_df['conversation_id'].apply(lambda x: corpus.get_conversation(x).meta['has_removed_comment'])

    # group by target_name and count social_orientation pairs
    social_pairs = social_orientation_pairs_df.groupby(target_name)['social_orientation'].value_counts().unstack(level=0).sort_values(by=True, ascending=False)
    # plot
    social_pairs.iloc[:10].plot(kind='bar')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('First 2 Social Orientation Tags')
    plt.ylabel('Count')
    plt.title(f'Most Common First 2 Social Orientation Tags for {args.dataset} {target_name}')
    plt.savefig(os.path.join('./logs/analysis', f'{args.dataset}_{target_name}_social_pairs_{social_source}.png'), dpi=300, bbox_inches='tight')
    plt.clf()

    # train a simple classifier to predict target_name from social_orientation
    logging.info(f'Training simple classifier to predict {args.dataset} {target_name} from social orientation tags (all turns)')
    clf, count_val_acc, count_val_clf_report = simple_classifier(train_df, val_df, corpus, args, count_vector=True, social_source=social_source)
    logging.info(f'Training simple classifier to predict {args.dataset} {target_name} from social orientation tags (first two turns)')
    clf, first_2_val_acc, first_2_val_clf_report = simple_classifier(train_df, val_df, corpus, args, count_vector=False, social_source=social_source)
    return count_val_acc, first_2_val_acc

def load_experiments(args):
    """Load experiment results from disk."""
    df = pd.read_csv(args.experiment_filepath[0])
    # TODO: remove this hardcoded value, and add it to train.py
    df['method'] = 'distilbert'
    # filter out rows where window_size == '2'
    df = df[df['window_size'] == 'all']
    if 'test_acc' not in df.columns:
        df['test_acc'] = np.nan
        df['test_loss'] = np.nan

    if len(args.experiment_filepath) == 2:
        # logistic model
        log_clf_df = pd.read_csv(args.experiment_filepath[1])
        # filter out rows where window_size == '2'
        log_clf_df = log_clf_df[log_clf_df['window_size'] == 'all']
        col_order = log_clf_df.columns
        df = df[col_order]
        df = pd.concat([df, log_clf_df], axis=0)

    df['social_orientation_prediction'] = df['social_orientation_filepaths'].apply(lambda x: social_orientation_source(x))
    # for logistic regression with distilbert features, we don't actually use social orientation predictions
    # so we can just set this to np.nan
    df.loc[df['method'] == 'logistic_distilbert', 'social_orientation_prediction'] = 'None'
    # same deal with logistic_tfidf
    df.loc[df['method'] == 'logistic_tfidf', 'social_orientation_prediction'] = 'None'

    df.reset_index(inplace=True, drop=True)
    return df

def main(args):
    # determine which analysis to perform
    if args.analysis_mode == 't-test' and args.experiment == 'window-size':
        df = pd.read_csv(args.experiment_filepath)
        # aggregate results across seeds and perform t-test
        agg_df = df.groupby(['method', 'window_size' 'social_orientation']).agg({'val_acc': ['mean', 'std'], 'val_loss': ['mean', 'std']})

        ttest_results_df = df.groupby(['window_size']).apply(lambda x: pd.Series({'p_value': t_test(x)}))
        row_order = ['2', '6', '10', 'all']
        all_present = True
        for row in row_order:
            if row not in ttest_results_df.index:
                all_present = False
                row_order = ttest_results_df.index
                break
        results_df = agg_df.unstack(level=1)[['val_acc']].loc[row_order]
        # add in dummy column levels to match results_df
        ttest_results_df.columns = pd.MultiIndex.from_tuples([('val_acc', None, 'p_value')])
        results_df = pd.concat([results_df, ttest_results_df], axis=1)
        # save results
        head, tail = os.path.split(args.experiment_filepath)
        filename = tail.split('.')[0]
        results_df.to_csv(os.path.join(args.analysis_dir, f'{filename}_ttest_results.csv'))
        exit(0)
    elif args.analysis_mode == 't-test' and args.experiment == 'subset':
        subset = 'test' # 'val'
        # load data (to get conversation counts)
        _, corpus = load_data(args.data_dir, include_speakers=args.include_speakers, social_orientation_filepaths=args.social_orientation_filepaths, include_social_orientation=args.include_social_orientation)
        convo_df = corpus.get_conversations_dataframe()
        num_train_conversations = len(convo_df[convo_df['meta.split'] == 'train'])
        df = load_experiments(args)
        # aggregate results across seeds and perform t-test
        agg_df = df.groupby(['method', 'subset_pct', 'social_orientation_prediction']).agg({f'{subset}_acc': ['mean', 'std'], f'{subset}_loss': ['mean', 'std']})
        
        # compare distilbert text only to logistic regression with social counts
        method_a = 'distilbert'
        method_b = 'logistic_social_counts'
        method_a_social_features = 'None'
        method_b_social_features = 'winsize_2_model_distilbert-base-uncased'
        ttest_results_df = df.groupby(['subset_pct']).apply(t_test_a_vs_b, method_a=method_a, method_b=method_b, method_a_social_features=method_a_social_features, method_b_social_features=method_b_social_features, subset=subset).to_frame().rename(columns={0: 'p_value'})
        results_df = agg_df.unstack(level=[0, 2])
        # select relevant columns
        results_df = results_df.loc[:, [(f'{subset}_acc', 'mean', method_a, method_a_social_features), (f'{subset}_acc', 'std', method_a, method_a_social_features), (f'{subset}_acc', 'mean', method_b, method_b_social_features), (f'{subset}_acc', 'std', method_b, method_b_social_features)]]
        # add in dummy column levels to match results_df
        ttest_results_df.columns = pd.MultiIndex.from_tuples([(f'{subset}_acc', None, None, 'p_value')])
        results_df = pd.concat([results_df, ttest_results_df], axis=1)
        # save results
        head, tail = os.path.split(args.experiment_filepath[0])
        filename = tail.split('.')[0]
        results_df.reset_index(inplace=True)
        # clean up column names
        columns = ['Subset %', 'DistilBERT', 'DistilBERT Std. Acc.', 'Logistic (Social Counts)', 'Logistic (Social Counts) Std. Acc.', 'p-value']
        results_df.columns = columns
        # drop std. columns
        results_df.drop(columns=['DistilBERT Std. Acc.', 'Logistic (Social Counts) Std. Acc.'], inplace=True)
        # add in conversation counts
        results_df['Convos.'] = results_df['Subset %'].apply(lambda x: int(num_train_conversations * x))
        col_order = ['Subset %', 'Convos.', 'DistilBERT', 'Logistic (Social Counts)', 'p-value']
        results_df = results_df[col_order]
        # multiply Subset, DistilBERT, and Logistic (Social Counts) columns by 100
        results_df['Subset %'] = results_df['Subset %'].apply(lambda x: x * 100)
        results_df['DistilBERT'] = results_df['DistilBERT'].apply(lambda x: x * 100)
        results_df['Logistic (Social Counts)'] = results_df['Logistic (Social Counts)'].apply(lambda x: x * 100)
        results_df.to_csv(os.path.join(args.analysis_dir, f'{filename}_t_test_log_vs_text_results_{subset}.csv'))
        results_df.to_latex(os.path.join(args.analysis_dir, f'{filename}_t_test_log_vs_text_results_{subset}.tex'), index=False, float_format="%.3f", formatters={'Subset %': '{:,.0f}'.format, 'Convos.': '{:,}'.format, 'DistilBERT': '{:,.2f}'.format, 'Logistic (Social Counts)': '{:,.2f}'.format})
        print(results_df)

        # compare distilbert text only to distilbert with text + social preds
        method_a = 'distilbert'
        method_b = 'distilbert'
        method_a_social_features = 'None'
        method_b_social_features = 'winsize_2_model_distilbert-base-uncased'
        ttest_results_df = df.groupby(['subset_pct']).apply(t_test_a_vs_b, method_a=method_a, method_b=method_b, method_a_social_features=method_a_social_features, method_b_social_features=method_b_social_features, subset=subset).to_frame().rename(columns={0: 'p_value'})
        results_df = agg_df.unstack(level=[0, 2])
        # select relevant columns
        results_df = results_df.loc[:, [(f'{subset}_acc', 'mean', method_a, method_a_social_features), (f'{subset}_acc', 'std', method_a, method_a_social_features), (f'{subset}_acc', 'mean', method_b, method_b_social_features), (f'{subset}_acc', 'std', method_b, method_b_social_features)]]
        # add in dummy column levels to match results_df
        ttest_results_df.columns = pd.MultiIndex.from_tuples([(f'{subset}_acc', None, None, 'p_value')])
        results_df = pd.concat([results_df, ttest_results_df], axis=1)
        # save results
        head, tail = os.path.split(args.experiment_filepath[0])
        filename = tail.split('.')[0]
        results_df.reset_index(inplace=True)
        # clean up column names
        columns = ['Subset %', 'DistilBERT', 'DistilBERT Std. Acc.', 'DistilBERT + Social', 'DistilBERT + Social Std. Acc.', 'p-value']
        results_df.columns = columns
        # drop std. columns
        results_df.drop(columns=['DistilBERT Std. Acc.', 'DistilBERT + Social Std. Acc.'], inplace=True)
        # add in conversation counts
        results_df['Convos.'] = results_df['Subset %'].apply(lambda x: int(num_train_conversations * x))
        col_order = ['Subset %', 'Convos.', 'DistilBERT', 'DistilBERT + Social', 'p-value']
        results_df = results_df[col_order]
        # multiply Subset, DistilBERT, and Logistic (Social Counts) columns by 100
        results_df['Subset %'] = results_df['Subset %'].apply(lambda x: x * 100)
        results_df['DistilBERT'] = results_df['DistilBERT'].apply(lambda x: x * 100)
        results_df['DistilBERT + Social'] = results_df['DistilBERT + Social'].apply(lambda x: x * 100)
        results_df.to_csv(os.path.join(args.analysis_dir, f'{filename}_t_test_text_vs_text_plus_social_results_{subset}.csv'))
        results_df.to_latex(os.path.join(args.analysis_dir, f'{filename}_t_test_text_vs_text_plus_social_results_{subset}.tex'), index=False, float_format="%.3f", formatters={'Subset %': '{:,.0f}'.format, 'Convos.': '{:,}'.format, 'DistilBERT': '{:,.2f}'.format, 'DistilBERT + Social': '{:,.2f}'.format})
        print(results_df)

        # compare logistic regression with social counts to: [distilbert (text only), distilbert (text + social preds)]
        method_as = ['distilbert']
        method_b = 'logistic_social_counts'
        method_a_social_feature_set = ['None', 'winsize_2_model_distilbert-base-uncased']
        if args.dataset == 'cga':
            method_a_social_feature_set.append('GPT-4')
        method_b_social_features = 'winsize_2_model_distilbert-base-uncased'
        results_df = agg_df.unstack(level=[0, 2])
        final_df = pd.DataFrame()
        for method_a in method_as:
            for method_a_social_features in method_a_social_feature_set:
                ttest_results_df = df.groupby(['subset_pct']).apply(t_test_a_vs_b, method_a=method_a, method_b=method_b, method_a_social_features=method_a_social_features, method_b_social_features=method_b_social_features, subset=subset).to_frame().rename(columns={0: 'p_value'})
                # select relevant columns
                temp_df = results_df.loc[:, [(f'{subset}_acc', 'mean', method_a, method_a_social_features), (f'{subset}_acc', 'mean', method_b, method_b_social_features)]]
                # add in dummy column levels to match results_df
                method_name = method_a + '_' + method_a_social_features
                ttest_results_df.columns = pd.MultiIndex.from_tuples([(f'{subset}_acc', None, None, f'p_value_{method_name}')])
                temp_df = pd.concat([temp_df, ttest_results_df], axis=1)
                final_df = pd.concat([final_df, temp_df], axis=1)
        
        # save results
        head, tail = os.path.split(args.experiment_filepath[0])
        filename = tail.split('.')[0]
        final_df.reset_index(inplace=True)
        # clean up column names
        columns = ['Subset %', 'DistilBERT', 'Logistic (Social Counts)', 'p-value (DistilBERT)', 'DistilBERT (Social)', 'Logistic (Social Counts)', 'p-value (DistilBERT (Social))']
        if args.dataset == 'cga':
            columns.extend(['DistilBERT (GPT-4)', 'Logistic (Social Counts)', 'p-value (DistilBERT (GPT-4))'])
        final_df.columns = columns
        # drop column numbers 2, and 5
        cols_to_drop = [2, 5] if args.dataset == 'cga' else [2]
        final_df = final_df.iloc[:, [i for i in range(final_df.shape[1]) if i not in cols_to_drop]]

        # add in conversation counts
        final_df['Conversations'] = final_df['Subset %'].apply(lambda x: int(num_train_conversations * x))
        col_order = ['Subset %', 'Conversations', 'DistilBERT', 'DistilBERT (Social)', 'Logistic (Social Counts)', 'p-value (DistilBERT)', 'p-value (DistilBERT (Social))']
        if args.dataset == 'cga':
            col_order = ['Subset %', 'Conversations', 'DistilBERT', 'DistilBERT (Social)', 'DistilBERT (GPT-4)', 'Logistic (Social Counts)', 'p-value (DistilBERT)', 'p-value (DistilBERT (Social))', 'p-value (DistilBERT (GPT-4))']
        final_df = final_df[col_order]

        # multiply by 100
        mult_cols = ['Subset %', 'DistilBERT', 'DistilBERT (Social)', 'Logistic (Social Counts)']
        if args.dataset == 'cga':
            mult_cols = ['Subset %', 'DistilBERT', 'DistilBERT (Social)', 'DistilBERT (GPT-4)', 'Logistic (Social Counts)']
        final_df[mult_cols] = final_df[mult_cols].apply(lambda x: x * 100)
        # multiply Subset, DistilBERT, and Logistic (Social Counts) columns by 100
        # results_df['Subset %'] = results_df['Subset %'].apply(lambda x: x * 100)
        # results_df['DistilBERT'] = results_df['DistilBERT'].apply(lambda x: x * 100)
        # results_df['Logistic (Social Counts)'] = results_df['Logistic (Social Counts)'].apply(lambda x: x * 100)
        final_df.to_csv(os.path.join(args.analysis_dir, f'{filename}_t_test_log_vs_neural_results_{subset}.csv'))
        formatters = {'Subset %': '{:,.0f}'.format, 'Conversations': '{:,}'.format, 'DistilBERT': '{:,.2f}'.format, 'DistilBERT (Social)': '{:,.2f}'.format, 'Logistic (Social Counts)': '{:,.2f}'.format}
        if args.dataset == 'cga':
            formatters['DistilBERT (GPT-4)'] = '{:,.2f}'.format
        final_df.to_latex(os.path.join(args.analysis_dir, f'{filename}_t_test_log_vs_neural_results_{subset}.tex'), index=False, float_format="%.3f", formatters=formatters)
        print(final_df)


        # compare logistic regression with social counts to: [distilbert (text only), distilbert (text + social preds)]
        method_as = ['logistic_social_counts', 'distilbert']
        method_b = 'distilbert'
        method_a_social_feature_set = ['winsize_2_model_distilbert-base-uncased', 'winsize_2_model_distilbert-base-uncased']
        if args.dataset == 'cga':
            method_as.append('distilbert')
            method_a_social_feature_set.append('GPT-4')
        method_b_social_features = 'None'
        results_df = agg_df.unstack(level=[0, 2])
        final_df = pd.DataFrame()
        for (method_a, method_a_social_features) in zip(method_as, method_a_social_feature_set):
            ttest_results_df = df.groupby(['subset_pct']).apply(t_test_a_vs_b, method_a=method_a, method_b=method_b, method_a_social_features=method_a_social_features, method_b_social_features=method_b_social_features, subset=subset, alternative='greater').to_frame().rename(columns={0: 'p_value'})
            # select relevant columns
            temp_df = results_df.loc[:, [(f'{subset}_acc', 'mean', method_a, method_a_social_features), (f'{subset}_acc', 'mean', method_b, method_b_social_features)]]
            # add in dummy column levels to match results_df
            method_name = method_a + '_' + method_a_social_features
            ttest_results_df.columns = pd.MultiIndex.from_tuples([(f'{subset}_acc', None, None, f'p_value_{method_name}')])
            temp_df = pd.concat([temp_df, ttest_results_df], axis=1)
            final_df = pd.concat([final_df, temp_df], axis=1)
        
        # breakpoint()
        
        # save results
        head, tail = os.path.split(args.experiment_filepath[0])
        filename = tail.split('.')[0]
        final_df.reset_index(inplace=True)
        # clean up column names
        columns = ['Subset %', 'Logistic (Social Counts)', 'DistilBERT', 'p-value (Logistic (Social Counts))', 'DistilBERT (Social)', 'DistilBERT', 'p-value (DistilBERT (Social))']
        if args.dataset == 'cga':
            columns.extend(['DistilBERT (GPT-4)', 'DistilBERT', 'p-value (DistilBERT (GPT-4))'])
        final_df.columns = columns
        # drop column numbers 2, and 5
        cols_to_drop = [2, 5] if args.dataset == 'cga' else [2]
        final_df = final_df.iloc[:, [i for i in range(final_df.shape[1]) if i not in cols_to_drop]]

        # add in conversation counts
        final_df['Conversations'] = final_df['Subset %'].apply(lambda x: int(num_train_conversations * x))
        col_order = ['Subset %', 'Conversations', 'Logistic (Social Counts)', 'DistilBERT', 'DistilBERT (Social)', 'p-value (Logistic (Social Counts))', 'p-value (DistilBERT (Social))']
        if args.dataset == 'cga':
            col_order = ['Subset %', 'Conversations', 'Logistic (Social Counts)', 'DistilBERT', 'DistilBERT (Social)', 'DistilBERT (GPT-4)', 'p-value (Logistic (Social Counts))', 'p-value (DistilBERT (Social))', 'p-value (DistilBERT (GPT-4))']
        final_df = final_df[col_order]

        # multiply by 100
        mult_cols = ['Subset %', 'DistilBERT', 'DistilBERT (Social)', 'Logistic (Social Counts)']
        if args.dataset == 'cga':
            mult_cols = ['Subset %', 'DistilBERT', 'DistilBERT (Social)', 'DistilBERT (GPT-4)', 'Logistic (Social Counts)']
        final_df[mult_cols] = final_df[mult_cols].apply(lambda x: x * 100)
        # multiply Subset, DistilBERT, and Logistic (Social Counts) columns by 100
        # results_df['Subset %'] = results_df['Subset %'].apply(lambda x: x * 100)
        # results_df['DistilBERT'] = results_df['DistilBERT'].apply(lambda x: x * 100)
        # results_df['Logistic (Social Counts)'] = results_df['Logistic (Social Counts)'].apply(lambda x: x * 100)
        final_df.to_csv(os.path.join(args.analysis_dir, f'{filename}_t_test_log_vs_neural_results_{subset}.csv'))
        formatters = {'Subset %': '{:,.0f}'.format, 'Conversations': '{:,}'.format, 'DistilBERT': '{:,.2f}'.format, 'DistilBERT (Social)': '{:,.2f}'.format, 'Logistic (Social Counts)': '{:,.2f}'.format}
        if args.dataset == 'cga':
            formatters['DistilBERT (GPT-4)'] = '{:,.2f}'.format
        final_df.to_latex(os.path.join(args.analysis_dir, f'{filename}_t_test_log_vs_neural_results_{subset}.tex'), index=False, float_format="%.3f", formatters=formatters)
        print(final_df)

        exit(0)

    elif args.analysis_mode == 'merge-data':
        df, corpus = load_data(args.data_dir, include_speakers=args.include_speakers, social_orientation_filepaths=args.social_orientation_filepaths, include_social_orientation=args.include_social_orientation)
        train_df, val_df, test_df, _ = get_data_splits(df, args.data_dir)

        # load predictions
        predictions = [os.path.expanduser(p) for p in args.prediction_filepaths]
        dfs = []
        for p in predictions:
            temp_df = pd.read_csv(p)
            dfs.append(temp_df)
        pred_df = pd.concat(dfs)
        # rename social_orientation to social_orientation_prediction
        pred_df.rename(columns={'social_orientation': 'social_orientation_prediction'}, inplace=True)

        # merge predictions with ground truth
        cols = ['conversation_id', 'utterance_id', 'speaker', 'social_orientation_prediction']
        merge_on = ['conversation_id', 'utterance_id', 'speaker']

        # concat train and val for easier merging
        train_val_df = pd.concat([train_df, val_df])
        train_val_df['cga_label'] = train_val_df['conversation_id'].apply(lambda x: corpus.get_conversation(x).meta['conversation_has_personal_attack'])
        train_val_df = train_val_df.merge(pred_df[cols], on=merge_on, how='left')
        logging.warning(f'{len(train_val_df[train_val_df["social_orientation"].isna()])}/{len(train_val_df)} rows with no social orientation label.')
        # save to analysis dir
        # make the save filename a function of the prediction filepaths
        head, tail = os.path.split(args.prediction_filepaths[0])
        filename = tail.split('.')[0]
        filename = 'train_val_' + '_'.join(filename.split('_')[1:])
        train_val_df.to_csv(os.path.join(args.analysis_dir, f'{filename}_social_orientation.csv'), index=False)
        exit(0)
    elif args.analysis_mode == 'compare-cga':
        # compare CGA predictions for models trained using predicted social tags
        # to models trained using GPT-4 social tags
        # TODO: fix this hack: currently using args.prediction_filepaths in a hardcoded order
        # assume cga with predicted social tags is first, then cga with gpt4 social tags
        cga_pred_df = pd.read_csv(args.prediction_filepaths[0])
        cga_gpt4_df = pd.read_csv(args.prediction_filepaths[1])

        # broadcast comment_has_personal_attack to all utterances in the conversation
        cga_pred_df['cga_awry'] = cga_pred_df.groupby('conversation_id')['meta.comment_has_personal_attack'].transform('max').astype(int)
        cga_gpt4_df['cga_awry'] = cga_gpt4_df.groupby('conversation_id')['meta.comment_has_personal_attack'].transform('max').astype(int)

        # get encoded label
        cga_pred_df['prediction'] = cga_pred_df['prediction'].apply(lambda x: CGA_LABEL2ID[x])
        cga_gpt4_df['prediction'] = cga_gpt4_df['prediction'].apply(lambda x: CGA_LABEL2ID[x])

        # quick evaluation of predictions
        cga_pred_df['meta.comment_has_personal_attack'] = cga_pred_df['meta.comment_has_personal_attack'].astype(int)
        cga_gpt4_df['meta.comment_has_personal_attack'] = cga_gpt4_df['meta.comment_has_personal_attack'].astype(int)

        pred_results_df = cga_pred_df.groupby('conversation_id')[['meta.comment_has_personal_attack', 'prediction']].max()
        pred_results_df['correct'] = pred_results_df['meta.comment_has_personal_attack'] == pred_results_df['prediction']
        logging.info(f"CGA accuracy using predicted social orientation tags: {pred_results_df['correct'].mean():.4f}")

        gpt4_results_df = cga_gpt4_df.groupby('conversation_id')[['meta.comment_has_personal_attack', 'prediction']].max()
        gpt4_results_df['correct'] = gpt4_results_df['meta.comment_has_personal_attack'] == gpt4_results_df['prediction']
        logging.info(f"CGA accuracy using GPT-4 social orientation tags: {gpt4_results_df['correct'].mean():.4f}")

        # rename columns to keep track of which model is which
        rename_pred_cols = {'prediction': 'prediction_pred', 'social_orientation': 'social_orientation_pred'}
        rename_gpt4_cols = {'prediction': 'prediction_gpt4', 'social_orientation': 'social_orientation_gpt4'}
        cga_pred_df.rename(columns=rename_pred_cols, inplace=True)
        cga_gpt4_df.rename(columns=rename_gpt4_cols, inplace=True)

        # merge dfs
        cols = rename_gpt4_cols.values()
        merged_df = pd.concat([cga_pred_df, cga_gpt4_df[cols]], axis=1)
        merged_df['agree'] = merged_df['prediction_pred'] == merged_df['prediction_gpt4']
        agree_conversations = merged_df.groupby('conversation_id')['agree'].max()
        agreement_rate = agree_conversations.mean()
        logging.info(f"CGA agreement rate between predicted social orientation model and GPT-4 social orientation tags: {agree_conversations.value_counts()[True]}/{merged_df['conversation_id'].nunique()}={agreement_rate:.4f}")

        # identify conversations where the models disagree
        disagree_conversations = agree_conversations[agree_conversations == False].index

        # further breakdown by predicted model thinks it's an attack and gpt4 model thinks it's not an attack
        pred_attack_df = merged_df[merged_df['conversation_id'].isin(disagree_conversations) & (merged_df['prediction_pred'] == 1) & (merged_df['prediction_gpt4'] == 0)]

        # create a confusion matrix for the disagreeing conversations
        # rows are gpt4 labels, columns are predicted labels
        plot_confusion_matrix(pred_attack_df['social_orientation_gpt4'], pred_attack_df['social_orientation_pred'], list(SOCIAL_ORIENTATION_LABEL2ID.keys()), model_name='Predicted Attack (columns) vs. GPT-4 No Attack (rows) Social Orientation', output_dir='./logs/analysis')

        # and vice versa
        gpt4_attack_df = merged_df[merged_df['conversation_id'].isin(disagree_conversations) & (merged_df['prediction_pred'] == 0) & (merged_df['prediction_gpt4'] == 1)]

        # create a confusion matrix for the disagreeing conversations
        # rows are gpt4 labels, columns are predicted labels
        plot_confusion_matrix(gpt4_attack_df['social_orientation_gpt4'], gpt4_attack_df['social_orientation_pred'], list(SOCIAL_ORIENTATION_LABEL2ID.keys()), model_name='Predicted No Attack (columns) vs. GPT-4 Attack (rows) Social Orientation', output_dir='./logs/analysis')

        # counts of both scenarios
        logging.info(f"Number of conversations where predicted model thinks it's an attack and GPT-4 model thinks it's not an attack: {pred_attack_df['conversation_id'].nunique()}")
        logging.info(f"Number of conversations where predicted model thinks it's not an attack and GPT-4 model thinks it's an attack: {gpt4_attack_df['conversation_id'].nunique()}")
        assert (pred_attack_df['conversation_id'].nunique() + gpt4_attack_df['conversation_id'].nunique()) == len(disagree_conversations)
        # among the disagreeing conversations, counts when gpt4 model thinks it's an attack and predicted model thinks it's not an attack, and vice versa
        # could create a plot here to make things easier to visualize
        merged_df[merged_df['conversation_id'].isin(disagree_conversations)].groupby(['conversation_id'])['prediction_gpt4'].max().value_counts()

        # load annotated data to merge in manual annotations, where available
        todd_annotated_df = pd.read_csv(os.path.join('./logs/analysis', 'Circumplex Annotation - Todd.csv'))
        todd_annotated_df.rename(columns={'social orientation': 'social_orientation_todd'}, inplace=True)
        pred_attack_df = pred_attack_df.merge(todd_annotated_df[['id', 'social_orientation_todd']], on='id', how='left')

        # in pred_attack_df, examine conversations that contain the following patterns:
        # pred model predicts ['Cold', 'Arrogant-Calculating'] while the gpt4
        # model predicts ['Unassuming-Ingenuous', 'Unassured-Submissive', 'Warm-Agreeable', 'Gregarious-Extraverted']
        pred_attack_df['gpt_4_correct'] = pred_attack_df['prediction_gpt4'] == pred_attack_df['cga_awry']
        logging.info(f"Disagreeing conversations where the GPT-4 supported model correctly predicts no attack:\n{pred_attack_df.groupby('conversation_id')['gpt_4_correct'].max().value_counts()}")

        conversation_ids = pred_attack_df[(pred_attack_df['social_orientation_pred'].isin(['Cold', 'Arrogant-Calculating'])) & (pred_attack_df['social_orientation_gpt4'].isin(['Unassuming-Ingenuous', 'Unassured-Submissive', 'Warm-Agreeable', 'Gregarious-Extraverted']))]['conversation_id'].unique()
        # get the conversations
        pred_attack_issue_conversations_df = pred_attack_df[pred_attack_df['conversation_id'].isin(conversation_ids)]
        # organize the columns
        cols = ['id', 'conversation_id', 'speaker', 'original_text', 'social_orientation_pred', 'social_orientation_gpt4', 'social_orientation_todd', 'prediction_pred', 'prediction_gpt4', 'cga_awry', 'text']
        pred_attack_issue_conversations_df = pred_attack_issue_conversations_df[cols]
        # save to disk for manual inspection
        pred_attack_issue_conversations_df.to_csv(os.path.join('./logs/analysis', 'pred_attack_issue_conversations.csv'), index=False)

        # gpt4_attack_df
        gpt4_attack_df = gpt4_attack_df.merge(todd_annotated_df[['id', 'social_orientation_todd']], on='id', how='left')
        gpt4_attack_df['gpt_4_correct'] = gpt4_attack_df['prediction_gpt4'] == gpt4_attack_df['cga_awry']
        logging.info(f"Disagreeing conversations where the GPT-4 supported model correctly predicts attack:\n{gpt4_attack_df.groupby('conversation_id')['gpt_4_correct'].max().value_counts()}")
        # in gpt4_attack_df, examine conversations that contain the following patterns:
        # pred model predicts ['Unassuming-Ingenuous', 'Unassured-Submissive', 'Warm-Agreeable'] while the gpt4
        # model predicts ['Cold', 'Arrogant-Calculating', 'Assured-Dominant']
        conversation_ids = gpt4_attack_df[(gpt4_attack_df['social_orientation_pred'].isin(['Unassuming-Ingenuous', 'Unassured-Submissive', 'Warm-Agreeable'])) & (gpt4_attack_df['social_orientation_gpt4'].isin(['Cold', 'Arrogant-Calculating', 'Assured-Dominant']))]['conversation_id'].unique()
        # get the conversations
        gpt4_attack_issue_conversations_df = gpt4_attack_df[gpt4_attack_df['conversation_id'].isin(conversation_ids)]
        # organize the columns
        cols = ['id', 'conversation_id', 'speaker', 'original_text', 'social_orientation_pred', 'social_orientation_gpt4', 'social_orientation_todd', 'prediction_pred', 'prediction_gpt4', 'cga_awry', 'text']
        gpt4_attack_issue_conversations_df = gpt4_attack_issue_conversations_df[cols]
        # save to disk for manual inspection
        gpt4_attack_issue_conversations_df.to_csv(os.path.join('./logs/analysis', 'gpt4_attack_issue_conversations.csv'), index=False)
        exit(0)
    elif args.analysis_mode == 'human-eval':
        # load GPT-4 social orientation predictions and optionally load model predictions
        # TODO: fix this hack: currently using args.prediction_filepaths in a hardcoded order
        # assume cga with GPT-4 predicted social tags is first, then cga with predicted social tags, optionally
        cga_gpt4_df = pd.read_csv(args.prediction_filepaths[0])
        rename_gpt4_cols = {'prediction': 'prediction_gpt4', 'social_orientation': 'social_orientation_gpt4'}
        cga_gpt4_df.rename(columns=rename_gpt4_cols, inplace=True)
        pred_df = cga_gpt4_df
        if len(args.prediction_filepaths) > 1:
            cga_pred_df = pd.read_csv(args.prediction_filepaths[1])
            # rename columns to keep track of which model is which
            rename_pred_cols = {'prediction': 'prediction_pred', 'social_orientation': 'social_orientation_pred'}
            cga_pred_df.rename(columns=rename_pred_cols, inplace=True)

            # merge dfs
            cols = rename_gpt4_cols.values()
            pred_df = pd.concat([cga_pred_df, pred_df[cols]], axis=1)

        # load human annotations and merge with predictions
        human_dfs = {}
        for p in args.human_annotation_filepaths:
            annotator = p.split(' - ')[-1].split('.')[0]
            temp_df = pd.read_csv(p)
            social_col_name = f'social_orientation_{annotator}'
            temp_df.rename(columns={'social orientation': social_col_name}, inplace=True)
            # merge with predictions
            # NB: we're dropping some additional columns here
            temp_df = pred_df.merge(temp_df[['id', social_col_name]], on='id', how='inner')
            human_dfs[social_col_name] = temp_df


        # compute accuracy for each human annotator
        for key in human_dfs:
            temp_df = human_dfs[key]
            # print classification report
            logging.info(f"Classification report for GPT-4 against {key}\n{classification_report(temp_df[key], temp_df['social_orientation_gpt4'], zero_division=0)}")
            if 'prediction_pred' in temp_df.columns:
                logging.info(f"Classification report for predicted against {key}\n{classification_report(temp_df[key], temp_df['social_orientation_pred'], zero_division=0)}")

            # create/save confusion matrix
            # plot confusion matrix
            plot_confusion_matrix(temp_df[key], temp_df['social_orientation_gpt4'], list(SOCIAL_ORIENTATION_LABEL2ID.keys()), model_name=f'GPT-4 vs. {key}', output_dir='./logs/analysis')

        # compute inter-annotator agreement
        # merge all human_dfs on id
        merged_df = human_dfs['social_orientation_Todd']
        for key in human_dfs:
            if key != 'social_orientation_Todd':
                merged_df = merged_df.merge(human_dfs[key], on='id', how='inner')

        # human_dfs['social_orientation_Todd']['social_orientation_Amith'] = random.choices(list(SOCIAL_ORIENTATION_LABEL2ID.keys()), k=len(human_dfs['social_orientation_Todd']))
        # human_dfs['social_orientation_Todd']['social_orientation_Yanda'] = random.choices(list(SOCIAL_ORIENTATION_LABEL2ID.keys()), k=len(human_dfs['social_orientation_Todd']))
        # Fleiss' kappa
        # aggregate ratings
        formatted_labels = aggregate_raters(merged_df[['social_orientation_Todd', 'social_orientation_Amith', 'social_orientation_Yanda']].applymap(lambda x: SOCIAL_ORIENTATION_LABEL2ID[x]), len(SOCIAL_ORIENTATION_LABEL2ID))
        # compute Fleiss' kappa
        kappa = fleiss_kappa(formatted_labels[0])
        logging.info(f"Fleiss' kappa for social orientation: {kappa:.4f}")

        # compute pairwise agreement
        # get all pairwise combinations of annotators
        annotators = ['social_orientation_Todd', 'social_orientation_Amith', 'social_orientation_Yanda']
        pairs = list(itertools.combinations(annotators, 2))
        for pair in pairs:
            annotator1, annotator2 = pair
            # compute pairwise agreement
            agreement = merged_df[annotator1] == merged_df[annotator2]
            agreement_rate = agreement.mean()
            logging.info(f"Pairwise agreement between {annotator1} and {annotator2}: {agreement_rate:.4f}")

            # value_counts where agreement is True and False
            # NB: we're implicitly assuming that annotator1 is the "correct" annotator
            agreement_True = merged_df[agreement][annotator1].value_counts().to_frame()
            agreement_True['agreement_rate'] = agreement_True['count'] / agreement_True['count'].sum()
            agreement_False = merged_df[~agreement][annotator1].value_counts().to_frame()
            agreement_False['disagreement_rate'] = agreement_False['count'] / agreement_False['count'].sum()
            logging.info(f"Value counts for {annotator1} and {annotator2} where agreement is True:\n{agreement_True}")
            logging.info(f"Value counts for {annotator1} and {annotator2} where agreement is False:\n{agreement_False}")

            # create/save confusion matrix
            # plot confusion matrix
            plot_confusion_matrix(merged_df[annotator1], merged_df[annotator2], list(SOCIAL_ORIENTATION_LABEL2ID.keys()), model_name=f'{annotator1} vs. {annotator2}', output_dir='./logs/analysis', xlabel=annotator2, ylabel=annotator1)

        exit(0)
    elif args.analysis_mode == 'data-ablation':
        df = load_experiments(args)
        # aggregate results across seeds and perform t-test
        subset = 'test' # 'val'
        agg_df = df.groupby(['method', 'social_orientation_prediction', 'subset_pct']).agg({f'{subset}_acc': ['mean', 'std'], f'{subset}_loss': ['mean', 'std']})
        agg_df = agg_df.unstack(level=[0, 1])[[f'{subset}_acc']]
        print(agg_df)
        # multiply subset_pct by 100
        agg_df.index = agg_df.index * 100
        # multiply all columns with mean in level 1 by 100
        agg_df.loc[:, (slice(None), 'mean', slice(None))] = agg_df.loc[:, (slice(None), 'mean', slice(None))] * 100
        # same for std. columns
        agg_df.loc[:, (slice(None), 'std', slice(None))] = agg_df.loc[:, (slice(None), 'std', slice(None))] * 100

                
        # plot results with standard deviation shadows
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = set(agg_df.columns.get_level_values(2))
        social_orientation_sources = set(agg_df.columns.get_level_values(3))

        method_names = {'distilbert': 'DistilBERT', 'logistic_tfidf': 'Logistic (TF-IDF)', 'logistic_social_counts': 'Logistic (Social Counts)', 'logistic_valence_counts': 'Logistic (Valence Counts)', 'logistic_sentiment': 'Logistic (Sentiment)', 'logistic_distilbert': 'Logistic (DistilBERT)'}
        sources = {'GPT-4': 'GPT-4', 'winsize_2_model_distilbert-base-uncased': 'Predicted'}
        # print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        # breakpoint()
        # establish consistent color scheme
        # colors = {'DistilBERT': 'orange', 'DistilBERT - GPT-4': 'red', 'DistilBERT - Predicted': 'blue', 'Logistic (DistilBERT)': 'purple', 'Logistic (Sentiment)': 'brown', 'Logistic (Social Counts) - Predicted': 'green', 'Logistic (TF-IDF)': 'pink'}
        colors = {'DistilBERT': '#ff7f0e', 'DistilBERT - GPT-4': '#d62728', 'DistilBERT - Predicted': '#1f77b4', 'Logistic (DistilBERT)': '#9467bd', 'Logistic (Sentiment)': '#8c564b', 'Logistic (Social Counts) - Predicted': '#2ca02c', 'Logistic (TF-IDF)': '#e377c2'}

        for method in methods:
            if method == 'logistic_valence_counts':
                continue
            for source in social_orientation_sources:
                legend_entry = f'{method_names[method]}'
                if source != 'None' and method != 'logistic_sentiment':
                    legend_entry += f' - {sources[source]}'
                # if column is all NaNs, skip it
                if agg_df[(f'{subset}_acc', 'mean', method, source)].isna().all():
                    continue
                ax.plot(agg_df.index, agg_df[(f'{subset}_acc', 'mean', method, source)], label=legend_entry, color=colors[legend_entry])
                # exclude border
                ax.fill_between(agg_df.index,
                                agg_df[(f'{subset}_acc', 'mean', method, source)] - agg_df[(f'{subset}_acc', 'std', method, source)],
                                agg_df[(f'{subset}_acc', 'mean', method, source)] + agg_df[(f'{subset}_acc', 'std', method, source)],
                                alpha=0.1, color=colors[legend_entry], edgecolor=None)
        if args.dataset == 'cga-cmv':
            dataset = 'CGA CMV'
        elif args.dataset == 'cga':
            dataset = 'CGA'
        ax.set_title(f'{dataset} Model Accuracy by Subset Percentage', fontsize=16)
        ax.set_xlabel('Subset Percentage', fontsize=14)
        subset_name = 'Test' if subset == 'test' else 'Validation'
        ax.set_ylabel(f'{subset_name} Accuracy', fontsize=14)
        # sort legend entries
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels, loc='lower right', fontsize=12)
        plt.savefig(os.path.join(args.analysis_dir, f'subset_ablation_{args.dataset}_{subset}.png'), dpi=300, bbox_inches='tight')
        plt.clf()
        exit(0)
    elif args.analysis_mode == 'explainability':
        # TODO: refactor the logistic regression pipeline into train.py
        df, corpus = load_data(args.data_dir, include_speakers=args.include_speakers, social_orientation_filepaths=args.social_orientation_filepaths, include_social_orientation=args.include_social_orientation)
        # wrap this in a loop so we can specify subset_pct
        args.subset_pcts = [1.0] if args.subset_pcts is None else args.subset_pcts
        # method,window_size,social_orientation,seed,subset_pct,social_orientation_filepaths,val_acc,val_loss
        results = []
        original_seed = args.seed
        for offset in range(args.num_runs):
            seed = original_seed + offset
            args.seed = seed
            for subset_pct in args.subset_pcts:
                set_random_seed(args.seed)
                args.subset_pct = subset_pct
                social_source = 'gpt4' if 'gpt4' in args.social_orientation_filepaths[0] else 'predicted'
                train_df, val_df, test_df, _ = get_data_splits(df, args.data_dir, subset_pct=subset_pct)
                # if args.dataset == 'cga':
                #     cga_explainability(train_df, val_df, corpus, args, social_source=social_source)
                # elif args.dataset == 'casino-satisfaction' or args.dataset == 'casino-opponent-likeness':
                #     explainability(train_df, val_df, corpus, args, social_source=social_source)
                # elif args.dataset == 'cga-cmv':
                count_val_acc, first_2_val_acc = explainability(train_df, val_df, corpus, args, social_source=social_source)
                results.append(('logistic', 'all', np.nan, seed, subset_pct, args.social_orientation_filepaths, count_val_acc, np.nan))
                results.append(('logistic', 2, np.nan, seed, subset_pct, args.social_orientation_filepaths, first_2_val_acc, np.nan))

        # save results
        results_df = pd.DataFrame(results, columns=['method', 'window_size', 'social_orientation', 'seed', 'subset_pct', 'social_orientation_filepaths', 'val_acc', 'val_loss'])
        filename = f'subset_{args.dataset}_logistic_clf.csv'
        filepath = os.path.join(args.exp_dir, filename)
        results_df.to_csv(filepath, index=False)
        exit(0)
    elif args.analysis_mode == 'gpt-4-preds':
        # load GPT-4 social orientation predictions and analyze label distribution
        # TODO: refactor the logistic regression pipeline into train.py
        df, corpus = load_data(args.data_dir, include_speakers=args.include_speakers, social_orientation_filepaths=args.social_orientation_filepaths, include_social_orientation=args.include_social_orientation)
        # plot label distribution
        df['social_orientation'].value_counts().plot(kind='bar')
        # plt.title('Social Orientation Label Distribution')
        plt.xlabel('Social Orientation Label', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        # add commas to y-axis
        plt.gca().yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.ylabel('Count', fontsize=16)
        plt.yticks(fontsize=14)
        plt.savefig(os.path.join(args.analysis_dir, 'social_orientation_label_distribution.png'), dpi=300, bbox_inches='tight')
        plt.clf()
        exit(0)
    elif args.analysis_mode == 'social-eval':
        df, corpus = load_data(args.data_dir, include_speakers=args.include_speakers, social_orientation_filepaths=args.social_orientation_filepaths, include_social_orientation=args.include_social_orientation)

        # plot confusion matrix comparing social orientation predictions to GPT-4 predictions
        predictions = [os.path.expanduser(p) for p in args.predicted_social_orientation_filepaths]
        pred_dfs = []
        for p in predictions:
            temp_df = pd.read_csv(p)
            pred_dfs.append(temp_df)
        pred_df = pd.concat(pred_dfs)
        # rename social_orientation to social_orientation_prediction
        pred_df.rename(columns={'social_orientation': 'social_orientation_prediction'}, inplace=True)

        # load GPT-4 predictions
        labels = [os.path.expanduser(p) for p in args.social_orientation_filepaths]
        label_dfs = []
        for p in labels:
            temp_df = pd.read_csv(p)
            label_dfs.append(temp_df)
        label_df = pd.concat(label_dfs)
        
        # merge predictions with ground truth
        cols = ['conversation_id', 'utterance_id', 'speaker', 'social_orientation_prediction']
        merge_on = ['conversation_id', 'utterance_id', 'speaker']
        label_df = label_df.merge(pred_df[cols], on=merge_on, how='inner')
        assert len(label_df) == len(pred_df), 'Mismatch in number of rows after merge'

        # identify CGA splits
        train_df, val_df, test_df, _ = get_data_splits(df, args.data_dir)

        # plot confusion matrix
        labels = list(SOCIAL_ORIENTATION_LABEL2ID.keys())
        for split, df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
            logging.info(f'Evaluating {split} split')
            temp_df = label_df[label_df['conversation_id'].isin(df['conversation_id'])]
            plot_confusion_matrix(temp_df['social_orientation'], temp_df['social_orientation_prediction'], labels, output_dir='./logs/analysis', xlabel='Predicted Social Orientation', ylabel='GPT-4 Social Orientation', split=split)
            logging.info(f'Accuracy: {accuracy_score(temp_df["social_orientation"], temp_df["social_orientation_prediction"])}')
            # compute classification report
            logging.info(classification_report(temp_df['social_orientation'], temp_df['social_orientation_prediction'], labels=labels))


if __name__ == '__main__':
    args = parse_args()
    main(args)