"""Analyze model predictions.

Examples:
    # plot the distribution of social orientation labels
    $ python analyze.py \
        --analysis-mode gpt-4-preds \
        --dataset cga \
        --social-orientation-filepaths \
            data/gpt-4-cga-social-orientation-labels/train_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/train-long_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/val_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/val-long_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/test_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/test-long_results_gpt4_parsed.csv \
        --analysis-dir logs/analysis
    
    # confusion matrix of social orientation predictions vs. GPT-4 predictions
    $ python analyze.py \
        --analysis-mode social-eval \
        --dataset cga \
        --social-orientation-filepaths \
            data/gpt-4-cga-social-orientation-labels/train_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/train-long_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/val_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/val-long_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/test_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/test-long_results_gpt4_parsed.csv \
        --predicted-social-orientation-filepaths \
            data/predictions/social-orientation-social/distilbert-base-uncased/train_winsize_2_model_distilbert-base-uncased.csv \
            data/predictions/social-orientation-social/distilbert-base-uncased/val_winsize_2_model_distilbert-base-uncased.csv \
            data/predictions/social-orientation-social/distilbert-base-uncased/test_winsize_2_model_distilbert-base-uncased.csv \
        --analysis-dir logs/analysis
    
    $ python analyze.py \
        --analysis-mode outcome-analysis \
        --dataset cga-cmv \
        --data-dir data/convokit/conversations-gone-awry-cmv-corpus \
        --social-orientation-filepaths \
            data/predictions/cga-cmv-social/distilbert-base-uncased/train_winsize_2_model_distilbert-base-uncased.csv \
            data/predictions/cga-cmv-social/distilbert-base-uncased/val_winsize_2_model_distilbert-base-uncased.csv \
            data/predictions/cga-cmv-social/distilbert-base-uncased/test_winsize_2_model_distilbert-base-uncased.csv \
        --analysis-dir logs/analysis
"""
from collections import Counter
from itertools import product
import logging

import numpy as np
from args import parse_args
import os

import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from data import load_data, get_data_splits, SOCIAL_ORIENTATION_LABEL2ID, SocialOrientationDataset, CGA_LABEL2ID

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

def main(args):
    # ensure analysis directory exists
    os.makedirs(args.analysis_dir, exist_ok=True)
    # determine which analysis to perform
    if args.analysis_mode == 't-test' and args.experiment == 'subset':
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
    elif args.analysis_mode == 'data-ablation':
        # this analysis mode is for analyzing the effect of varying the percentage of the training data
        # and plots the learning curves for different methods
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
    elif args.analysis_mode == 'outcome-analysis':
        # plot outcome rates by predicted social orientation label on the CGA CMV dataset
        df, corpus = load_data(args.data_dir, include_speakers=args.include_speakers, social_orientation_filepaths=args.social_orientation_filepaths, include_social_orientation=args.include_social_orientation)

        target_name = 'cga_label'
        label_name = 'has_removed_comment'
        df[target_name] = df['conversation_id'].apply(lambda x: corpus.get_conversation(x).meta[label_name])
        social_splits = df.groupby(target_name)['social_orientation'].value_counts()
        social_splits = social_splits.unstack(level=0)
        social_splits = social_splits.loc[SOCIAL_ORIENTATION_LABEL2ID.keys()]
        social_splits.columns.name = None
        social_splits.columns = ['Success', 'Failure']
        # exclude 'Not Available' label
        social_splits = social_splits.drop('Not Available', errors='ignore')
        social_splits = social_splits.div(social_splits.sum(axis=0))
        # multiply by 100 to get percentage
        social_splits = social_splits * 100
        social_splits.plot(kind='bar')
        plt.xlabel('Social Orientation')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Percentage')
        # plt.title(f'Percentage of Social Orientation Labels by Conversation Outcome')
        plt.tight_layout()
        plt.savefig('logs/analysis/cga_cmv_social_orientation_by_class.png', dpi=300)
        plt.clf()

        # examine co-occurrence rate of social orientation labels with outcome labels
        # get social orientiation tags for each speaker in each conversation
        speakers_to_labels = df.groupby(['conversation_id', 'speaker'])['social_orientation'].apply(set)
        speakers_to_labels = speakers_to_labels.to_frame().reset_index()

        # get co-occurrence counts for each conversation
        co_occurrence_counts = speakers_to_labels.groupby('conversation_id').apply(count_co_occurrences)
        co_occurrence_counts = co_occurrence_counts.to_frame().reset_index().rename(columns={0: 'co_occurrence_counts'})
        co_occurrence_counts['cga_label'] = co_occurrence_counts['conversation_id'].apply(lambda x: corpus.get_conversation(x).meta[label_name])
        # group by outcome label and merge Counter objects
        co_occurence_by_outcome = co_occurrence_counts.groupby('cga_label')['co_occurrence_counts'].apply(lambda x: x.sum())
        civil_df = co_occurence_by_outcome.loc[False].reset_index().rename(columns={'index': 'co_occurrence'})
        civil_df[['speaker', 'other_speakers']] = pd.DataFrame(civil_df['co_occurrence'].tolist())
        civil_df.drop(columns=['co_occurrence'], inplace=True)
        civil_df = civil_df.pivot(index='speaker', columns='other_speakers', values='co_occurrence_counts')
        civil_df.fillna(0, inplace=True)
        # normalize entire matrix
        civil_df = civil_df.div(civil_df.values.sum())

        uncivil_df = co_occurence_by_outcome.loc[True].reset_index().rename(columns={'index': 'co_occurrence'})
        uncivil_df[['speaker', 'other_speakers']] = pd.DataFrame(uncivil_df['co_occurrence'].tolist())
        uncivil_df.drop(columns=['co_occurrence'], inplace=True)
        uncivil_df = uncivil_df.pivot(index='speaker', columns='other_speakers', values='co_occurrence_counts')
        uncivil_df.fillna(0, inplace=True)
        uncivil_df = uncivil_df.div(uncivil_df.values.sum())
        
        # divide uncivil by civil
        ratio_df = uncivil_df / civil_df
        # exclude not available
        ratio_df.drop(columns=['Not Available'], index=['Not Available'], inplace=True, errors='ignore')
        # order index and columns
        order = list(SOCIAL_ORIENTATION_LABEL2ID.keys())
        order.remove('Not Available')
        ratio_df = ratio_df.loc[order, order]
        # plot as heatmap
        sns.heatmap(ratio_df, annot=True)
        plt.xlabel('Other Speakers')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Speaker')
        # plt.title(f'Ratio of Social Orientation Co-occurrences for {"cga"} {target_name}')
        plt.tight_layout()
        plt.savefig('logs/analysis/cga_cmv_social_orientation_co_occurrence_ratio.png', dpi=300)
        plt.clf()
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