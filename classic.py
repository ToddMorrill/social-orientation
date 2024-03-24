"""This module implements classic machine learning models like Logistic Regression
paired with hand-crafted features.

Examples:
    $ python classic.py \
        --dataset cga \
        --include-speakers \
        --social-orientation-filepaths \
            data/predictions/social-orientation-social/distilbert-base-uncased/train_winsize_2_model_distilbert-base-uncased.csv \
            data/predictions/social-orientation-social/distilbert-base-uncased/val_winsize_2_model_distilbert-base-uncased.csv \
            data/predictions/social-orientation-social/distilbert-base-uncased/test_winsize_2_model_distilbert-base-uncased.csv \
        --num-runs 5 \
        --subset-pcts 0.01 0.1 0.2 0.5 1.0 \
        --window-size all \
        --features social_counts distilbert tfidf sentiment \
        --use-cache \
        --analysis-dir logs/analysis \
        --experiment-dir logs/experiments
    
    $ python classic.py \
        --dataset cga-cmv \
        --data-dir ~/Documents/data/convokit/conversations-gone-awry-cmv-corpus \
        --include-speakers \
        --social-orientation-filepaths \
            predictions/cga-cmv-social/distilbert-base-uncased/train_winsize_2_model_distilbert-base-uncased.csv \
            predictions/cga-cmv-social/distilbert-base-uncased/val_winsize_2_model_distilbert-base-uncased.csv \
            predictions/cga-cmv-social/distilbert-base-uncased/test_winsize_2_model_distilbert-base-uncased.csv \
        --num-runs 5 \
        --subset-pcts 0.01 0.1 0.2 0.5 1.0 \
        --window-size all \
        --features social_counts distilbert tfidf sentiment \
        --use-cache
"""
import numpy as np
import torch
from tqdm import tqdm
from args import parse_args
import logging
import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline
from data import SOCIAL_ORIENTATION_LABEL2ID, get_data_loaders, get_data_splits, load_data
from utils import set_random_seed
from transformers import DistilBertModel, DistilBertTokenizer


def get_windowed_social_orientation_counts(df,
                                           args,
                                           window_size=2,
                                           normalize=True):
    """Returns counts of social orientation tags in a window of size window_size.
    
    Args:
        df (pd.DataFrame): dataframe containing social orientation tags
        args (argparse.Namespace): arguments
        window_size (int): size of window to use
        normalize (bool): whether to normalize counts by the total number of tags in the window
    """
    # get counts of social orientation tags in the window
    # include all turns except the last one
    # window size can either be an int or None
    # if None, we take all turns
    s = slice(None, window_size, None)
    if args.dataset == 'cga':
        social_counts_df = df.groupby(
            'conversation_id')['social_orientation'].apply(
                lambda x: x.iloc[:-1].iloc[s].value_counts()).unstack(level=1)
    else:
        # otherwise take all turns, even the last one
        social_counts_df = df.groupby(
            'conversation_id')['social_orientation'].apply(
                lambda x: x.iloc[s].value_counts()).unstack(level=1)

    # fill in missing values
    social_counts_df.fillna(0, inplace=True)
    # normalize counts
    if normalize:
        social_counts_df = social_counts_df.div(social_counts_df.sum(axis=1),
                                                axis=0)
    # ensure that all columns are present and if not, fill with 0
    for label in SOCIAL_ORIENTATION_LABEL2ID.keys():
        if label not in social_counts_df.columns:
            social_counts_df[label] = 0.0
    # arrange columns according to SOCIAL_ORIENTATION_LABEL2ID
    social_counts_df = social_counts_df[SOCIAL_ORIENTATION_LABEL2ID.keys()]
    return social_counts_df


def get_windowed_sentiment_counts(df,
                                  pipeline,
                                  args,
                                  window_size=2,
                                  normalize=True):
    """Returns counts of sentiment tags in a window of size window_size."""
    # make predictions, truncate tokenized input, as need
    max_length = 512
    preds = pipeline(df['text'].values.tolist(),
                     truncation=True,
                     batch_size=args.batch_size,
                     max_length=max_length)
    df['sentiment'] = [pred['label'] for pred in preds]
    # get counts of sentiment tags in the window
    # include all turns except the last one
    # window size can either be an int or None
    # if None, we take all turns
    s = slice(None, window_size, None)
    if args.dataset == 'cga':
        sentiment_counts_df = df.groupby('conversation_id')['sentiment'].apply(
            lambda x: x.iloc[:-1].iloc[s].value_counts()).unstack(level=1)
    else:
        # otherwise take all turns, even the last one
        sentiment_counts_df = df.groupby('conversation_id')['sentiment'].apply(
            lambda x: x.iloc[s].value_counts()).unstack(level=1)

    # fill in missing values
    sentiment_counts_df.fillna(0, inplace=True)
    # normalize counts
    if normalize:
        sentiment_counts_df = sentiment_counts_df.div(
            sentiment_counts_df.sum(axis=1), axis=0)

    labels = ['negative', 'neutral', 'positive']
    # ensure that all columns are present and if not, fill with 0
    for label in labels:
        if label not in sentiment_counts_df.columns:
            sentiment_counts_df[label] = 0.0
    # arrange columns
    sentiment_counts_df = sentiment_counts_df[labels]
    return sentiment_counts_df


def tfidf_features(df, args, tfidf_model=None):
    """Returns TFIDF features for each conversation in the dataframe.
    
    Args:
        df (pd.DataFrame): dataframe containing social orientation tags
        args (argparse.Namespace): arguments
    """
    # prepare text for TFIDF
    # include all turns except the last one
    # window size can either be an int or None
    # if None, we take all turns
    s = slice(None, args.window_size, None)
    if args.dataset == 'cga':
        conversation_text = df.groupby('conversation_id')['text'].apply(
            lambda x: '\n'.join(x.iloc[:-1].iloc[s].values.tolist()))
    else:
        # otherwise take all turns, even the last one
        conversation_text = df.groupby('conversation_id')['text'].apply(
            lambda x: '\n'.join(x.iloc[s].values.tolist()))
    if tfidf_model is None:
        tfidf_model = TfidfVectorizer(max_features=10000)
        tfidf_model.fit(conversation_text)
    tfidf_features = tfidf_model.transform(conversation_text)
    return tfidf_features, tfidf_model


def distilbert_features_labels(loader, args, model):
    """Returns DistilBERT features for each conversation in the dataframe.
    
    Args:
        loader (torch.utils.data.DataLoader): dataloader containing tokenized conversations
        args (argparse.Namespace): arguments
        model (DistilBertModel): DistilBERT model
    """
    cls_features = []
    labels = []
    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        labels.extend(batch['labels'])
        outputs = model(input_ids, attention_mask=attention_mask)
        # get CLS token for each conversation
        features = outputs.last_hidden_state[:, 0, :].detach()
        cls_features.append(features)
    cls_features = torch.cat(cls_features, dim=0).cpu().numpy()
    return cls_features, labels


def get_conversation_labels(df, corpus, args):
    """Returns labels for each conversation in the dataframe.
    
    Args:
        df (pd.DataFrame): dataframe containing social orientation tags
        corpus (Corpus): corpus containing conversations
        args (argparse.Namespace): arguments
    """
    if args.dataset == 'cga':
        y = df.groupby('conversation_id').apply(
            lambda x: corpus.get_conversation(x['conversation_id'].iloc[
                0]).meta['conversation_has_personal_attack']).astype(int)
    elif args.dataset == 'cga-cmv':
        y = df.groupby('conversation_id').apply(
            lambda x: corpus.get_conversation(x['conversation_id'].iloc[
                0]).meta['has_removed_comment']).astype(int)
    return y


def classifier(train_df,
               val_df,
               test_df,
               y_train,
               y_val,
               y_test,
               args,
               count_vector=True,
               social_source='gpt4',
               save_coefficients=False):
    """Trains a simple logistic classifier using only social orientation tags.
    
    If count_vector=True, uses counts of all social orientation tags in the conversation
    as features. Otherwise, creates a binary feature for each social orientation tag for
    the first 2 utterances in the conversation
    """
    X_train = train_df
    X_val = val_df
    X_test = test_df
    if isinstance(train_df, pd.DataFrame):
        X_train = train_df.values
    if isinstance(val_df, pd.DataFrame):
        X_val = val_df.values
    if isinstance(test_df, pd.DataFrame):
        X_test = test_df.values

    clf = LogisticRegression(random_state=args.seed,
                             max_iter=1000).fit(X_train, y_train)
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    logging.info(
        f'Simple classifier accuracy with {args.subset_pct*100:.2f}% of the data: {val_acc:.4f}'
    )
    logging.info(
        f'\n{classification_report(y_val, val_preds, zero_division=0)}')
    val_clf_report = classification_report(y_val,
                                           val_preds,
                                           zero_division=0,
                                           output_dict=True)

    test_preds = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    # logging.info(f'Simple classifier accuracy on test set: {test_acc:.4f}')
    # logging.info(f'\n{classification_report(y_test, test_preds, zero_division=0)}')
    test_clf_report = classification_report(y_test,
                                            test_preds,
                                            zero_division=0,
                                            output_dict=True)

    # examine coefficients (note: only works when train_df is a pandas dataframe)
    if save_coefficients:
        coefficients_df = pd.DataFrame({
            'Feature': train_df.columns,
            'Coefficient': clf.coef_[0]
        })
        coefficients_df['abs_coefficient'] = coefficients_df[
            'Coefficient'].abs()
        coefficients_df.sort_values(by='abs_coefficient',
                                    ascending=False,
                                    inplace=True)
        coefficients_df.drop(columns=['abs_coefficient'], inplace=True)
        logging.info(f'Most important features:\n{coefficients_df.head(10)}')
        # save coefficients to disk
        count_all = 'count_all' if count_vector else 'count_first_two'
        coefficients_df.to_csv(os.path.join(
            args.analysis_dir,
            f'{args.dataset}_logistic_classifier_coefficients_{count_all}_{social_source}_{args.subset_pct}.csv'
        ),
                               index=False)
    return clf, val_acc, val_clf_report, test_acc, test_clf_report


def run_classifier_subset_experiments(args):
    # use a cache so we don't rerun experiments everytime
    exp_cache_filepath = os.path.join(
        args.exp_dir, f'subset_{args.dataset}_logistic_clf.csv')
    results_df = pd.DataFrame()
    if os.path.exists(exp_cache_filepath) and args.use_cache:
        logging.info(f'Loading results from {exp_cache_filepath}')
        results_df = pd.read_csv(exp_cache_filepath)

    # load data
    df, corpus = load_data(
        args.data_dir,
        include_speakers=args.include_speakers,
        social_orientation_filepaths=args.social_orientation_filepaths,
        include_social_orientation=args.include_social_orientation)

    # load models once
    if 'distilbert' in args.features:
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        model.eval()
        model.to(args.device)
        tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased')
    if 'sentiment' in args.features:
        device = torch.device(
            "cpu"
        ) if args.device == 'cpu' else f'cuda:{torch.cuda.current_device()}'
        sentiment_pipeline = pipeline(
            'sentiment-analysis',
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device)

    args.subset_pcts = [1.0] if args.subset_pcts is None else args.subset_pcts
    # method,window_size,social_orientation,seed,subset_pct,social_orientation_filepaths,val_acc,val_loss
    results = []
    original_seed = args.seed
    for offset in range(args.num_runs):
        seed = original_seed + offset
        args.seed = seed
        for subset_pct in args.subset_pcts:
            for features in args.features:
                # check if we've already run this experiment
                if len(results_df) > 0:
                    rows = (results_df['seed'] == seed) & (
                        results_df['subset_pct'] == subset_pct) & (
                            results_df['method'] == f'logistic_{features}')
                    if rows.any():
                        logging.info(
                            f'Skipping logistic_{features} experiment with seed {seed} and subset_pct {subset_pct}'
                        )
                        continue
                logging.info(
                    f'Running logistic_{features} experiment with seed {seed} and subset_pct {subset_pct}'
                )

                set_random_seed(args.seed)
                args.subset_pct = subset_pct
                social_source = 'gpt4' if 'gpt4' in args.social_orientation_filepaths[
                    0] else 'predicted'
                train_df, val_df, test_df, data_splits = get_data_splits(
                    df, args.data_dir, subset_pct=subset_pct)
                # get labels
                y_train = get_conversation_labels(train_df, corpus, args)
                y_val = get_conversation_labels(val_df, corpus, args)
                y_test = get_conversation_labels(test_df, corpus, args)

                # get features
                if features == 'tfidf':
                    train_features_df, tfidf_model = tfidf_features(
                        train_df, args)
                    val_features_df, _ = tfidf_features(
                        val_df, args, tfidf_model=tfidf_model)
                    test_features_df, _ = tfidf_features(
                        test_df, args, tfidf_model=tfidf_model)
                elif features == 'social_counts':
                    train_features_df = get_windowed_social_orientation_counts(
                        train_df,
                        args,
                        window_size=args.window_size,
                        normalize=True)
                    val_features_df = get_windowed_social_orientation_counts(
                        val_df,
                        args,
                        window_size=args.window_size,
                        normalize=True)
                    test_features_df = get_windowed_social_orientation_counts(
                        test_df,
                        args,
                        window_size=args.window_size,
                        normalize=True)
                elif features == 'distilbert':
                    # ensure we don't shuffle the training data
                    args.disable_train_shuffle = True
                    # load data
                    train_loader, val_loader, test_loader = get_data_loaders(
                        args, tokenizer)
                    # get features
                    # also get labels from the data loaders to ensure everything is aligned
                    train_features_df, y_train = distilbert_features_labels(
                        train_loader, args, model)
                    val_features_df, y_val = distilbert_features_labels(
                        val_loader, args, model)
                    test_features_df, y_test = distilbert_features_labels(
                        test_loader, args, model)
                elif features == 'sentiment':
                    train_features_df = get_windowed_sentiment_counts(
                        train_df,
                        sentiment_pipeline,
                        args,
                        window_size=args.window_size,
                        normalize=True)
                    val_features_df = get_windowed_sentiment_counts(
                        val_df,
                        sentiment_pipeline,
                        args,
                        window_size=args.window_size,
                        normalize=True)
                    test_features_df = get_windowed_sentiment_counts(
                        test_df,
                        sentiment_pipeline,
                        args,
                        window_size=args.window_size,
                        normalize=True)
                # train classifier
                clf, val_acc, val_clf_report, test_acc, test_clf_report = classifier(
                    train_features_df,
                    val_features_df,
                    test_features_df,
                    y_train,
                    y_val,
                    y_test,
                    args,
                    count_vector=True,
                    social_source=social_source,
                    save_coefficients=False)

                # update cache
                result = {
                    'method': f'logistic_{features}',
                    'window_size': 'all',
                    'social_orientation': np.nan,
                    'seed': seed,
                    'subset_pct': subset_pct,
                    'social_orientation_filepaths':
                    args.social_orientation_filepaths,
                    'val_acc': val_acc,
                    'val_loss': np.nan,
                    'test_acc': test_acc,
                    'test_loss': np.nan
                }
                temp_df = pd.DataFrame([result])
                results_df = pd.concat([results_df, temp_df], axis=0)
                results_df.to_csv(exp_cache_filepath, index=False)


def main(args):
    run_classifier_subset_experiments(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
