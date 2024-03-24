"""This module pushes the model and dataset to Hugging Face Hub."""
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, DatasetInfo

from data import load_data

def push_model_to_hub():
    local_model_path = 'model/distilbert-social-winsize-2-hf/checkpoint-11350'
    model_name = 'social-orientation'
    model_description = 'DistilBERT model for social orientation classification'
    model_tags = ['social-orientation', 'distilbert', 'classification']
    model = AutoModelForSequenceClassification.from_pretrained(
        local_model_path)
    with open('hf_token.txt', 'r') as file:
        token = file.read()
    model.push_to_hub(repo_id=model_name,
                      commit_message='Initial commit',
                      private=True,
                      tags=model_tags,
                      token=token)

    # push the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    tokenizer.push_to_hub(repo_id=model_name,
                           commit_message='Initial commit',
                           private=True,
                           tags=model_tags,
                           token=token)

def push_XLMR_model_to_hub():
    local_model_path = 'model/xlmr-social-winsize-2-hf/checkpoint-8200'
    model_name = 'social-orientation-multilingual'
    model_description = 'XLM-RoBERTa model for social orientation classification'
    model_tags = ['social-orientation', 'xlm-roberta', 'classification']
    model = AutoModelForSequenceClassification.from_pretrained(
        local_model_path)
    with open('hf_token.txt', 'r') as file:
        token = file.read()
    model.push_to_hub(repo_id=model_name,
                      commit_message='Initial commit',
                      private=True,
                      tags=model_tags,
                      token=token)

    # push the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    tokenizer.push_to_hub(repo_id=model_name,
                           commit_message='Initial commit',
                           private=True,
                           tags=model_tags,
                           token=token)


def push_dataset_to_hub():
    cga_data_dir = 'data/convokit/conversations-gone-awry-corpus'
    social_labels_dir = [
        'data/gpt-4-cga-social-orientation-labels/train_results_gpt4_parsed.csv',
        'data/gpt-4-cga-social-orientation-labels/val_results_gpt4_parsed.csv',
        'data/gpt-4-cga-social-orientation-labels/train-long_results_gpt4_parsed.csv',
        'data/gpt-4-cga-social-orientation-labels/val-long_results_gpt4_parsed.csv',
        'data/gpt-4-cga-social-orientation-labels/test_results_gpt4_parsed.csv',
        'data/gpt-4-cga-social-orientation-labels/test-long_results_gpt4_parsed.csv'
    ]
    # NB: have to do it this way because in load_data, we filter out headers
    df, corpus = load_data(cga_data_dir,
                           social_orientation_filepaths=social_labels_dir,
                           include_social_orientation=True)
    full_df = corpus.get_utterances_dataframe()
    # join labels into full_df, which contain headers
    full_df = full_df.merge(df[['id', 'social_orientation']],
                            on='id',
                            how='left')
    full_df['social_orientation'] = full_df['social_orientation'].fillna(
        'Not Available')
    # restrict to [id, social_orientation]
    # in the future, we may want to include complete information from convokit
    # there is a slight concern that a change in IDs would break these labels
    social_orientation_df = full_df[['id', 'social_orientation']]
    assert social_orientation_df['social_orientation'].isna().sum() == 0
    assert social_orientation_df['id'].nunique(
    ) == social_orientation_df.shape[0]

    dataset = Dataset.from_pandas(social_orientation_df)
    dataset_name = 'social-orientation'
    dataset_description = 'Social orientation labels for all utterances in the Conversations Gone Awry corpus.'
    # TODO: update citation
    citation = """@misc{morrill2024social,
      title={Social Orientation: A New Feature for Dialogue Analysis}, 
      author={Todd Morrill and Zhaoyuan Deng and Yanda Chen and Amith Ananthram and Colin Wayne Leach and Kathleen McKeown},
      year={2024},
      eprint={2403.04770},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
        }"""
    license_ = 'MIT'
    dataset.info.dataset_name = dataset_name
    dataset.info.description = dataset_description
    dataset.info.citation = citation
    dataset.info.license = license_

    dataset_tags = ['social-orientation', 'classification']
    with open('hf_token.txt', 'r') as file:
        token = file.read()
    dataset.push_to_hub(repo_id=dataset_name,
                        commit_message='Initial commit',
                        private=True,
                        token=token)

def main():
    # push_model_to_hub()
    # push_dataset_to_hub()
    push_XLMR_model_to_hub()


if __name__ == '__main__':
    main()
