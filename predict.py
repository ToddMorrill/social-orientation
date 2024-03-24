"""This module runs predictions on the passed examples and optionally saves the
predictions to a file.

Examples:    
    $ python predict.py \
        --model-dir model/distilbert-social-winsize-2 \
        --window-size 2 \
        --checkpoint best \
        --dataset social-orientation \
        --include-speakers \
        --social-orientation-filepaths \
            data/gpt-4-cga-social-orientation-labels/train_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/val_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/train-long_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/val-long_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/test_results_gpt4_parsed.csv \
            data/gpt-4-cga-social-orientation-labels/test-long_results_gpt4_parsed.csv \
        --prediction-dir data/predictions \
        --batch-size 256 \
        --disable-train-shuffle
        
    $ python predict.py \
        --model-dir model/distilbert-social-winsize-2 \
        --window-size 2 \
        --checkpoint best \
        --dataset cga-cmv \
        --data-dir ~/Documents/data/convokit/conversations-gone-awry-cmv-corpus \
        --include-speakers \
        --prediction-dir predictions \
        --batch-size 256 \
        --add-tokens \
        --return-utterances \
        --disable-train-shuffle \
        --dont-return-labels

    $ python predict.py \
        --model-dir model/xlmr-social-winsize-2 \
        --model-name-or-path xlm-roberta-base \
        --window-size 2 \
        --checkpoint best \
        --dataset social-orientation \
        --include-speakers \
        --prediction-dir predictions \
        --batch-size 256 \
        --add-tokens \
        --return-utterances \
        --disable-train-shuffle \
        --social-orientation-filepaths \
            ~/Documents/data/circumplex/transformed/train_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/val_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/train-long_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/val-long_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/test_results_gpt4_parsed.csv \
            ~/Documents/data/circumplex/transformed/test-long_results_gpt4_parsed.csv
"""
import logconfig
import logging
import os
from args import parse_args
from tqdm import tqdm

import torch
from callbacks import Accuracy
from data import get_data_loaders, get_labels, get_tokenizer, SOCIAL_ORIENTATION_LABEL2ID, SOCIAL_ORIENTATION_ID2LABEL
from train import get_model
from utils import get_checkpoint, add_utterance_id


class Predictor(object):

    def __init__(self,
                 args,
                 tokenizer,
                 tokens2ids=None,
                 id2label=None,
                 label2id=None) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.tokens2ids = tokens2ids
        self.id2label = id2label
        self.label2id = label2id
        self._load_model()

    def _tokenize(self, texts: list[str]):
        inputs = self.tokenizer(texts, return_tensors='pt')
        # move input to device
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        return inputs

    def predict(
        self,
        text_or_loader: str | list[str] | torch.utils.data.DataLoader,
    ):
        """Generates predictions for the given samples."""
        # handle raw text instances
        if isinstance(text_or_loader, str) or isinstance(text_or_loader, list):
            if isinstance(text_or_loader, str):
                texts = [text_or_loader]
            elif isinstance(text_or_loader, list):
                texts = text_or_loader
            inputs = self._tokenize(texts)  # also moves inputs to device
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                predictions = [
                    self.id2label[prediction.item()]
                    for prediction in predictions
                ]
            labels = torch.tensor([])
            return predictions, logits, labels

        # otherwise, assume it's a dataloader
        loader = text_or_loader
        predictions = []
        labels = []
        logits = []
        for batch in tqdm(loader):
            inputs = {k: v.to(self.args.device) for k, v in batch.items()}
            if 'labels' in batch:
                labels.append(batch['labels'])

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_logits = outputs.logits
                logits.append(batch_logits)
                batch_predictions = torch.argmax(batch_logits, dim=1)
                predictions.append(batch_predictions)

        # convert predictions to labels
        predictions = torch.cat(predictions, dim=0).cpu()
        predictions = [
            self.id2label[prediction.item()] for prediction in predictions
        ]
        # convert to tensors
        if len(labels) > 0:
            labels = torch.cat(labels, dim=0)
        else:
            labels = torch.tensor([])
        logits = torch.cat(logits, dim=0)
        labels = labels.to(self.args.device)
        return predictions, logits, labels

    def _load_model(self):
        # get model architecture
        self.model = get_model(self.args, self.label2id, self.id2label,
                               self.tokenizer, self.tokens2ids)

        # load checkpoint
        if os.path.isdir(self.args.checkpoint):
            checkpoint = self.args.checkpoint
        # otherwise, use the specified strategy to get the checkpoint
        else:
            checkpoint = get_checkpoint(model_dir=self.args.model_dir,
                                        strategy=self.args.checkpoint)

        if checkpoint is None:
            raise ValueError(f'No checkpoint found for {self.args.checkpoint}')

        logging.info(f'Loading checkpoint: {checkpoint}')
        self.model.load_state_dict(
            torch.load(os.path.join(checkpoint, 'model.pt'),
                       map_location=self.args.device))
        self.model.to(self.args.device)
        self.model.eval()

        # if parallel
        if self.args.parallel:
            self.model = torch.nn.DataParallel(self.model)
            # extract id2label
            self.id2label = self.model.module.config.id2label
        else:
            self.id2label = self.model.config.id2label


def main(args):
    # get the tokenizer
    # TODO: save the tokenizer, especially if we've modified it
    # the current approach is to transform the tokenizer in the exact same way
    # as was done during training, but this this is error-prone
    label2id, id2label = get_labels(args)
    # TODO: revisit if we're ever going to need to run inference for the
    # downstream classifiers (e.g., cga-cmv) versus just the social orientation
    label2id = SOCIAL_ORIENTATION_LABEL2ID
    id2label = SOCIAL_ORIENTATION_ID2LABEL
    added_tokens = SOCIAL_ORIENTATION_LABEL2ID.keys(
    ) if args.add_tokens else []
    tokenizer, tokens2ids = get_tokenizer(args, added_tokens)

    # load data
    train_loader, val_loader, test_loader = get_data_loaders(args, tokenizer)
    predictor = Predictor(args,
                          tokenizer,
                          tokens2ids=tokens2ids,
                          id2label=id2label,
                          label2id=label2id)
    sample = 'I am a very social person!'
    predictions = predictor.predict(sample)
    logging.debug(predictions[0])

    # get predictions and save results
    model_name = args.model_name_or_path.replace('/', '-')
    task = 'social'  # this is hardcoded for now
    complete_task = args.dataset + '-' + task
    prediction_dir = os.path.join(args.prediction_dir, complete_task,
                                  model_name)
    os.makedirs(prediction_dir, exist_ok=True)
    columns = [
        'id', 'conversation_id', 'chunk_id', 'utterance_id', 'speaker',
        'social_orientation'
    ]
    results = {}
    for split, loader in zip(['train', 'val', 'test'],
                             [train_loader, val_loader, test_loader]):
        logging.info(f'Predicting on {split} set')
        predictions, logits, labels = predictor.predict(loader)
        results[split] = {
            'predictions': predictions,
            'logits': logits,
            'labels': labels
        }
        # TODO: generalize this handle datasets other than social orientation
        if len(results[split]
               ['labels']) > 0 and args.dataset == 'social-orientation':
            accuracy = Accuracy()
            accuracy.update(logits, labels)
            logging.info(
                f'Accuracy on the {split} set: {accuracy.compute()*100:.2f}%')

        # merge predictions into dataframe
        df = loader.dataset.df
        assert len(df) == len(predictions)
        df['social_orientation'] = predictions
        # if chunk_id not in df, then add it
        if 'chunk_id' not in df.columns:
            df['chunk_id'] = 0
        # if original_text == '' or pd.isna(original_text) then predict Not Available
        empty_text = df['original_text'].apply(
            lambda x: len(x) == 0) | df['original_text'].isna()
        df.loc[empty_text, 'social_orientation'] = 'Not Available'
        df = df.groupby('conversation_id',
                        group_keys=False).apply(add_utterance_id)

        # save predictions
        # replace any slashes in model name with underscores
        model_name_or_path = args.model_name_or_path.replace('/', '_')
        save_filepath = os.path.join(
            prediction_dir,
            f'{split}_winsize_{args.window_size}_model_{model_name_or_path}.csv'
        )
        logging.info(f'Saving predictions to {save_filepath}')
        df[columns].to_csv(save_filepath, index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
