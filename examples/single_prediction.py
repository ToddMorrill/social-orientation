"""This module shows the most minimal working example of loading the social orientation model and making a prediction."""
import pprint

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    sample_input_1 = 'Speaker 1: This is really terrific work!'
    sample_input_2 = [
        'Speaker 1: These edits are terrible. Please review my comments above again.',
        'Speaker 2: I reviewed your comments, which were not helpful. Roll up your sleeves and do some work.'
    ]
    model = AutoModelForSequenceClassification.from_pretrained(
        'tee-oh-double-dee/social-orientation')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        'tee-oh-double-dee/social-orientation')
    input_1 = tokenizer(sample_input_1, return_tensors='pt')
    # NOTE: we use the sep_token between speakers
    # NOTE: we're only making a social orientation prediction for Speaker 2
    # (i.e., the last speaker) given the provided context
    sample_input_2 = sample_input_2[0] + tokenizer.sep_token + sample_input_2[1]
    input_2 = tokenizer(sample_input_2, return_tensors='pt')
    output_1 = model(**input_1)
    output_1_probs = output_1.logits.softmax(dim=1)
    output_2 = model(**input_2)
    output_2_probs = output_2.logits.softmax(dim=1)
    id2label = model.config.id2label
    print(f'Input 1: {sample_input_1}')
    pred_dict = {
        id2label[i]: output_1_probs[0][i].item()
        for i in range(len(id2label))
    }
    print(f'Probability predictions for sample input 1:')
    pprint.pprint(pred_dict)
    print()
    print(f'Input 2: {sample_input_2}')
    pred_dict = {
        id2label[i]: output_2_probs[0][i].item()
        for i in range(len(id2label))
    }
    print(f'Probability predictions for sample input 2:')
    pprint.pprint(pred_dict)

if __name__ == '__main__':
    main()
