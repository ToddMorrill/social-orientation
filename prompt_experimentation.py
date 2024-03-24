"""This module contains code for testing the prompt-based approach to social
orientation classification.

Examples:
    $ python test.py --experiment single --model gpt-3.5-turbo --temperature 0.0 --top-p 1
    $ python test.py --experiment sweep
    $ python test.py --experiment analyze --output-dir ./logs/prompt-results/sweep_052323_084950
"""
import argparse
from datetime import datetime
import os
import time

import matplotlib.pyplot as plt
import openai
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tiktoken

from data import SOCIAL_ORIENTATION_LABEL2ID
from utils import load_prompt, create_line

# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def parse_row(row):
    """Parse a row of markdown formatted data.
    | utterance_id | participant_id | text | --> dict
    """
    row = row[2:-2]
    row = row.split(' | ')
    # convert to into if possible
    try:
        utterance_id = int(row[0])
    except ValueError:
        utterance_id = row[0]
    participant_id = row[1]
    # rejoin in case there are ' | ' in the text
    label = ' | '.join(row[2:])
    return {
        'utterance_id': utterance_id,
        'participant_id': participant_id,
        'label': label,
    }


def single_experiment(args, convos, prompt, pred_df):
    responses = []
    num_tokens = []
    completion_tokens = []
    times = []
    finish_reasons = []
    total_calls = 0
    failed_calls = 0
    errors = []
    for convo_id, convo in convos:
        complete_prompt = prompt + '\n' + convo
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": complete_prompt
        }]
        num_tokens_ = num_tokens_from_messages(messages, args.model)
        print(f"Number of tokens: {num_tokens_}")
        num_tokens.append(num_tokens_)
        response = None
        counter = 0
        # give it up to 5 retries
        while (response is None) and (counter < 5):
            try:
                start = time.perf_counter()
                # openai.error.APIError: Bad gateway. {"error":{"code":502,"message":"Bad gateway.","param":null,"type":"cf_bad_gateway"}} 502 {'error': {'code': 502, 'message': 'Bad gateway.', 'param': None, 'type': 'cf_bad_gateway'}} {'Date': 'Tue, 23 May 2023 03:13:41 GMT', 'Content-Type': 'application/json', 'Content-Length': '84', 'Connection': 'keep-alive', 'X-Frame-Options': 'SAMEORIGIN', 'Referrer-Policy': 'same-origin', 'Cache-Control': 'private, max-age=0, no-store, no-cache, must-revalidate, post-check=0, pre-check=0', 'Expires': 'Thu, 01 Jan 1970 00:00:01 GMT', 'Server': 'cloudflare', 'CF-RAY': '7cba17bec9134392-EWR', 'alt-svc': 'h3=":443"; ma=86400, h3-29=":443"; ma=86400'}
                # openai.error.RateLimitError: That model is currently overloaded with other requests. You can retry your request, or contact us through our help center at help.openai.com if the error persists. (Please include the request ID d536953e12f480a47c9a282f34fc39da in your message.)
                response = openai.ChatCompletion.create(
                    messages=messages,
                    model=args.model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                end = time.perf_counter()
                total_calls += 1
            except Exception as e:
                print(e)
                counter += 1
                total_calls += 1
                failed_calls += 1
                errors.append((response, e))
                response = None
                print(
                    f'Failed to get response for conversation {convo_id}, retrying ({counter}/5)'
                )
                continue

        if response is not None:
            # record metadata
            print(f"Finish reason: {response['choices'][0]['finish_reason']}")
            print(f"Time: {end - start:0.2f}")
            print(response['choices'][0]['message']['content'])
            completion_tokens.append(response['usage']['completion_tokens'])
            times.append(end - start)
            finish_reasons.append(response['choices'][0]['finish_reason'])

            # parse response
            response_content = response['choices'][0]['message']['content']
            rows = [parse_row(x) for x in response_content.split('\n')[2:]]
            response_df = pd.DataFrame(rows)
            response_df['conversation_id'] = convo_id
            responses.append(response_df)
        else:
            print(f'Failed to get response for conversation {convo_id}')

    if len(responses) == 0:
        response_df = pd.DataFrame(columns=[
            'utterance_id', 'participant_id', 'label', 'conversation_id'
        ])
    else:
        response_df = pd.concat(responses)
    # join with pred_df on conversation_id and utterance_id
    # copy this dataframe so that we don't modify the original
    copy_df = pred_df.copy(deep=True)
    merge_df = pd.merge(copy_df,
                        response_df,
                        on=['conversation_id', 'utterance_id'],
                        how='left')
    # compute accuracy
    merge_df['correct'] = merge_df.apply(
        lambda x: x['label'] == x['social_orientation'], axis=1)
    accuracy = merge_df['correct'].mean()
    print(f"Accuracy: {accuracy}")
    experiment_result = {
        'accuracy': accuracy,
        'model': args.model,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'num_tokens': num_tokens,
        'completion_tokens': completion_tokens,
        'finish_reasons': finish_reasons,
        'times': times,
        'total_calls': total_calls,
        'failed_calls': failed_calls,
        'failure_rate': failed_calls / total_calls
    }
    return experiment_result, merge_df, errors


def sweep_experiment(args, convos, prompt, pred_df):
    """Sweep over a range of temperatures and top_p values."""
    temperatures = [0.0, 0.2, 0.4, 0.6, 0.8]
    models = ['gpt-3.5-turbo', 'gpt-4']
    results = []
    pred_dfs = []
    all_errors = []
    for temperature in temperatures:
        for model in models:
            print(
                f'Running experiment with temperature {temperature} and model {model}...'
            )
            args.temperature = temperature
            args.model = model
            experiment_result, exp_df, errors = single_experiment(
                args, convos, prompt, pred_df)
            exp_df['temperature'] = temperature
            exp_df['model'] = model
            results.append(experiment_result)
            pred_dfs.append(exp_df)
            all_errors.append({
                'temperature': temperature,
                'model': model,
                'errors': errors
            })
    return results, pred_dfs, all_errors


def main(args):
    prompt = load_prompt(args.prompt)
    if args.experiment != 'analyze':
        os.makedirs(args.output_dir, exist_ok=True)
    anno_filepath = '/home/iron-man/Documents/data/convokit/conversations-gone-awry-corpus/sample_conversations_anno.csv'
    df = pd.read_csv(anno_filepath)
    # '345292304.3937.3937',
    prompt_convo_ids = [
        '368040175.20376.20376', '536876507.8124.8124', '68434244.25865.25865',
        '820812359.51399.51399'
    ]
    pred_df = df[df['conversation_id'].apply(
        lambda x: x not in prompt_convo_ids)]
    # filter out any empty rows
    pred_df = pred_df[pred_df['text'].notna()]
    # filter out section headers now for ease, address this later
    pred_df = pred_df[~pred_df['meta.is_section_header']]
    pred_df = pred_df.groupby('conversation_id',
                              group_keys=False).apply(add_utterance_id)
    pred_df['gpt_line'] = pred_df.apply(create_line, axis=1)
    convos = list(
        pred_df.groupby('conversation_id')['gpt_line'].apply(
            lambda x: '\n'.join(list(x))).items())

    now = datetime.now().strftime("%m%d%y_%H%M%S")
    # single experiment
    if args.experiment == 'single':
        exp_result, exp_df, errors = single_experiment(args, convos, prompt,
                                                       pred_df)
        output_dir = os.path.join(
            args.output_dir,
            f'{args.experiment}_{args.model}_{args.temperature}_{args.top_p}_{now}'
        )
        os.makedirs(output_dir, exist_ok=True)
        exp_result_df = pd.DataFrame([exp_result])
        exp_result_df.to_csv(os.path.join(output_dir, 'results.csv'),
                             index=False)
        exp_df.to_csv(os.path.join(output_dir, 'preds.csv'), index=False)
    elif args.experiment == 'sweep':
        exp_results, exp_dfs, errors = sweep_experiment(
            args, convos, prompt, pred_df)
        output_dir = os.path.join(args.output_dir, f'{args.experiment}_{now}')
        os.makedirs(output_dir, exist_ok=True)
        exp_result_df = pd.DataFrame(exp_results)
        exp_df = pd.concat(exp_dfs)
        exp_result_df.to_csv(os.path.join(output_dir, 'results.csv'),
                             index=False)
        exp_df.to_csv(os.path.join(output_dir, 'preds.csv'), index=False)
    elif args.experiment == 'analyze':
        # load the data stored in args.output_dir
        results_filepath = os.path.join(args.output_dir, 'results.csv')
        preds_filepath = os.path.join(args.output_dir, 'preds.csv')
        results_df = pd.read_csv(results_filepath)
        preds_df = pd.read_csv(preds_filepath)
        # plot accuracy vs. temperature for each model
        models = ['gpt-3.5-turbo', 'gpt-4']
        for model in models:
            model_results_df = results_df[results_df['model'] == model]
            plt.plot(model_results_df['temperature'],
                     model_results_df['accuracy'],
                     label=model)
        plt.xlabel('Temperature')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir,
                                 'accuracy_vs_temperature.png'),
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()

        # plot confusion matrix for each model at the best temperature
        for i, model in enumerate(models):
            model_results_df = results_df[results_df['model'] == model]
            best_temperature = model_results_df['temperature'].loc[
                model_results_df['accuracy'].idxmax()]
            print(f"Best temperature for model {model}: {best_temperature}")
            best_preds_df = preds_df[(preds_df['model'] == model) & (
                preds_df['temperature'] == best_temperature)]
            cm = confusion_matrix(best_preds_df['social_orientation'],
                                  best_preds_df['label'],
                                  labels=SOCIAL_ORIENTATION_LABEL2ID.keys())
            cm_display = ConfusionMatrixDisplay(
                cm, display_labels=SOCIAL_ORIENTATION_LABEL2ID.keys()).plot()
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.title(
                f'Confusion Matrix for {model} at Temperature {best_temperature}'
            )
            plt.savefig(os.path.join(args.output_dir,
                                     f'confusion_matrix_{model}.png'),
                        dpi=300,
                        bbox_inches='tight')
            plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type=str,
        default='single',
        choices=['single', 'sweep', 'analyze'],
        help=
        'If single, use specified arguments, if sweep, search over a variety of configurations in search of the best accuracy.'
    )
    parser.add_argument('--output-dir',
                        type=str,
                        default='./logs/prompt-results')
    parser.add_argument('--prompt', type=str, default='prompt.txt')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1)
    args = parser.parse_args()
    main(args)