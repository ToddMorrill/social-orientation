# Social Orientation: A New Feature for Dialogue Analysis
This repository implements the experiments presented in the paper, Social Orientation: A New Feature for Dialogue Analysis, which was accepted to LREC-COLING 2024.

**Authors:** Todd Morrill, Zhaoyuan Deng, Yanda Chen, Amith Ananthram, Colin Wayne Leach, Kathleen McKeown

**arXiv link:** [https://arxiv.org/abs/2403.04770](https://arxiv.org/abs/2403.04770)

**TLDR;** There are many settings where it is useful to predict and explain the success or failure of a dialogue. Circumplex theory from psychology models the social orientations (e.g., Warm-Agreeable, Arrogant-Calculating) of conversation participants, which can in turn can be used to predict and explain the outcome of social interactions, such as in online debates over Wikipedia page edits or on the Reddit ChangeMyView forum.

[![Figure 1](assets/circumplex-figure1.png)](https://arxiv.org/abs/2403.04770)

## Environment setup
With a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
With Conda:
```bash
conda create -n soc-orientation python=3.12 -y
conda activate soc-orientation
pip install -r requirements.txt
```

## Collect social orientation tags from GPT (not required - code provided for reference)
The following command will generate prompts to collect social orientation tags for the [Conversations Gone Awry (CGA)](https://convokit.cornell.edu/documentation/awry.html) dataset using OpenAI GPT models. Prompts can must be generated for the training, validation, and test set CGA conversations. Many conversations will fit within the token limit for GPT-4 (as of 06/2023), though some conversations are chunked into multiple API calls, which we denote as train-long, val-long, and test-long.
```bash
python gpt_prompts.py \
    --gpt-mode train \
    --gpt-model gpt-4 \
    --gpt-data-dir data/gpt-4-input \
    --calculate-cost
```
Next, we send data to the GPT API using `api_request_parallel_processor.py`. See the module docstring for usage details. Again, we need to run this for all splits of the data in {train, val, test, train-long, val-long, test-long}.
```bash
python api_request_parallel_processor.py \
    --requests_filepath data/gpt-4-input/train-long.jsonl \
    --save_filepath data/gpt-4-output/train-long_results_gpt4.jsonl \
    --request_url https://api.openai.com/v1/chat/completions \
    --max_requests_per_minute 175 \
    --max_tokens_per_minute 35000 \
    --model gpt-4-0314 \
    --temperature 0.4 \
    --top_p 1.0 \
    --token_encoding_name gpt-4-0314 \
    --max_attempts 10 \
    --logging_level 20 \
```

## Parse GPT API responses (required)
Finally, we parse the GPT-4 API response using `parse.py`, where we specify which GPT response files to parse. This outputs CSV files, where each row is a social orientation label for an utterance in a specific conversation.
```bash
python parse.py \
    --gpt-outputs-filepath \
        data/gpt-4-output/train_results_gpt4.jsonl \
        data/gpt-4-output/train-long_results_gpt4.jsonl \
        data/gpt-4-output/val_results_gpt4.jsonl \
        data/gpt-4-output/val-long_results_gpt4.jsonl \
        data/gpt-4-output/test_results_gpt4.jsonl \
        data/gpt-4-output/test-long_results_gpt4.jsonl \
    --parse-output-dir \
        data/gpt-4-cga-social-orientation-labels
```

## Train a social orientation tagger using GPT-4 labels
Train and predict social orientation tags on the CGA dataset using GPT-4 labels:
```bash
CUDA_VISIBLE_DEVICES=0 \
nohup \
python -m train \
    --dataset social-orientation \
    --model-name-or-path distilbert-base-uncased \
    --batch-size 32 \
    --lr 1e-6 \
    --val-steps 50 \
    --eval-test \
    --early-stopping-patience 10 \
    --include-speakers \
    --social-orientation-filepaths \
        data/gpt-4-cga-social-orientation-labels/train_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/val_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/train-long_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/val-long_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/test_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/test-long_results_gpt4_parsed.csv \
    --fp16 \
    --window-size 2 \
    --save-steps 50 \
    --num-checkpoints 2 \
    --model-dir model/distilbert-social-winsize-2 \
    --weighted-loss \
> logs/runs/distilbert-social-winsize-2.log 2>&1 &
```

<!-- Alternatively, train with the HuggingFace trainer:
```bash
CUDA_VISIBLE_DEVICES=0 \
nohup \
python -m train \
    --dataset social-orientation \
    --model-name-or-path distilbert-base-uncased \
    --batch-size 32 \
    --lr 1e-6 \
    --val-steps 50 \
    --eval-test \
    --early-stopping-patience 10 \
    --include-speakers \
    --social-orientation-filepaths \
        data/gpt-4-cga-social-orientation-labels/train_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/val_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/train-long_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/val-long_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/test_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/test-long_results_gpt4_parsed.csv \
    --fp16 \
    --window-size 2 \
    --save-steps 50 \
    --num-checkpoints 2 \
    --model-dir model/distilbert-social-winsize-2-hf \
    --weighted-loss \
    --trainer hf \
> logs/runs/distilbert-social-winsize-2-hf.log 2>&1 &
``` -->

Predict social orientation tags on the CGA dataset using the trained model. You can include the GPT-4 social orientation filepaths to evaluate the model. 
```bash
python predict.py \
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
```

## Run data ablation experiments, varying the dataset size
The following experiment trains CGA models as we vary the size of the training dataset. It also compares the inclusion and exclusion of social orientation tags, both predicted, and from GPT-4.
```bash
nohup \
python -m train \
    --dataset cga \
    --model-name-or-path distilbert-base-uncased \
    --batch-size 32 \
    --lr 5e-6 \
    --val-steps 50 \
    --early-stopping-patience 10 \
    --include-speakers \
    --include-social-orientation \
    --social-orientation-filepaths \
        data/gpt-4-cga-social-orientation-labels/train_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/val_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/train-long_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/val-long_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/test_results_gpt4_parsed.csv \
        data/gpt-4-cga-social-orientation-labels/test-long_results_gpt4_parsed.csv \
    --predicted-social-orientation-filepaths \
        data/predictions/social-orientation-social/distilbert-base-uncased/train_winsize_2_model_distilbert-base-uncased.csv \
        data/predictions/social-orientation-social/distilbert-base-uncased/val_winsize_2_model_distilbert-base-uncased.csv \
        data/predictions/social-orientation-social/distilbert-base-uncased/test_winsize_2_model_distilbert-base-uncased.csv \
    --fp16 \
    --add-tokens \
    --window-size all \
    --use-multiprocessing \
    --num-runs 5 \
    --experiment subset \
    --subset-pcts 0.01 0.1 0.2 0.5 1.0 \
    --use-cache \
    --seed 42 \
    --jobs-per-gpu 3 \
    --eval-test \
> logs/runs/subset-cga.log 2>&1 &
```

Run experiments with logistic regression model.
```bash
python classic.py \
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
```

Plot results:
```
python analyze.py \
    --analysis-mode data-ablation \
    --experiment-filepath \
        logs/experiments/subset_cga_distilbert-base-uncased.csv \
        logs/experiments/subset_cga_logistic_clf.csv
```

Run t-test:
```
python analyze.py \
    --analysis-mode t-test \
    --experiment subset \
    --experiment-filepath \
        logs/experiments/subset_cga_distilbert-base-uncased.csv \
        logs/experiments/subset_cga_logistic_clf.csv
```

## Run data ablation experiment for CGA CMV subset
First make social orientation predictions.
```
python predict.py \
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
```
Then train CGA CMV models with predicted social orientation tags, varying the subset size.
```
nohup \
python -m train \
    --dataset cga-cmv \
    --data-dir ~/Documents/data/convokit/conversations-gone-awry-cmv-corpus \
    --model-name-or-path distilbert-base-uncased \
    --batch-size 32 \
    --lr 5e-6 \
    --val-steps 50 \
    --early-stopping-patience 10 \
    --include-speakers \
    --include-social-orientation \
    --social-orientation-filepaths \
        predictions/cga-cmv-social/distilbert-base-uncased/train_winsize_2_model_distilbert-base-uncased.csv \
        predictions/cga-cmv-social/distilbert-base-uncased/val_winsize_2_model_distilbert-base-uncased.csv \
        predictions/cga-cmv-social/distilbert-base-uncased/test_winsize_2_model_distilbert-base-uncased.csv \
    --fp16 \
    --add-tokens \
    --window-size all \
    --use-multiprocessing \
    --num-runs 5 \
    --experiment subset \
    --subset-pcts 0.01 0.1 0.2 0.5 1.0 \
    --use-cache \
    --jobs-per-gpu 4 \
    --eval-test \
> logs/runs/subset-cga-cmv.log 2>&1 &
```
Or run CGA & CGA CMV experiments together:
```
nohup ./subset.sh > logs/runs/subset-cga-cga-cmv.log 2>&1 &
```

Run experiments with logistic regression model.
```
python classic.py \
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
```

Plot results:
```
python analyze.py \
    --analysis-mode data-ablation \
    --dataset cga-cmv \
    --experiment-filepath \
        logs/experiments/subset_cga-cmv_distilbert-base-uncased.csv \
        logs/experiments/subset_cga-cmv_logistic_clf.csv
```

Run t-test:
```
python analyze.py \
    --analysis-mode t-test \
    --experiment subset \
    --dataset cga-cmv \
    --data-dir ~/Documents/data/convokit/conversations-gone-awry-cmv-corpus \
    --experiment-filepath \
        logs/experiments/subset_cga-cmv_distilbert-base-uncased.csv \
        logs/experiments/subset_cga-cmv_logistic_clf.csv
```

## Train a single CGA CMV model with predicted social orientation tags
```bash
nohup \
python -u -m train \
    --dataset cga-cmv \
    --data-dir ~/Documents/data/convokit/conversations-gone-awry-cmv-corpus \
    --model-name-or-path distilbert-base-uncased \
    --batch-size 32 \
    --lr 1e-6 \
    --val-steps 50 \
    --early-stopping-patience 5 \
    --include-speakers \
    --include-social-orientation \
    --social-orientation-filepaths \
        predictions/cga-cmv-social/distilbert-base-uncased/train_winsize_2_model_distilbert-base-uncased.csv \
        predictions/cga-cmv-social/distilbert-base-uncased/val_winsize_2_model_distilbert-base-uncased.csv \
        predictions/cga-cmv-social/distilbert-base-uncased/test_winsize_2_model_distilbert-base-uncased.csv \
    --fp16 \
    --add-tokens \
    --window-size all \
    --eval-test \
    --save-steps 50 \
    --num-checkpoints 2 \
    --model-dir ./model/distilbert-cga-cmv-distilbert-winsize-2 \
> logs/runs/distilbert-cga-cmv-distilbert-winsize-2.log 2>&1 &
```

## Train a social orientation tagger using XLMR
```CUDA_VISIBLE_DEVICES=1 nohup \
python -m train \
    --dataset social-orientation \
    --model-name-or-path xlm-roberta-base \
    --batch-size 32 \
    --lr 1e-6 \
    --val-steps 50 \
    --early-stopping-patience 10 \
    --include-speakers \
    --social-orientation-filepaths \
        ~/Documents/data/circumplex/transformed/train_results_gpt4_parsed.csv \
        ~/Documents/data/circumplex/transformed/val_results_gpt4_parsed.csv \
        ~/Documents/data/circumplex/transformed/train-long_results_gpt4_parsed.csv \
        ~/Documents/data/circumplex/transformed/val-long_results_gpt4_parsed.csv \
        ~/Documents/data/circumplex/transformed/test_results_gpt4_parsed.csv \
        ~/Documents/data/circumplex/transformed/test-long_results_gpt4_parsed.csv \
    --fp16 \
    --add-tokens \
    --window-size 2 \
    --save-steps 50 \
    --num-checkpoints 2 \
    --model-dir ./model/xlmr-social-winsize-2-weighted \
    --eval-test \
    --weighted-loss \
> logs/runs/xlmr-social-winsize-2.log 2>&1 &
```