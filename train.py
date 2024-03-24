"""Training pipeline for all models.

Examples:
    $ python -m train \
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
        --weighted-loss
"""
import json
import logging
import torch.multiprocessing as mp
import signal
import os

import gpustat
from evaluate import load
import numpy as np
import torch
from torch import nn
import transformers
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import pandas as pd

from args import parse_args
from data import get_labels, get_tokenizer, get_data_loaders, SOCIAL_ORIENTATION_LABEL2ID, SOCIAL_ORIENTATION_ID2LABEL, CGA_LABEL2ID, CGA_ID2LABEL
from trainer import Trainer, CustomHFTrainer
from utils import log, dist_cleanup, set_random_seed #shutdown_handler
from callbacks import ModelSaver, EarlyStopping, Accuracy

transformers.logging.set_verbosity_warning()
pd.options.mode.chained_assignment = None

# signal.signal(signal.SIGINT, shutdown_handler)


def compute_metrics(eval_pred):
    acc = load('accuracy')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return acc.compute(predictions=predictions, references=labels)


def get_model(args, label2id=None, id2label=None, tokenizer=None, tokens2ids=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # if specified, add new tokens to embedding layer
    if tokens2ids is not None and bool(tokens2ids):
        logging.info(f'Adding {len(tokens2ids)} new tokens to embedding layer')
        embedding_tensor = model.get_input_embeddings().weight.data
        tokens_init = []
        for token, idxs in tokens2ids.items():
            tokens_init.append(embedding_tensor[idxs].mean(dim=0))
        
        # NB: this is forcing the embedding layer to be trainable
        # if you only want to train the new tokens, you can manually zero out
        # the gradients for the rest of the embedding layer during after the
        # backward pass
        model.resize_token_embeddings(len(tokenizer))

        new_embeddings = torch.stack(tokens_init)
        model.get_input_embeddings(
        ).weight.data[-len(tokens_init):] = new_embeddings
        # assert equal
        assert torch.allclose(
            model.get_input_embeddings().weight.data[-len(tokens_init):],
            new_embeddings)

    logging.info(
        f'{args.model_name_or_path} parameter count: {model.num_parameters():,}'
    )
    return model


def pipeline(args):
    log('Starting pipeline..')
    label2id, id2label = get_labels(args)
    if args.add_tokens:
        # add social orientation labels to tokenizer
        tokenizer, tokens2ids = get_tokenizer(args, SOCIAL_ORIENTATION_LABEL2ID.keys())
    else:
        tokenizer, tokens2ids = get_tokenizer(args)
    model = get_model(args, label2id, id2label, tokenizer, tokens2ids)

    # load data
    train_loader, val_loader, test_loader = get_data_loaders(args, tokenizer)

    # if args.weighted_loss, then we need to compute the class weights
    custom_loss_fn = None
    if args.weighted_loss:
        logging.info('Computing class weights..')
        # get the number of samples in each class from the training set
        all_labels = []
        for batch in train_loader:
            labels = batch['labels']
            all_labels.extend(labels)
        all_labels = torch.tensor(all_labels)
        # count the number of samples in each class
        class_counts = torch.bincount(all_labels)
        # compute the class weights
        class_weights = class_counts.sum() / class_counts.float()
        # normalize the class weights so that they sum to 1 and then multiply by the number of classes
        class_weights = (class_weights / class_weights.sum()) * len(class_counts)
        # TODO: implement power law smoothing
        # by doing this, the average weight applied to each sample will be 1, which is the same as not using class weights
        # so we don't have to worry that we're changing the scale of the loss on average
        # pretty print the class weights with the corresponding labels
        class_weights_pretty = {id2label[i]: weight.item() for i, weight in enumerate(class_weights)}
        class_weights_pretty = json.dumps(class_weights_pretty, indent=4)
        logging.info(f'Class weights:\n{class_weights_pretty}')
        custom_loss_fn = weighted_loss_fn(class_weights, num_labels=len(class_counts))


    # get number of training steps for learning rate scheduler
    optimization_steps_per_epoch = len(train_loader) / args.gradient_accumulation_steps
    args.num_train_steps = optimization_steps_per_epoch * args.epochs
    
    if args.scale_lr:
        # adjust learning rate as a function of number of devices and batch size
        lr = args.lr
        effective_batch_size = args.batch_size
        if args.distributed:
            effective_batch_size *= args.world_size
        # assume batch size of 32 is the base case and apply a linear scaling rule
        lr = lr * (effective_batch_size / 32)
        args.lr = lr

    if args.trainer == 'torch':
        early_stopping = EarlyStopping(patience=args.early_stopping_patience,
                                       monitor=args.monitored_metric,
                                       eval_every_n_steps=args.val_steps,
                                       minimize=(not args.maximize_metric))
        # model saving should be done after early stopping
        # TODO: add a check that will move modelsaver to end of callbacks
        model_saver = ModelSaver(args.model_dir,
                                 save_every_n_steps=args.save_steps,
                                 num_checkpoints=args.num_checkpoints,
                                 monitor=args.monitored_metric,
                                 minimize=(not args.maximize_metric))
        callbacks = [early_stopping, model_saver]
        accuracy = Accuracy()
        metrics = [accuracy]

        passed_test_loader = test_loader if args.eval_test else None
        # train model
        trainer = Trainer(args=args,
                          model=model,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          test_loader=passed_test_loader,
                          callbacks=callbacks,
                          metrics=metrics,
                          custom_loss_fn=custom_loss_fn,
                          tokenizer=tokenizer,)
        trainer.train()
        
        # TODO: refactor this into a new function
        # if load_best_model_at_end is True, and make_predictions is True, then
        # disable shuffling of the training data and predict on the train and 
        # val sets with the best model
        if args.load_best_model_at_end and args.make_predictions:
            args.disable_train_shuffle = True
            train_loader, val_loader, test_loader = get_data_loaders(args, tokenizer)
            train_preds = trainer.predict(train_loader)
            args.disable_train_shuffle = False
            val_preds = trainer.predict(val_loader)
            test_preds = trainer.predict(test_loader)

            # argmax predictions and decode
            train_preds = np.argmax(train_preds, axis=1)
            val_preds = np.argmax(val_preds, axis=1)
            test_preds = np.argmax(test_preds, axis=1)

            if args.dataset == 'social-orientation':
                # merge with dfs and save predictions
                train_df = train_loader.dataset.df
                val_df = val_loader.dataset.df
                test_df = test_loader.dataset.df
                train_df['prediction'] = train_preds
                val_df['prediction'] = val_preds
                test_df['prediction'] = test_preds
            else:
                # if the dataset is not social-orientation, then we're predicting at the conversation level
                # so we need to merge predictions with conversation_ids and then merge with the original df
                train_preds = pd.DataFrame(train_preds, columns=['prediction'])
                val_preds = pd.DataFrame(val_preds, columns=['prediction'])
                test_preds = pd.DataFrame(test_preds, columns=['prediction'])
                train_preds['conversation_id'] = train_loader.dataset.convo_ids
                val_preds['conversation_id'] = val_loader.dataset.convo_ids
                test_preds['conversation_id'] = test_loader.dataset.convo_ids
                train_df = pd.merge(train_loader.dataset.df, train_preds, on='conversation_id', how='left')
                val_df = pd.merge(val_loader.dataset.df, val_preds, on='conversation_id', how='left')
                test_df = pd.merge(test_loader.dataset.df, test_preds, on='conversation_id', how='left')
                # sanity check val accuracy
                ground_truth = val_df.groupby('conversation_id')['meta.comment_has_personal_attack'].max().astype(int)
                acc = (ground_truth.values == val_preds['prediction'].values).sum() / len(ground_truth)
                logging.info(f'Validation set accuracy: {acc:.4f}')

            # decode predictions
            train_df['prediction'] = train_df['prediction'].map(id2label)
            val_df['prediction'] = val_df['prediction'].map(id2label)
            test_df['prediction'] = test_df['prediction'].map(id2label)

            filename_suffix = f'_{args.dataset}_{args.window_size}_{args.model_name_or_path}.csv'
            # if prediction_dir present in args, save predictions there, othwerwise, default to model_dir
            output_dir = args.prediction_dir if args.prediction_dir is not None else args.model_dir
            os.makedirs(output_dir, exist_ok=True)
            train_df.to_csv(os.path.join(output_dir, f'train_preds{filename_suffix}'))
            val_df.to_csv(os.path.join(output_dir, f'val_preds{filename_suffix}'))
            test_df.to_csv(os.path.join(output_dir, f'test_preds{filename_suffix}'))

    elif args.trainer == 'hf':
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience, )
        # train with huggingface trainer
        training_args = TrainingArguments(
            output_dir=args.model_dir,
            report_to='none',
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            learning_rate=args.lr,
            lr_scheduler_type='linear',
            warmup_steps=args.num_warmup_steps,
            evaluation_strategy='steps',
            eval_steps=args.val_steps,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            logging_steps=args.reporting_steps,
            dataloader_num_workers=args.num_dataloader_workers,
            num_train_epochs=args.epochs,
            save_total_limit=args.num_checkpoints,
            load_best_model_at_end=True,
            save_strategy='steps',
            save_steps=args.save_steps,
            seed=args.seed,
        )
        # pass class_weights to the trainer
        training_args.weighted_loss = args.weighted_loss
        if args.weighted_loss:
            training_args.class_weights = class_weights
        trainer = CustomHFTrainer(
            model=model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            data_collator=train_loader.collate_fn,
            compute_metrics=compute_metrics,
            tokenizer=train_loader.dataset.tokenizer,
            callbacks=[early_stopping])
        trainer.train()

    return trainer


def load_cache(args, filepath):
    # if we're using multiprocessing, we have to load the cache, otherwise,
    # we will clobber the cache.
    # so in order to overwrite the cache, we have to delete it first
    if (args.use_cache or args.use_multiprocessing) and os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # if social_orientation_filepaths is present and is a string, then eval it to get a list
        # this ensures cache checks work as expected
        if 'social_orientation_filepaths' in df.columns:
            df['social_orientation_filepaths'] = df['social_orientation_filepaths'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        return df    
    return pd.DataFrame()

def get_best_val_loss_acc(trainer):
    """Retrieves the best validation loss and corresponding accuracy achieved
    during training. This also retrieves the best test loss and corresponding
    accuracy achieved, if available."""
    results = {}
    # get the index corresponding to the minimum validation loss
    # achieved
    # the following assumes trainer.metrics is present (may not be true for hf trainer)
    val_loss = trainer.metrics['val_loss']['values']
    val_loss = np.array(val_loss)
    min_val_loss_idx = np.argmin(val_loss)
    min_val_loss = val_loss[min_val_loss_idx]
    results['min_val_loss'] = min_val_loss

    # get the maximum validation accuracy achieved
    val_accuracy = trainer.metrics['val_accuracy']['values']
    val_accuracy = np.array(val_accuracy)
    max_val_acc = val_accuracy[min_val_loss_idx]
    results['max_val_acc'] = max_val_acc

    # if test loss is present, get the corresponding test accuracy
    if 'test_loss' in trainer.metrics:
        results['min_test_loss'] = trainer.metrics['test_loss']['values'][min_val_loss_idx]
        results['max_test_acc'] = trainer.metrics['test_accuracy']['values'][min_val_loss_idx]
    return results


def get_free_gpus(gpu_memory_threshold=26000):
    stats = gpustat.GPUStatCollection.new_query()
    free_gpus = []
    for gpu in stats:
        # e.g. less than 30000MB (30GB) used --> double stacking sometimes
        # useful on A100s with large memory capacities
        if gpu.memory_used < gpu_memory_threshold:
            free_gpus.append(gpu.index)
    return free_gpus

def multiprocess_worker(q, lock, device_id, args):
    # specify the device to use
    args.device = torch.device(f'cuda:{device_id}')
    
    # pull from queue until the sentinel value is reached
    while True:
        configuration = q.get()
        # if sentinel value is reached, break
        if configuration is None:
            break
        
        # otherwise run the pipeline
        # unpack the configuration and default to the args value if the configuration value is None
        seed = configuration.get('seed', args.seed)
        subset_pct = configuration.get('subset_pct', args.subset_pct)
        window_size = configuration.get('window_size', args.window_size)
        include = configuration.get('include_social_orientation', args.include_social_orientation)
        social_orientation_filepaths = configuration.get('social_orientation_filepaths', args.social_orientation_filepaths)
        
        # set the seed
        args.seed = seed
        set_random_seed(args.seed)
        args.subset_pct = subset_pct
        # if subset_pct is really small, then we need to evaluate more frequently
        # otherwise, we won't record any validation metrics
        if subset_pct <= 0.2:
            args.val_steps = 10
            args.reporting_steps = 10
            args.early_stopping_patience = 10
            args.lr = 5e-7
        args.window_size = window_size
        args.include_social_orientation = include
        args.social_orientation_filepaths = social_orientation_filepaths

        if window_size is None:
            # args.window_size will be passed to the pipeline and interpreted as 'all'
            # but we want to save the window size as 'all' in the cache
            window_size = 'all'
        
        log(f'Running pipeline with subset size: {args.subset_pct}, window size: {window_size}, social orientation: {args.include_social_orientation}, seed: {args.seed}, and social orientation filepaths: {args.social_orientation_filepaths}')

        # acquire lock and check if the configuration has already been run
        with lock:
            df = load_cache(args, args.cache_filepath)

            if args.use_cache and len(df) > 0:
                # checking social_orientation_filepaths is not so trivial because args.social_orientation_filepaths may be None and when we load the cache, it will be an np.nan
                # so we have to check if the value is None or np.nan
                if args.social_orientation_filepaths is None:
                    social_orientation_filepaths_check = df['social_orientation_filepaths'].isna()
                else:
                    social_orientation_filepaths_check = df['social_orientation_filepaths'].apply(lambda x: x == args.social_orientation_filepaths)
                df_subset = df[(df['window_size'] == window_size) & (df['social_orientation'] == args.include_social_orientation) & (df['seed'] == args.seed) & (df['subset_pct'] == args.subset_pct) & (social_orientation_filepaths_check)]
                if len(df_subset) > 0:
                    log(f'Subset size: {args.subset_pct}, window size: {window_size}, social orientation: {args.include_social_orientation}, seed: {args.seed}, and social orientation filepaths: {args.social_orientation_filepaths} already ran. Skipping.')
                    continue
        
        # call the training pipeline
        # TODO: address logging issue
        trainer = pipeline(args)

        # get best val and test metrics
        results = get_best_val_loss_acc(trainer)
        min_val_loss, max_val_acc = results['min_val_loss'], results['max_val_acc']
        min_test_loss, max_test_acc = results.get('min_test_loss', np.nan), results.get('max_test_acc', np.nan)
        
        # save results to the cache
        # again acquire lock to ensure no other process is writing to the cache
        with lock:
            temp_df = pd.DataFrame([{'window_size': window_size, 'social_orientation': args.include_social_orientation, 'seed': args.seed, 'subset_pct': args.subset_pct, 'social_orientation_filepaths': args.social_orientation_filepaths, 'val_acc': max_val_acc, 'val_loss': min_val_loss, 'test_acc': max_test_acc, 'test_loss': min_test_loss}])
            df = load_cache(args, args.cache_filepath)
            df = pd.concat([df, temp_df], ignore_index=True)
            df.to_csv(args.cache_filepath, index=False)
            log(f'Subset size: {args.subset_pct}, window size: {window_size}, social orientation: {args.include_social_orientation}, seed: {args.seed}, and social orientation filepaths: {args.social_orientation_filepaths}, val_loss: {min_val_loss:.4f}, val_accuracy: {max_val_acc:.4f}.')
        
        # delete the trainer to clear all memory references
        # probably not necessary since everything is going out of scope
        del trainer
    return None

def subset_experiment(args):
    log('Starting subset experiment.')
    os.makedirs(args.exp_dir, exist_ok=True)
    model_name = args.model_name_or_path.replace('/', '_')
    cache_filepath = os.path.join(args.exp_dir, f'subset_{args.dataset}_{model_name}.csv')
    args.cache_filepath = cache_filepath

    subset_pcts = [0.01, 0.1, 0.2, 0.5, 1.0]
    if args.subset_pcts is not None:
        subset_pcts = args.subset_pcts
    include_social_orientation = [False, True]
    if args.dont_compare_social_orientation:
        include_social_orientation = [args.include_social_orientation]
    social_orientation_sets = []
    if True in include_social_orientation and args.social_orientation_filepaths is not None:
        # if we're running the pipeline with social orientation features included, and we've passed filepaths in
        social_orientation_sets.append(args.social_orientation_filepaths)
    if True in include_social_orientation and args.predicted_social_orientation_filepaths is not None:
        # if we're running the pipeline with social orientation features included, and we've passed additional filepaths in
        social_orientation_sets.append(args.predicted_social_orientation_filepaths)

    # generate all possible configurations
    jobs = []
    original_seed = args.seed
    for offset in range(args.num_runs):
        seed = original_seed + offset
        for subset in subset_pcts:
            for include in include_social_orientation:
                if include == False: # i.e. include == False
                    # if social orientation is not included, then we don't need to worry about the social orientation filepaths
                    jobs.append({'seed': seed, 'subset_pct': subset, 'include_social_orientation': include, 'social_orientation_filepaths': None})
                    continue
                for social_orientation_set in social_orientation_sets:
                    jobs.append({'seed': seed, 'subset_pct': subset, 'include_social_orientation': include, 'social_orientation_filepaths': social_orientation_set})
    q = mp.Queue()
    for job in jobs:
        q.put(job)

    # create a lock to prevent multiple processes from reading/writing to the cache file at once
    lock = mp.Lock()

    # determine the number of GPUs available, which can be determined by checking how many GPUs have used < 300 MB of memory
    if args.jobs_per_gpu == 1:
        free_gpus = get_free_gpus()
    else:
        # if we're using multiple jobs per GPU, then we need to check how many GPUs have used < 300 MB of memory
        # and then multiply that number by the number of jobs per GPU
        free_gpus = get_free_gpus()
        free_gpus = free_gpus * args.jobs_per_gpu
    
    # put sentinel values into the queue
    if args.use_multiprocessing:
        for _ in range(len(free_gpus)):
            q.put(None)
    else:
        # only need one sentinel value if not using multiprocessing
        q.put(None)

    # start len(free_gpus) processes and let each pull from the queue until the sentinel value is reached
    if args.use_multiprocessing:
        processes = []
        for device_id in free_gpus:
            if args.use_multiprocessing:
                p = mp.multiprocessing.Process(target=multiprocess_worker, args=(q, lock, device_id, args))
                p.start()
                processes.append(p)
        # wait for all processes to finish
        for p in processes:
            p.join()
    else:
        # allow a single worker to run all jobs in the queue
        # NB: assuming GPUs are available
        multiprocess_worker(q, lock, free_gpus[0], args)
    return None

def weighted_loss_fn(class_weights, num_labels):
    def loss_fn(logits, labels):
        weights = class_weights.to(labels.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return loss
    return loss_fn

def main(args):
    # must use spawn method to create new processes (and can only call this once)
    # https://pytorch.org/docs/stable/notes/multiprocessing.html
    mp.set_start_method('spawn')

    if args.experiment == 'subset':
        subset_experiment(args)
    else:
        trainer = pipeline(args)

    # cleanup distributed processes, if any
    dist_cleanup(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)