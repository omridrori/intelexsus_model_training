import os
import numpy as np
import math
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from datasets import load_dataset, disable_progress_bar, enable_progress_bar, load_from_disk
import torch
import logging


def compute_metrics(p: EvalPrediction):
    """Computes MLM accuracy from model predictions."""
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    mask = labels != -100  # Ignore non-masked tokens
    accuracy = (preds[mask] == labels[mask]).mean()
    return {"mlm_accuracy": accuracy}


def run_training(training_args, model_config, tokenized_dataset_path, tokenizer_path, project_name, train_samples=0, eval_samples=0):
    """
    Core function to run the BERT model training and evaluation loop.
    It now expects a path to a pre-tokenized dataset saved by `save_to_disk`.
    """
    os.environ["WANDB_PROJECT"] = project_name
    logging.info(f"Set W&B Project to '{project_name}'")

    # 1. Load Tokenizer
    print(f"Loading tokenizer from: {tokenizer_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer directory not found at {tokenizer_path}")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    model_config.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # 2. Load Pre-tokenized Dataset
    print(f"Loading pre-tokenized dataset from: {tokenized_dataset_path}")
    if not os.path.isdir(tokenized_dataset_path):
        raise FileNotFoundError(
            f"Tokenized dataset directory not found at {tokenized_dataset_path}. "
            f"Please run the `preprocessing/tokenize_data.py` script first."
        )
    
    tokenized_dataset = load_from_disk(tokenized_dataset_path)

    # --- Dataset Subsetting Logic ---
    if train_samples > 0 and len(tokenized_dataset) > train_samples:
        print(f"Selecting a subset of {train_samples} examples for training/evaluation.")
        tokenized_dataset = tokenized_dataset.shuffle(seed=42).select(range(train_samples))

    train_dataset = tokenized_dataset
    eval_dataset = None

    if eval_samples > 0:
        if len(tokenized_dataset) < eval_samples:
            raise ValueError(f"Requested {eval_samples} evaluation samples, but the dataset (or subset) only has {len(tokenized_dataset)} examples.")
        
        # Split the (potentially subsetted) dataset into training and evaluation sets
        split_dataset = train_dataset.train_test_split(test_size=eval_samples, shuffle=True, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        print(f"Dataset split into {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples.")

    print("Pre-tokenized dataset loaded and formatted.")
    

    # 3. Initialize Model
    print("Initializing a new BERT model from scratch...")
    model = BertForMaskedLM(config=model_config)
    print(f"Model created. Number of parameters: {model.num_parameters():,}")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if eval_dataset else None,
    )

    # --- Start Training ---
    print("Starting model training...")
    trainer.train()
    print("Training finished.")

    # --- Save Final Model ---
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    print(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("Model and tokenizer saved.")
    
    return trainer 