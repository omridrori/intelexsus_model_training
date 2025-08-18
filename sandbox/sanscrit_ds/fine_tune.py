import os
# Set the environment variable to avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Any, Dict

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import numpy as np
import evaluate


def main():
    """Fine-tune a model for root commentary classification."""
    # --- Configuration ---
    base_model_path = Path("mbert-sanskrit-continual/final_model")
    tokenizer_path = Path("sanscrit/sanskrit-bert-tokenizer")
    train_file = Path("data/downstream_tasks/root_commentary_class/train.jsonl")
    test_file = Path("data/downstream_tasks/root_commentary_class/test.jsonl")
    output_dir = Path("sandbox/sanscrit_ds/fine_tuning_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_project_name = "Sanskrit-DS-Root-Commentary-Classification"
    os.environ["WANDB_PROJECT"] = wandb_project_name

    # --- Load Tokenizer ---
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # --- Load and Preprocess Data ---
    print("Loading and preprocessing data...")
    dataset = load_dataset("json", data_files={"train": str(train_file), "test": str(test_file)})

    def preprocess_function(examples):

        tokenized_inputs = tokenizer(
            examples["root"],
            examples["commentary"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        # The label field is already a string "0" or "1", convert to int
        tokenized_inputs["label"] = [int(label) for label in examples["label"]]
        return tokenized_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    # --- Load Model ---
    print(f"Loading model from {base_model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=2)

    # --- Metrics ---
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

    # --- Trainer Setup ---
    per_device_train_batch_size = 8
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        logging_steps=10,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- Train and Evaluate ---
    print("Starting training...")
    trainer.train()

    print("Evaluating final model...")
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # --- Save Final Model ---
    final_model_path = output_dir / "final_model"
    print(f"Saving final model to {final_model_path}")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    print("Fine-tuning complete!")


if __name__ == "__main__":
    main()