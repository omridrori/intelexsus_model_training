"""Fine-tune sequence-classification models on the *root vs. commentary* task.

The logic is adapted from the experimental sandbox script but turned into a
re-usable function so that *any* pretrained checkpoint (mBERT / continual /
Sanskrit-BERT-scratch) can be evaluated under identical hyper-parameters.

Example
-------
>>> from sanscrit.model.downstream import fine_tune_root_commentary
>>> fine_tune_root_commentary(
...     model_name_or_path="bert-base-multilingual-cased",
...     tokenizer_name_or_path="bert-base-multilingual-cased",
...     output_dir="experiments/baseline_mbert",
... )
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

__all__ = ["fine_tune_root_commentary"]

# ---------------------------------------------------------------------------
# Paths (relative to repo-root so the function works outside the sandbox) ----

_DATA_DIR = Path("data") / "downstream_tasks" / "root_commentary_class"
_TRAIN_FILE = _DATA_DIR / "train.jsonl"
_TEST_FILE = _DATA_DIR / "test.jsonl"

# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------

def fine_tune_root_commentary(
    *,
    model_name_or_path: str | os.PathLike,
    tokenizer_name_or_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    eval_steps: int = 50,
    overwrite_output_dir: bool = True,
    wandb_project: str = "Sanskrit-Root-Commentary-FT",
) -> Dict[str, Any]:
    """Fine-tune *model* on the downstream task & return final metrics.

    Parameters
    ----------
    model_name_or_path
        HF Hub identifier or local path to a *sequence-classification* or
        *encoder-only* checkpoint.
    tokenizer_name_or_path
        Path/identifier of a compatible tokenizer (must match the checkpoint's
        vocabulary).
    output_dir
        Directory where all artefacts (checkpoints, log_history.json, final
        model) will be stored.
    num_epochs, learning_rate, batch_size, eval_steps
        Training hyper-parameters.  Defaults mirror the research notebook.
    overwrite_output_dir
        If *True* (default) will remove existing contents.
    wandb_project
        Project name used for experiment tracking.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ensure deterministic behaviour on each call
    torch.manual_seed(42)
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid OpenMP clash on Windows

    # ------------------------------------------------------------------
    # Tokeniser & dataset
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    dataset = load_dataset(
        "json",
        data_files={"train": str(_TRAIN_FILE), "test": str(_TEST_FILE)},
    )

    def _tokenise(example):
        toks = tokenizer(
            example["root"],
            example["commentary"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        toks["label"] = int(example["label"])
        return toks

    tokenised = dataset.map(_tokenise, remove_columns=dataset["train"].column_names)

    # ------------------------------------------------------------------
    # Model & Trainer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=2
    )

    metrics_accuracy = evaluate.load("accuracy")
    metrics_f1 = evaluate.load("f1")

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = metrics_accuracy.compute(predictions=preds, references=labels)["accuracy"]
        f1 = metrics_f1.compute(predictions=preds, references=labels, average="weighted")[
            "f1"
        ]
        return {"accuracy": acc, "f1": f1}

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=overwrite_output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )

    trainer.train()

    # persist log history for later plotting --------------------------------
    import json

    with open(output_dir / "log_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    final_metrics = trainer.evaluate()
    # also save the final model under /final_model so that downstream code
    # mirrors the sandbox structure.
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))

    return final_metrics
