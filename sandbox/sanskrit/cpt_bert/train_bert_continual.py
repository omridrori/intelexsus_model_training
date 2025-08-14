from __future__ import annotations

import os
# ✨ Set this BEFORE any other imports to avoid OpenMP conflicts ✨
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""Continual pre-training of mBERT on Sanskrit.

This script loads a *custom* Sanskrit tokenizer, resizes the embeddings
of the multilingual BERT base model to the new vocabulary, and
continues masked-language-model (MLM) training on the provided corpus.

Run from the project root:
```
python sandbox/sanskrit/cpt_bert/train_bert_continual.py
```
"""

from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants ───────────────────────────────────────────────────────────
BASE_MODEL_NAME: str = "bert-base-multilingual-cased"
TOKENIZER_VOCAB: Path = Path("sanscrit/sanskrit-bert-tokenizer/vocab.txt")
DATA_FILE: Path = Path("data/sanskrit_preprocessed/sanskrit_bert_ready_512.txt")
OUTPUT_DIR: Path = Path("mbert-sanskrit-continual")
WANDB_PROJECT: str = "Sanskrit-mBERT-Continual"

# Training hyper-parameters (can be overridden via CLI, env vars, etc.)
TRAINING_ARGS_KWARGS: dict[str, Any] = dict(
    overwrite_output_dir=True,
    num_train_epochs=3,          # keep short for continual tuning
    per_device_train_batch_size=8,
    learning_rate=2e-5,         # low LR for fine-tuning
    fp16=True,
    save_steps=5_000,
    save_total_limit=2,
    logging_steps=20,
    report_to="wandb",
    dataloader_num_workers=4,
    warmup_ratio=0.06,
    weight_decay=0.01,
    max_grad_norm=1.0,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions ─────────────────────────────────────────────────────────────

def load_tokenizer(vocab_path: Path) -> BertTokenizerFast:
    """Load a *vocab.txt* into a fast tokenizer."""
    if not vocab_path.exists():
        raise FileNotFoundError(f"Tokenizer vocab not found at {vocab_path}")

    print(f"Loading custom tokenizer from: {vocab_path}")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), lowercase=False)
    print(f"Tokenizer loaded – vocabulary size: {tokenizer.vocab_size}")
    return tokenizer


def prepare_dataset(tokenizer: BertTokenizerFast):
    """Load text file dataset and apply tokenisation with a progress bar."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_FILE}")

    print(f"Loading dataset from {DATA_FILE} …")
    dataset = load_dataset("text", data_files={"train": str(DATA_FILE)})
    print(f"Loaded {len(dataset['train'])} training examples")

    def _tokenise(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    print("Tokenising dataset …")
    tokenised = dataset.map(
        _tokenise,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenising dataset",
    )
    print("Tokenisation complete.")
    return tokenised


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    # WandB setup (no-op if WANDB disabled)
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    # ------------------------------------------------------------------
    # 1. Tokenizer & model ------------------------------------------------

    tokenizer = load_tokenizer(TOKENIZER_VOCAB)

    print(f"Loading pretrained base model: {BASE_MODEL_NAME}")
    model = BertForMaskedLM.from_pretrained(BASE_MODEL_NAME)
    print("Resizing token embeddings to match new vocabulary …")
    model.resize_token_embeddings(len(tokenizer))
    print(f"New vocabulary size: {len(tokenizer)}")

    # ------------------------------------------------------------------
    # 2. Training arguments ----------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(output_dir=str(OUTPUT_DIR), **TRAINING_ARGS_KWARGS)

    # ------------------------------------------------------------------
    # 3. Dataset & data collator -----------------------------------------

    tokenised_dataset = prepare_dataset(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # ------------------------------------------------------------------
    # 4. Trainer ----------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_dataset["train"],
        data_collator=data_collator,
    )

    # ------------------------------------------------------------------
    # 5. Training ---------------------------------------------------------

    effective_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    steps_per_epoch = len(tokenised_dataset["train"]) // effective_bs

    print("\n=== Continual Pre-Training Configuration ===")
    print(f"Epochs:           {training_args.num_train_epochs}")
    print(f"Effective BS:     {effective_bs}")
    print(f"Steps / epoch:    {steps_per_epoch}")
    print("Starting training …")

    trainer.train()

    # ------------------------------------------------------------------
    # 6. Save final model -------------------------------------------------

    final_path = OUTPUT_DIR / "final_model"
    print(f"Saving final model and tokenizer to {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print("Training completed successfully! 🎉")


if __name__ == "__main__":
    main()
