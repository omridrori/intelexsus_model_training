from __future__ import annotations

import os
# ✨ Avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""Continual pre-training of mBERT on Tibetan Wylie.

This script mirrors *sandbox/sanskrit/cpt_bert/train_bert_continual.py*
but points to the Tibetan corpus and tokenizer.
Run from the project root:
```
python sandbox/tibetian/cpt_bert/train_bert_continual.py
```
"""

from pathlib import Path
from typing import Any

import torch  # noqa: F401 – imported for future fp16 / bfloat16 support
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm  # noqa: F401 – progress bars
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ────────────────────────────────────────────────────────────────────────────────
# Paths & constants ──────────────────────────────────────────────────────────────
BASE_MODEL_NAME: str = "bert-base-multilingual-cased"
TOKENIZER_VOCAB: Path = Path("tibetan-tokenizer/vocab.txt")
DATA_FILE: Path = Path("data/tibetan_bert_ready_512.txt")  # produced by prepare_bert_corpus.py
OUTPUT_DIR: Path = Path("mbert-tibetan-continual")
WANDB_PROJECT: str = "Tibetan-mBERT-Continual"

# Training hyper-parameters (override via CLI / env if needed)
TRAINING_ARGS_KWARGS: dict[str, Any] = dict(
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
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

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions ──────────────────────────────────────────────────────────────

def load_tokenizer(vocab_path: Path) -> BertTokenizerFast:
    """Load a *vocab.txt* into a fast tokenizer."""
    if not vocab_path.exists():
        raise FileNotFoundError(f"Tokenizer vocab not found at {vocab_path}")

    print(f"Loading custom tokenizer from: {vocab_path}")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), lowercase=False)
    print(f"Tokenizer loaded – vocabulary size: {tokenizer.vocab_size}")
    return tokenizer


def prepare_dataset(tokenizer: BertTokenizerFast) -> Dataset:
    """Load pre-chunked plain-text corpus and tokenise."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_FILE}")

    print(f"Loading dataset from {DATA_FILE} …")
    dataset_dict = load_dataset("text", data_files={"train": str(DATA_FILE)})
    print(f"Loaded {len(dataset_dict['train'])} training sequences (<=512 tokens raw)")

    def _tokenise(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    tokenised = dataset_dict["train"].map(
        _tokenise,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenising dataset",
    )
    print("Tokenisation complete.")
    return tokenised


# ────────────────────────────────────────────────────────────────────────────────
# Main entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    # 1. Tokenizer & model -------------------------------------------------------
    tokenizer = load_tokenizer(TOKENIZER_VOCAB)

    print(f"Loading pretrained base model: {BASE_MODEL_NAME}")
    model = BertForMaskedLM.from_pretrained(BASE_MODEL_NAME)
    print("Resizing token embeddings to match new vocabulary …")
    model.resize_token_embeddings(len(tokenizer))
    print(f"New vocabulary size: {len(tokenizer)}")

    # 2. Training arguments ------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(output_dir=str(OUTPUT_DIR), **TRAINING_ARGS_KWARGS)

    # 3. Dataset & data collator -------------------------------------------------
    tokenised_dataset = prepare_dataset(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # 4. Trainer -----------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_dataset,
        data_collator=data_collator,
    )

    # 5. Training ---------------------------------------------------------------
    effective_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    steps_per_epoch = len(tokenised_dataset) // effective_bs

    print("\n=== Continual Pre-Training Configuration ===")
    print(f"Epochs:           {training_args.num_train_epochs}")
    print(f"Effective BS:     {effective_bs}")
    print(f"Steps / epoch:    {steps_per_epoch}")
    print("Starting training …")

    trainer.train()

    # 6. Save final model --------------------------------------------------------
    final_path = OUTPUT_DIR / "final_model"
    print(f"Saving final model and tokenizer to {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print("Training completed successfully! 🎉")


if __name__ == "__main__":
    main()
