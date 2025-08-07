"""Pre-train a BERT model from scratch on Sanskrit data.

This is a mostly *unchanged* port of the original sandbox script but turned
into reusable functions.  Import `pretrain_bert()` or run this file as a
module.  All paths/constants come from `sanscrit.config` so you only have to
change them once.
"""

from __future__ import annotations

import os
# Ensure the env var is set **before** any library (Torch/Tokenizers) loads OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from ..config import (
    TOKENIZER_PATH,
    BERT_READY_FILE,
    MODEL_OUTPUT_DIR,
    WANDB_PROJECT_NAME,
)
from ..utils.logging import get_logger
from ..utils.tokenizer_utils import load_tokenizer

# Make Intel MKL happy on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

LOGGER = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model / training hyper-parameters -----------------------------------------

MODEL_CONFIG = BertConfig(
    vocab_size=32_000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
)

DEFAULT_TRAINING_ARGS: Dict[str, Any] = dict(
    overwrite_output_dir=False,  # allows resume
    num_train_epochs=5,
    # memory-saving tricks ---------------------------------------------------
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,  # effective batch 32
    gradient_checkpointing=True,
    fp16=True,
    # logging / checkpointing ----------------------------------------------
    save_steps=100,
    save_total_limit=3,
    prediction_loss_only=True,
    logging_steps=20,
    report_to="wandb",
    # misc ------------------------------------------------------------------
    dataloader_num_workers=4,
    warmup_ratio=0.06,
    learning_rate=1e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
)

# ---------------------------------------------------------------------------
# Public API -----------------------------------------------------------------


def pretrain_bert(
    data_file: Path | str | None = None,
    model_output_dir: Path | str | None = None,
    training_args_kwargs: Dict[str, Any] | None = None,
    num_epochs: int | None = None,
) -> None:
    """Run BERT pre-training from scratch.

    Parameters
    ----------
    data_file: path to *pre-tokenised* training text (one sentence per line).
               Defaults to ``sanscrit.config.BERT_READY_FILE``.
    model_output_dir: where to store checkpoints.  Defaults to
                      ``sanscrit.config.MODEL_OUTPUT_DIR``.
    training_args_kwargs: extra/override kwargs passed to
                          ``transformers.TrainingArguments``.
    num_epochs: optional convenience override for epoch count.
    """
    data_file = Path(data_file or BERT_READY_FILE)
    model_output_dir = Path(model_output_dir or MODEL_OUTPUT_DIR)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    if num_epochs is not None:
        DEFAULT_TRAINING_ARGS["num_train_epochs"] = num_epochs

    # Merge custom kwargs ---------------------------------------------------
    if training_args_kwargs:
        combined_args = DEFAULT_TRAINING_ARGS | training_args_kwargs
    else:
        combined_args = DEFAULT_TRAINING_ARGS.copy()

    training_args = TrainingArguments(output_dir=str(model_output_dir), **combined_args)

    # ----------------------------------------------------------------------
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME
    LOGGER.info("WANDB project set to %s", WANDB_PROJECT_NAME)

    # ----------------------------------------------------------------------
    # Load tokenizer & dataset --------------------------------------------
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    LOGGER.info("Loading dataset from %s", data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found at {data_file}")

    dataset = load_dataset("text", data_files={"train": str(data_file)})
    LOGGER.info("Loaded %d training examples", len(dataset["train"]))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    LOGGER.info("Tokenising dataset …")
    tokenised_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenising dataset",
    )

    # ----------------------------------------------------------------------
    model = BertForMaskedLM(config=MODEL_CONFIG)
    LOGGER.info("Model created with %s parameters", f"{model.num_parameters():,}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenised_dataset["train"],
    )

    LOGGER.info("Starting training for %d epochs", training_args.num_train_epochs)
    trainer.train()

    # ----------------------------------------------------------------------
    final_model_path = model_output_dir / "final_model"
    LOGGER.info("Saving final model to %s", final_model_path)
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    LOGGER.info("Training completed successfully!")


# ---------------------------------------------------------------------------
# CLI helper -----------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    pretrain_bert()