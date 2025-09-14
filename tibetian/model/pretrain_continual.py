"""Continual pre-training of mBERT on Tibetan.

Use `continual_pretrain_mbert()` from code, or run CLI:

    python -m tibetian.cli.train_bert_continual
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from ..config import (
    TOKENIZER_PATH,
    BERT_READY_FILE,
    MODEL_OUTPUT_DIR,
    WANDB_PROJECT_NAME_CONTINUAL,
)
from ..utils.logging import get_logger


LOGGER = get_logger(__name__)


DEFAULT_TRAINING_ARGS: Dict[str, Any] = dict(
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


def _load_custom_tokenizer(tokenizer_path: Path | str) -> BertTokenizerFast:
    tokenizer_path = Path(tokenizer_path)
    if (tokenizer_path / "vocab.txt").exists():
        LOGGER.info("Loading custom tokenizer from %s", tokenizer_path)
        try:
            tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_path), lowercase=False)
        except Exception:
            tokenizer = BertTokenizerFast(vocab_file=str(tokenizer_path / "vocab.txt"), lowercase=False)
        LOGGER.info("Tokenizer loaded – vocabulary size: %d", tokenizer.vocab_size)
        return tokenizer
    raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")


def continual_pretrain_mbert(
    data_file: Path | str | None = None,
    model_output_dir: Path | str | None = None,
    training_args_kwargs: Dict[str, Any] | None = None,
    num_epochs: int | None = None,
    base_model: str = "bert-base-multilingual-cased",
) -> None:
    """Continue MLM training starting from a pretrained base model."""
    data_file = Path(data_file or BERT_READY_FILE)
    model_output_dir = Path(model_output_dir or MODEL_OUTPUT_DIR)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    if num_epochs is not None:
        DEFAULT_TRAINING_ARGS["num_train_epochs"] = num_epochs
    combined_args = DEFAULT_TRAINING_ARGS | (training_args_kwargs or {})
    training_args = TrainingArguments(output_dir=str(model_output_dir), **combined_args)

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME_CONTINUAL
    LOGGER.info("WANDB project set to %s", WANDB_PROJECT_NAME_CONTINUAL)

    tokenizer = _load_custom_tokenizer(TOKENIZER_PATH)

    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found at {data_file}")
    LOGGER.info("Loading dataset from %s", data_file)
    dataset = load_dataset("text", data_files={"train": str(data_file)})
    LOGGER.info("Loaded %d training examples", len(dataset["train"]))

    def _tokenise(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=512, padding="max_length"
        )

    LOGGER.info("Tokenising dataset …")
    tokenised = dataset.map(
        _tokenise,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenising dataset",
    )

    LOGGER.info("Loading base model: %s", base_model)
    model = BertForMaskedLM.from_pretrained(base_model)
    LOGGER.info("Resizing token embeddings to match tokenizer vocab …")
    model.resize_token_embeddings(len(tokenizer))
    LOGGER.info("New vocabulary size: %d", len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        data_collator=data_collator,
    )

    LOGGER.info("Starting continual pre-training for %d epochs", training_args.num_train_epochs)
    trainer.train()

    final_path = model_output_dir / "final_model"
    LOGGER.info("Saving final model and tokenizer to %s", final_path)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    LOGGER.info("Continual training completed successfully!")


if __name__ == "__main__":  # pragma: no cover
    continual_pretrain_mbert()





