from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid OpenMP conflicts

from datasets import load_dataset, Dataset
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


# Paths
BASE_MODEL_NAME: str = "bert-base-multilingual-cased"
CHUNKS_TXT: Path = Path("sandbox/tibetian/all_unicode/results/prepare_bert_corpus_unicode_results/bert_ready_512_unicode.txt")
OUTPUT_DIR: Path = Path("mbert-tibetan-continual-unicode")
WANDB_PROJECT: str = "Tibetan-mBERT-Continual-Unicode"

# Training args (adjust as needed)
TRAINING_ARGS_KWARGS: dict[str, Any] = dict(
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    fp16=True,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=100,
    report_to="wandb",
    dataloader_num_workers=4,
    warmup_ratio=0.06,
    weight_decay=0.01,
    max_grad_norm=1.0,
)


def load_tokenizer_from_base() -> BertTokenizerFast:
    """
    Use the base mBERT tokenizer. Since our corpus is Unicode Tibetan and we train with MLM,
    continual pretraining can still adapt the model, even if segmentation isn't perfect.
    """
    print(f"Loading base tokenizer: {BASE_MODEL_NAME}")
    tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
    print(f"Tokenizer loaded – vocabulary size: {tokenizer.vocab_size}")
    return tokenizer


def prepare_dataset(tokenizer: BertTokenizerFast) -> Dataset:
    if not CHUNKS_TXT.exists():
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_TXT}")

    print(f"Loading dataset from {CHUNKS_TXT} …")
    dataset_dict = load_dataset("text", data_files={"train": str(CHUNKS_TXT)})
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from (e.g., mbert-tibetan-continual-unicode/checkpoint-10000)",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Auto-detect latest checkpoint in OUTPUT_DIR and resume if found",
    )
    args = parser.parse_args()

    # Weights & Biases setup (uses WANDB_MODE/WANDB_API_KEY if present)
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

    # 1) Tokenizer & model
    tokenizer = load_tokenizer_from_base()

    print(f"Loading pretrained base model: {BASE_MODEL_NAME}")
    model = BertForMaskedLM.from_pretrained(BASE_MODEL_NAME)
    print(f"Base vocabulary size: {len(tokenizer)}")

    # 2) Training args
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(output_dir=str(OUTPUT_DIR), **TRAINING_ARGS_KWARGS)

    # 3) Dataset & collator
    tokenised_dataset = prepare_dataset(tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # 4) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_dataset,
        data_collator=data_collator,
    )

    # 5) Train
    print("Starting continual pretraining … (logging to:", os.environ.get("WANDB_PROJECT", "wandb-disabled"), ")")

    resume_ckpt: str | None = None
    if args.auto_resume:
        resume_ckpt = get_last_checkpoint(str(OUTPUT_DIR))
        if resume_ckpt:
            print(f"Auto-resuming from latest checkpoint: {resume_ckpt}")
        else:
            print(f"No checkpoint found in {OUTPUT_DIR}; starting fresh.")
    elif args.resume_from_checkpoint:
        resume_ckpt = args.resume_from_checkpoint
        print(f"Resuming from checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # 6) Save
    final_path = OUTPUT_DIR / "final_model"
    print(f"Saving final model and tokenizer to {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print("Done.")


if __name__ == "__main__":
    main()


