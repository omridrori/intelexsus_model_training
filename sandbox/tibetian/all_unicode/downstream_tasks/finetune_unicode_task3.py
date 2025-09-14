"""Fine-tune (or continue fine-tuning) an MBERT model on tib_03_task using task_3_label (Unicode text).

Usage (example):
python finetune_unicode_task3.py \
  --model_name_or_path ../../../../mbert-tibetan-continual-unicode/checkpoint-10000 \
  --output_dir ../../../../experiments/aact_unicode_mbert_task3 \
  --epochs 3

All results (trainer logs, metrics) are copied to
results/finetune_unicode_task3_results under the script's parent directory.
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# This resolves to the repository root from this file's location
ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / 'data' / 'downstream_tasks' / 'Tibetan' / 'tib_03_task'

# Per workspace rule: save script results under parent_dir/results/<script_name>_results
PARENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PARENT_DIR / 'results' / 'finetune_unicode_task3_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def load_jsonl(path: Path, text_key: str, label_key: str) -> List[Dict[str, str]]:
    """Load a jsonl file into a list of dicts with 'text' and 'label' keys.

    Progress bar is shown while reading lines.
    """
    records: List[Dict[str, str]] = []
    with path.open(encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {path.name}"):
            obj = json.loads(line)
            text = obj.get(text_key, '')
            label = obj.get(label_key)
            records.append({'text': text, 'label': label})
    return records


def build_dataset(text_key: str = 'text', label_key: str = 'task_3_label') -> Tuple[DatasetDict, List[str]]:
    """Read train/test jsonl and return DatasetDict and label list."""
    train_records = load_jsonl(DATA_DIR / 'train.jsonl', text_key, label_key)
    test_records = load_jsonl(DATA_DIR / 'test.jsonl', text_key, label_key)

    label_list = sorted({rec['label'] for rec in train_records + test_records})
    label2id = {label_value: idx for idx, label_value in enumerate(label_list)}

    for rec in train_records + test_records:
        rec['label'] = label2id[rec['label']]

    ds = DatasetDict({
        'train': Dataset.from_list(train_records),
        'test': Dataset.from_list(test_records),
    })
    return ds, label_list


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted', zero_division=0),
        'recall': recall_score(labels, preds, average='weighted', zero_division=0),
        'f1': f1_score(labels, preds, average='weighted', zero_division=0),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default=str(ROOT / 'mbert-tibetan-continual-unicode' / 'checkpoint-140000'), help='Path or hub id of the base model')
    parser.add_argument('--output_dir', default=str(ROOT / 'experiments' / 'aact_unicode_mbert_task3'), help='Where to save the fine-tuned model')
    parser.add_argument('--tokenizer_path', default=None, help='Optional path to tokenizer (if different from model)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print('Loading dataset (Unicode tib_03_task)...')
    ds, label_list = build_dataset()
    num_labels = len(label_list)

    tok_source = args.tokenizer_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_source)

    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, max_length=512)

    ds_tok = ds.map(tokenize_fn, batched=True, desc='Tokenizing')
    data_collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label={i: l for i, l in enumerate(label_list)},
        label2id={l: i for i, l in enumerate(label_list)}
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy='epoch',
        save_strategy='epoch',
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_dir=str(Path(args.output_dir) / 'logs'),
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok['train'],
        eval_dataset=ds_tok['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print('Starting training...')
    trainer.train()

    print('Evaluating...')
    metrics = trainer.evaluate()
    print(metrics)

    # Save metrics and artifacts to results directory
    results_path = RESULTS_DIR / 'metrics.json'
    with results_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f'Metrics saved to {results_path}')

    # Save label list for reference
    (RESULTS_DIR / 'label_list.txt').write_text('\n'.join(label_list), encoding='utf-8')

    # Save run args for reproducibility
    (RESULTS_DIR / 'run_args.json').write_text(
        json.dumps({
            'model_name_or_path': args.model_name_or_path,
            'output_dir': args.output_dir,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'seed': args.seed,
        }, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )


if __name__ == '__main__':
    main()






