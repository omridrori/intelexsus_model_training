"""Fine-tune (or continue fine-tuning) an MBERT model on Wylie-transcribed tib_03_task using task_3_label (NSCR vs SCR).

Usage (example):
python finetune_aact_wylie.py \
  --model_name_or_path bert-base-multilingual-cased \
  --output_dir runs/aact_wylie_mbert \
  --epochs 3

All results (trainer logs, metrics) will also be copied to
results/finetune_aact_wylie_results.
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Dict
import argparse

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / 'data' / 'downstream_tasks' / 'Tibetan' / 'tib_03_task_wylie'
RESULTS_DIR = Path(__file__).resolve().parent / 'results' / 'finetune_aact_wylie_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """Load a jsonl file into list of dicts with 'text' and 'label' keys."""
    records: List[Dict[str, str]] = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            records.append({'text': obj.get('text', ''), 'label': obj.get('task_3_label')})
    return records


def build_dataset() -> DatasetDict:
    from datasets import Dataset
    train_records = load_jsonl(DATA_DIR / 'train.jsonl')
    test_records = load_jsonl(DATA_DIR / 'test.jsonl')
    label_list = sorted({rec['label'] for rec in train_records + test_records})
    label2id = {l: i for i, l in enumerate(label_list)}
    for rec in train_records + test_records:
        rec['label'] = label2id[rec['label']]
    ds = DatasetDict({
        'train': Dataset.from_list(train_records),
        'test': Dataset.from_list(test_records)
    })
    ds.info = {'label_list': label_list}
    return ds


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
    parser.add_argument('--model_name_or_path', default='bert-base-multilingual-cased', help='Path or hub id')
    parser.add_argument('--output_dir', default=str(ROOT / 'experiments' / 'aact_wylie_mbert'), help='Where to save model')
    parser.add_argument('--tokenizer_path', default=None, help='Optional path to tokenizer (if different from model)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print('Loading dataset...')
    ds = build_dataset()
    label_list = ds.info['label_list']  # type: ignore
    num_labels = len(label_list)

    tok_source = args.tokenizer_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_source)

    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, max_length=512)

    ds_tok = ds.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=num_labels, id2label={i: l for i, l in enumerate(label_list)}, label2id={l: i for i, l in enumerate(label_list)})

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
        logging_dir=str(Path(args.output_dir)/'logs'),
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

    # Save metrics to results directory
    results_path = RESULTS_DIR / 'metrics.json'
    with results_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f'Metrics saved to {results_path}')

    # Copy label list for reference
    (RESULTS_DIR / 'label_list.txt').write_text('\n'.join(label_list), encoding='utf-8')


if __name__ == '__main__':
    main()
