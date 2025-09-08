"""Fine-tune baseline mBERT on tib_04_task Wylie using task_3_label (binary NSCR/SCR).
Usage:
python finetune_tib_04_baseline.py \
  --epochs 3 --batch 8 --lr 2e-5 
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
import argparse

from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / 'data' / 'downstream_tasks' / 'Tibetan' / 'tib_04_task_wylie'
RESULTS_DIR = HERE / 'results' / 'finetune_tib_04_baseline_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------

def load_jsonl(path: Path, label_field: str) -> List[Dict[str, str]]:
    recs = []
    with path.open(encoding='utf-8') as f:
        for ln in f:
            obj = json.loads(ln)
            recs.append({'text': obj.get('text', ''), 'label': obj.get(label_field)})
    return recs


def build_dataset(label_field: str):
    train = load_jsonl(DATA_DIR / 'train.jsonl', label_field)
    test = load_jsonl(DATA_DIR / 'test.jsonl', label_field)
    labels = sorted({r['label'] for r in train + test})
    l2id = {l:i for i,l in enumerate(labels)}
    for rec in train + test:
        rec['label'] = l2id[rec['label']]
    ds = DatasetDict({'train': Dataset.from_list(train), 'test': Dataset.from_list(test)})
    return ds, labels


def metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted', zero_division=0),
        'recall': recall_score(labels, preds, average='weighted', zero_division=0),
        'f1': f1_score(labels, preds, average='weighted', zero_division=0)
    }

# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert-base-multilingual-cased')
    parser.add_argument('--tokenizer', default='bert-base-multilingual-cased')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label_field', default='task_4_label')
    args = parser.parse_args()

    set_seed(args.seed)
    ds, label_names = build_dataset(args.label_field)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    def tok_fn(b):
        return tok(b['text'], truncation=True, max_length=512)
    ds_tok = ds.map(tok_fn, batched=True)
    collator = DataCollatorWithPadding(tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(label_names), id2label={i:l for i,l in enumerate(label_names)}, label2id={l:i for i,l in enumerate(label_names)})

    out_dir = RESULTS_DIR / 'model'
    train_args = TrainingArguments(
        output_dir=str(out_dir), overwrite_output_dir=True, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        learning_rate=args.lr, eval_strategy='epoch', save_strategy='epoch',
        logging_dir=str(out_dir/'logs'), logging_steps=50, seed=args.seed, load_best_model_at_end=True, metric_for_best_model='f1')

    trainer = Trainer(model=model, args=train_args, train_dataset=ds_tok['train'], eval_dataset=ds_tok['test'],
                      tokenizer=tok, data_collator=collator, compute_metrics=metrics)
    trainer.train()
    res = trainer.evaluate()
    (RESULTS_DIR/'metrics.json').write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Finished. Metrics saved.')

if __name__ == '__main__':
    main()
