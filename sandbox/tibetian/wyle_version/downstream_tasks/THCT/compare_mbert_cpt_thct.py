"""Compare baseline mBERT vs CPT-continued mBERT on tib_04_task Wylie (task_4_label).
Trains each model for a few epochs, evaluates every N steps, and plots weighted-F1 curves.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
import argparse
import matplotlib.pyplot as plt

from datasets import DatasetDict, Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
import numpy as np
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / 'data' / 'downstream_tasks' / 'Tibetan' / 'tib_04_task_wylie'
RESULTS_DIR = HERE / 'results' / 'compare_mbert_cpt_thct_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS: Dict[str, Dict[str, str]] = {
    'baseline': {
        'model': 'bert-base-multilingual-cased',
        'tokenizer': 'bert-base-multilingual-cased',
    },
    'cpt': {
        'model': 'mbert-tibetan-continual/final_model',
        'tokenizer': 'mbert-tibetan-continual/final_model',
    },
}

LABEL_FIELD = 'task_4_label'


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    recs: List[Dict[str, str]] = []
    with path.open(encoding='utf-8') as f:
        for ln in f:
            obj = json.loads(ln)
            recs.append({'text': obj.get('text', ''), 'label': obj.get(LABEL_FIELD)})
    return recs


def build_dataset() -> tuple[DatasetDict, List[str]]:
    train_recs = load_jsonl(DATA_DIR / 'train.jsonl')
    test_recs = load_jsonl(DATA_DIR / 'test.jsonl')
    labels = sorted({r['label'] for r in train_recs + test_recs})
    l2id = {l: i for i, l in enumerate(labels)}
    for r in train_recs + test_recs:
        r['label'] = l2id[r['label']]
    ds = DatasetDict({'train': Dataset.from_list(train_recs), 'test': Dataset.from_list(test_recs)})
    return ds, labels


def compute_f1(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {'f1': f1_score(labels, preds, average='weighted')}


def train_and_collect(tag: str, model_path: str, tok_path: str, ds: DatasetDict, labels: List[str], args):
    tok = AutoTokenizer.from_pretrained(tok_path)

    def tok_fn(batch):
        return tok(batch['text'], truncation=True, max_length=512)

    ds_tok = ds.map(tok_fn, batched=True)
    collator = DataCollatorWithPadding(tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
    )

    out_dir = RESULTS_DIR / f'{tag}_run'
    tr_args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        eval_strategy='steps',
        save_strategy='no',
        eval_steps=args.eval_steps,
        logging_steps=args.eval_steps,
        logging_dir=str(out_dir / 'logs'),
        seed=args.seed,
    )

    # Collect (step, f1) pairs
    metrics_hist: List[tuple[int, float]] = []

    from transformers import TrainerCallback

    class LogF1Callback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if state.is_world_process_zero:
                f1 = metrics.get('eval_f1')
                if f1 is not None:
                    metrics_hist.append((state.global_step, f1))

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=ds_tok['train'],
        eval_dataset=ds_tok['test'],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_f1,
        callbacks=[LogF1Callback()],
    )

    trainer.train()
    return metrics_hist


def plot_curves(curves: Dict[str, List[tuple[int, float]]]):
    plt.figure(figsize=(8, 5))
    for tag, pts in curves.items():
        if not pts:
            continue
        steps, f1s = zip(*pts)
        plt.plot(steps, f1s, label=tag)
    plt.xlabel('Global step')
    plt.ylabel('Weighted F1')
    plt.title('mBERT vs CPT fine-tuning on tib_04 (task_4_label)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = RESULTS_DIR / 'f1_comparison.png'
    plt.savefig(out_path, dpi=300)
    print(f'Plot saved to {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--eval_steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    ds, labels = build_dataset()

    curves: Dict[str, List[tuple[int, float]]] = {}
    for tag, paths in MODELS.items():
        print(f'\n=== Training {tag} model ===')
        curves[tag] = train_and_collect(tag, paths['model'], paths['tokenizer'], ds, labels, args)

    plot_curves(curves)


if __name__ == '__main__':
    main()
