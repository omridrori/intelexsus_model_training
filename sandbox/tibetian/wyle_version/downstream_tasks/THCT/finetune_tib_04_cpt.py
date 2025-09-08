from pathlib import Path,  PurePath
import argparse, json
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, set_seed)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / 'data' / 'downstream_tasks' / 'Tibetan' / 'tib_04_task_wylie'
RESULTS_DIR = HERE / 'results' / 'finetune_tib_04_cpt_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CPT_MODEL_PATH = 'mbert-tibetan-continual/final_model'


def load_jsonl(p: Path, fld: str):
    with p.open(encoding='utf-8') as f:
        return [{'text': (o:=(json.loads(l)))['text'], 'label': o.get(fld)} for l in f]

def build_dataset(fld: str):
    tr = load_jsonl(DATA_DIR/'train.jsonl', fld)
    te = load_jsonl(DATA_DIR/'test.jsonl', fld)
    labels = sorted({r['label'] for r in tr+te})
    m = {l:i for i,l in enumerate(labels)}
    for r in tr+te: r['label']=m[r['label']]
    return DatasetDict({'train':Dataset.from_list(tr),'test':Dataset.from_list(te)}), labels

def metrics(p):
    logits, labs = p
    preds = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labs,preds), 'precision': precision_score(labs,preds,average='weighted',zero_division=0), 'recall': recall_score(labs,preds,average='weighted',zero_division=0), 'f1': f1_score(labs,preds,average='weighted',zero_division=0)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--lr',type=float,default=2e-5)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--label_field',default='task_4_label')
    args = ap.parse_args()

    set_seed(args.seed)
    ds, label_names = build_dataset(args.label_field)
    tok = AutoTokenizer.from_pretrained(CPT_MODEL_PATH)
    ds_tok = ds.map(lambda b: tok(b['text'], truncation=True, max_length=512), batched=True)
    col = DataCollatorWithPadding(tok)
    model = AutoModelForSequenceClassification.from_pretrained(CPT_MODEL_PATH, num_labels=len(label_names), id2label={i:l for i,l in enumerate(label_names)}, label2id={l:i for i,l in enumerate(label_names)})

    out_dir = RESULTS_DIR/'model'
    targs = TrainingArguments(output_dir=str(out_dir),overwrite_output_dir=True,num_train_epochs=args.epochs,per_device_train_batch_size=args.batch,per_device_eval_batch_size=args.batch,learning_rate=args.lr,eval_strategy='epoch',save_strategy='epoch',logging_dir=str(out_dir/'logs'),logging_steps=50,seed=args.seed,load_best_model_at_end=True,metric_for_best_model='f1')

    trainer = Trainer(model=model,args=targs,train_dataset=ds_tok['train'],eval_dataset=ds_tok['test'],tokenizer=tok,data_collator=col,compute_metrics=metrics)
    trainer.train()
    res = trainer.evaluate()
    (RESULTS_DIR/'metrics.json').write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Done. Metrics saved.')

if __name__=='__main__':
    main()
