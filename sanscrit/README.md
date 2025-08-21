# Sanskrit NLP Package (`sanscrit`)
---

## Installation (development mode)

```bash
# From repository root
pip install -e .[dev]
```

> **Note** – Make sure you have Python ≥3.10 and CUDA-compatible PyTorch if you plan to train the model on GPU.

---

## Using Pre-trained Models from Hugging Face Hub

The final, pre-trained models from this project are available on the Hugging Face Hub. You can use them directly in your projects with the `transformers` library without needing to run the training pipeline.

- **[OMRIDRORI/sanskrit-bert-from-scratch](https://huggingface.co/OMRIDRORI/sanskrit-bert-from-scratch)**: A BERT model trained from scratch on Sanskrit.
- **[OMRIDRORI/mbert-sanskrit-continual](https://huggingface.co/OMRIDRORI/mbert-sanskrit-continual)**: A multilingual BERT model continually trained on Sanskrit.

### Example Usage

Here is how you can use the models for a `fill-mask` task.

```python
from transformers import pipeline

# 1. Use the model trained from scratch
scratch_model = "OMRIDRORI/sanskrit-bert-from-scratch"
unmasker_scratch = pipeline('fill-mask', model=scratch_model)
result_scratch = unmasker_scratch("sa maharṣir uvāca anena [MASK] vacanena")
print(f"Result from 'from-scratch' model: {result_scratch}")


# 2. Use the continually trained model
continual_model = "OMRIDRORI/mbert-sanskrit-continual"
unmasker_continual = pipeline('fill-mask', model=continual_model)
result_continual = unmasker_continual("sa maharṣir uvāca anena [MASK] vacanena")
print(f"Result from 'continual' model: {result_continual}")
```

---

## End-to-End Pipeline

The full pre-training pipeline consists of **three** independent steps. Each step is available both as a **Python API** _and_ a **CLI command**.

| Stage | Purpose | CLI | Python API |
|-------|---------|-----|------------|
| 1. Preprocess | Clean raw JSONL documents, split into ~40-word chunks | `python -m sanscrit.cli.preprocess` | `sanscrit.preprocessing.run_preprocessing()` |
| 2. Pack | Tokenise chunks and concatenate until ≤ 512 tokens | `python -m sanscrit.cli.pack_tokens` | `sanscrit.preprocessing.pack_chunks_to_sequences()` |
| 3. Train | Pre-train BERT-MLM from scratch | `python -m sanscrit.cli.train_bert` | `sanscrit.model.pretrain.pretrain_bert()` |
| 3b. Continual | Continue pre-training mBERT on Sanskrit | `python -m sanscrit.cli.train_bert_continual` | `sanscrit.model.pretrain_continual.continual_pretrain_mbert()` |
| 4. Down-stream FT | Fine-tune any checkpoint on *root-vs-commentary* & log metrics | `python -m sanscrit.cli.fine_tune_root_commentary --model <ckpt> --tokenizer <tok> --out <dir>` | `sanscrit.model.downstream.fine_tune_root_commentary()` |

All paths (raw data, processed data, tokenizer, checkpoints) live in `sanscrit.config`. Override them via keyword arguments or command-line flags if needed.

### Minimal example

```bash
# 1) Convert raw documents into clean 40-word chunks
python -m sanscrit.cli.preprocess 

# 2) Pack chunks into <=512-token sequences
python -m sanscrit.cli.pack_tokens 

# 3) Train BERT
python -m sanscrit.cli.train_bert --epochs 5

# 3b) Continual pre-train mBERT 
python -m sanscrit.cli.train_bert_continual --epochs 3
```

### Fine-tuning on the *Root ↔ Commentary* downstream task

Run the helper CLI once for **each** checkpoint you want to compare. 

```bash
# 1) Continual-pretrained mBERT
python -m sanscrit.cli.fine_tune_root_commentary --model mbert-sanskrit-continual/final_model --tokenizer sanscrit/sanskrit-bert-tokenizer --out experiments/continual_mbert

# 2) Baseline mBERT (HF)
python -m sanscrit.cli.fine_tune_root_commentary --model bert-base-multilingual-cased --tokenizer bert-base-multilingual-cased --out experiments/baseline_mbert

# 3) Sanskrit-BERT trained *from scratch*
python -m sanscrit.cli.fine_tune_root_commentary --model sanskrit-bert-from-scratch/final_model --tokenizer sanscrit/sanskrit-bert-tokenizer --out experiments/sanskrit_scratch
```

 

Each directory written to `--out` will contain a `log_history.json` (for plotting) and a `final_model/` folder with the fine-tuned weights.

---

## Directory Structure

```
sanscrit/
├── README.md              ← you are here
├── __init__.py
├── config.py              ← centralised paths & constants
│
├── utils/                 ← shared helpers
│   ├── logging.py
│   ├── tokenizer_utils.py
│   └── data_helpers.py
│
├── preprocessing/         ← text cleaning & packing
│   ├── chunking.py
│   ├── pack_tokens.py
│   └── __init__.py
│
├── model/                 ← training / fine-tuning
│   ├── pretrain.py
│   └── __init__.py
│
└── cli/                   ← thin wrappers around the API
    ├── preprocess.py
    ├── pack_tokens.py
    ├── train_bert.py
    └── __init__.py
```

---

### Comparing multiple checkpoints

After running fine-tuning for several checkpoints (e.g. baseline mBERT, continual-pretrained mBERT, and the Sanskrit-BERT-from-scratch model) you will have three directories like:

```
experiments/
├── baseline_mbert/
│   └── log_history.json
├── continual_mbert/
│   └── log_history.json
└── sanskrit_scratch/
    └── log_history.json
```

Use the helper below to **plot accuracy vs steps** for an arbitrary list of runs:

```python
from pathlib import Path
from sanscrit.utils.plotting import plot_accuracy_curves

plot_accuracy_curves(
    runs={
        "Baseline mBERT": Path("experiments/baseline_mbert/log_history.json"),
        "Continual mBERT": Path("experiments/continual_mbert/log_history.json"),
        "Sanskrit BERT scratch": Path("experiments/sanskrit_scratch/log_history.json"),
    },
    out_png="experiments/accuracy_comparison.png",
)
```





