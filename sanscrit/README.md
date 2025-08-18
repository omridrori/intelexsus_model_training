# Sanskrit NLP Package (`sanscrit`)
---

## Installation (development mode)

```bash
# From repository root
pip install -e .[dev]
```

> **Note** – Make sure you have Python ≥3.10 and CUDA-compatible PyTorch if you plan to train the model on GPU.

---

## End-to-End Pipeline

The full pre-training pipeline consists of **three** independent steps. Each step is available both as a **Python API** _and_ a **CLI command**.

| Stage | Purpose | CLI | Python API |
|-------|---------|-----|------------|
| 1. Preprocess | Clean raw JSONL documents, split into ~40-word chunks | `python -m sanscrit.cli.preprocess` | `sanscrit.preprocessing.run_preprocessing()` |
| 2. Pack | Tokenise chunks and concatenate until ≤ 512 tokens | `python -m sanscrit.cli.pack_tokens` | `sanscrit.preprocessing.pack_chunks_to_sequences()` |
| 3. Train | Pre-train BERT-MLM from scratch | `python -m sanscrit.cli.train_bert` | `sanscrit.model.pretrain.pretrain_bert()` |
| 3b. Continual | Continue pre-training mBERT on Sanskrit | `python -m sanscrit.cli.train_bert_continual` | `sanscrit.model.pretrain_continual.continual_pretrain_mbert()` |

All paths (raw data, processed data, tokenizer, checkpoints) live in `sanscrit.config`. Override them via keyword arguments or command-line flags if needed.

### Minimal example

```bash
# 1) Convert raw documents into clean 40-word chunks
python -m sanscrit.cli.preprocess 

# 2) Pack chunks into <=512-token sequences
python -m sanscrit.cli.pack_tokens 

# 3) Train BERT
python -m sanscrit.cli.train_bert --epochs 5

# 3b) Continual pre-train mBERT (optional)
python -m sanscrit.cli.train_bert_continual --epochs 3
```

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




