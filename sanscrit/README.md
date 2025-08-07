# Sanskrit NLP Package (`sanscrit`)

Modular, production-ready toolkit for preprocessing Sanskrit corpora and pre-training Transformer models from scratch.

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

All paths (raw data, processed data, tokenizer, checkpoints) live in `sanscrit.config`. Override them via keyword arguments or command-line flags if needed.

### Minimal example

```bash
# 1) Convert raw documents into clean 40-word chunks
python -m sanscrit.cli.preprocess 

# 2) Pack chunks into <=512-token sequences
python -m sanscrit.cli.pack_tokens 

# 3) Train BERT
python -m sanscrit.cli.train_bert --epochs 5
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

## Extending the Toolkit

1. **Continual pre-training** – create a new module `sanscrit.model.continual.py` reusing utilities from `pretrain.py`.
2. **Multilingual models (Sanskrit + Tibetan)** – share tokeniser or introduce a `config.MULTI_VOCAB_SIZE` and adapt the packing script.
3. **Downstream evaluation** – add a `cli/evaluate.py` that loads checkpoints and benchmarks on a task of your choice.

Because everything is function-based, new workflows can be scripted with a few lines of code.

---

## Contributing

• Follow [PEP8](https://peps.python.org/pep-0008/) conventions (we ignore the project linter for now).  
• All print/log messages **must be in English**.  
• Progress bars (`tqdm`) are preferred for long-running loops.

---

## License

MIT © 2024 Your Name
