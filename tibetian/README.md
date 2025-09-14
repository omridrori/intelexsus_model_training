# Tibetan NLP Package (`tibetian`)

## Installation (development mode)

```bash
pip install -e .[dev]
```

> Requires Python ≥3.10 and CUDA-enabled PyTorch for GPU training.

---

## End-to-End Pipeline

This mirrors the `sanscrit` package and provides Tibetan-specific preprocessing and continual pre-training utilities.

| Stage | Purpose | CLI | Python API |
|-------|---------|-----|------------|
| 1. Prepare corpus | Split Wylie documents into ≤512-token sequences | `python -m tibetian.cli.prepare_corpus` | `tibetian.preprocessing.prepare_corpus.prepare_corpus()` |
| 2. Continual | Continue pre-training mBERT on Tibetan | `python -m tibetian.cli.train_bert_continual` | `tibetian.model.pretrain_continual.continual_pretrain_mbert()` |

All paths (raw data, processed data, tokenizer, checkpoints) live in `tibetian.config`. CLI defaults write outputs under `results/<script>_results` under each script's parent directory.

### Minimal example

```bash
# 1) Prepare Wylie corpus into <=512-token sequences
python -m tibetian.cli.prepare_corpus \
  --input_dir cleaned_data/step3_wylie_converted \
  --tokenizer tibetan-tokenizer/tokenizer.json

# 2) Continual pre-train mBERT
python -m tibetian.cli.train_bert_continual --epochs 3
```

---

## Directory Structure

```
tibetian/
├── README.md
├── __init__.py
├── config.py
│
├── utils/
│   ├── logging.py
│   ├── tokenizer_utils.py
│   └── data_helpers.py
│
├── preprocessing/
│   ├── prepare_corpus.py
│   └── __init__.py
│
├── model/
│   ├── pretrain_continual.py
│   └── __init__.py
│
└── cli/
    ├── prepare_corpus.py
    └── train_bert_continual.py
```

---

## Notes

- Outputs (models, text files, images) created by CLI scripts are written into `results/<script>_results` in the script directory by default.
- Comments are in English.



