"""Shared configuration and paths used across the package.

All important locations (raw data, processed data, tokenizer, model
checkpoints, etc.) are centralised here so that every sub-module imports
and stays in sync.  Update these paths once when moving folders and the
rest of the code will follow automatically.
"""
from pathlib import Path

# Base directory of the *package* (…/intelexsus/sanscrit)
BASE_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Data locations ----------------------------------------------------------------
# Raw JSONL documents (downloaded data)
RAW_DATA_DIR = BASE_DIR.parent / "data" / "sanskrit"

# Directory where intermediate preprocessing artefacts live
PROCESSED_DIR = BASE_DIR.parent / "data" / "sanskrit_preprocessed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Tokenizer / Model checkpoints -------------------------------------------------
# We keep model assets *inside* the `sanscrit` folder for ease of packaging.
TOKENIZER_PATH = BASE_DIR / "sanskrit-bert-tokenizer"
MODEL_OUTPUT_DIR = BASE_DIR / "sanskrit-bert-from-scratch"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Training / Experiment tracking ------------------------------------------------
WANDB_PROJECT_NAME = "Sanskrit-BERT-Pretraining"

# Default filenames inside PROCESSED_DIR
CHUNKS_FILE = PROCESSED_DIR / "sanskrit_chunks_final.txt"
BERT_READY_FILE = PROCESSED_DIR / "sanskrit_bert_ready_512.txt"

