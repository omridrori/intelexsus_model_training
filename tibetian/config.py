from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Data locations ----------------------------------------------------------------
# Raw JSONL documents (downloaded data)
RAW_DATA_DIR = BASE_DIR.parent / "data" / "tibetan"

# Directory where intermediate preprocessing artefacts live
PROCESSED_DIR = BASE_DIR.parent / "data" / "tibetan_preprocessed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Tokenizer / Model checkpoints -------------------------------------------------
# We keep model assets inside the package for ease of packaging.
TOKENIZER_PATH = BASE_DIR.parent / "tibetan-tokenizer"
MODEL_OUTPUT_DIR = BASE_DIR.parent / "mbert-tibetan-continual"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Training / Experiment tracking ------------------------------------------------
WANDB_PROJECT_NAME_CONTINUAL = "Tibetan-mBERT-Continual"

# Default filenames inside PROCESSED_DIR
BERT_READY_FILE = BASE_DIR.parent / "tibetan_bert_ready_512_wylie.txt"





