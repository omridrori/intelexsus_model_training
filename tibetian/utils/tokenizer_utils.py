"""Helpers for loading Tibetan tokenizers (HF or JSON tokenizers)."""
from pathlib import Path

from tokenizers import Tokenizer
from transformers import BertTokenizerFast

from .logging import get_logger
from ..config import TOKENIZER_PATH


LOGGER = get_logger(__name__)


def load_hf_tokenizer(tokenizer_path: Path | str = TOKENIZER_PATH) -> BertTokenizerFast:
    """Load a fast tokenizer from a directory with HF files (vocab.txt, etc.)."""
    tokenizer_path = Path(tokenizer_path)
    LOGGER.info("Loading HF tokenizer from %s", tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    # Prefer from_pretrained; fallback to direct vocab if only vocab.txt exists
    try:
        tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_path), lowercase=False)
    except Exception:
        tokenizer = BertTokenizerFast(vocab_file=str(tokenizer_path / "vocab.txt"), lowercase=False)
    LOGGER.info("Loaded tokenizer with vocab size %d", tokenizer.vocab_size)
    return tokenizer


def load_json_tokenizer(json_path: Path | str) -> Tokenizer:
    """Load a `tokenizers` JSON model (e.g., BPE)."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    LOGGER.info("Loading JSON tokenizer from %s", json_path)
    return Tokenizer.from_file(str(json_path))





