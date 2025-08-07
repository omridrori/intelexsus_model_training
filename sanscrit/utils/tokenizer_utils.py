"""Helpers for loading (and eventually training) tokenizers."""
from pathlib import Path

from transformers import BertTokenizerFast

from .logging import get_logger
from ..config import TOKENIZER_PATH

LOGGER = get_logger(__name__)


def load_tokenizer(tokenizer_path: Path | str = TOKENIZER_PATH) -> BertTokenizerFast:
    """Load a *fast* tokenizer from *tokenizer_path* (default: shared config)."""
    tokenizer_path = Path(tokenizer_path)
    LOGGER.info("Loading tokenizer from %s", tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_path))
    LOGGER.info("Loaded tokenizer with vocab size %d", tokenizer.vocab_size)
    return tokenizer
