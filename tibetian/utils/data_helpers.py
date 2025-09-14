"""Dataset helpers for Tibetan Wylie plain-text corpora."""
from pathlib import Path

from datasets import Dataset, load_dataset

from .logging import get_logger


LOGGER = get_logger(__name__)


def load_text_dataset(path: Path | str, split: str = "train") -> Dataset:
    """Load a plain-text dataset from *path* via the ``text`` loader."""
    path = Path(path)
    LOGGER.info("Loading text dataset from %s", path)
    data = load_dataset("text", data_files={split: str(path)})
    LOGGER.info("Loaded %d records", len(data[split]))
    return data[split]





