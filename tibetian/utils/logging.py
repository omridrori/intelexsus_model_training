"""Lightweight logging helper compatible with tqdm progress bars."""
import logging
from typing import Optional

from tqdm.auto import tqdm


class TqdmLoggingHandler(logging.Handler):
    """A logging handler that uses ``tqdm.write`` to avoid breaking progress bars."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            print(record.getMessage())


def get_logger(name: str = "tibetian", level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger initialised once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler: logging.Handler = TqdmLoggingHandler()
        fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger





