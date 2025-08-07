"""Lightweight logging helper that also plays nicely with tqdm progress bars."""
import logging
from typing import Optional

from tqdm.auto import tqdm  # ensures shiny progress bars in notebooks & terminals


class TqdmLoggingHandler(logging.Handler):
    """A logging handler that redirects logs through ``tqdm.write``.

    When you have active progress bars, plain ``print`` or standard logging ruins
    formatting.  This handler makes log messages appear *above* the progress
    bars instead.
    """

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:  # pragma: no cover
            # Fallback: if tqdm is somehow broken, just print normally.
            print(record.getMessage())


def get_logger(name: str = "sanscrit", level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger initialised once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Only configure first time.
        handler: logging.Handler = TqdmLoggingHandler()
        fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
