"""
Sanskrit NLP Production Package

This package contains reusable preprocessing utilities, model–training
routines and CLI entry-points for building and fine-tuning language models
for Sanskrit (and, in the future, Tibetan).  Import modules such as
`sanscrit.preprocessing` or `sanscrit.model` in your scripts instead of
copy-pasting code.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "config",
    "utils",
    "preprocessing",
    "model",
]