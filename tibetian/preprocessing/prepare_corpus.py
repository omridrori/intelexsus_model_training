"""Prepare Tibetan Wylie corpus for BERT continual pre-training.

Reads all *.jsonl files in a directory (each line → document), splits each
document into "sentences" using Tibetan shad markers (// and /) or newlines,
and emits sequences up to 512 tokens that are document-contiguous and do not
break individual sentences.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List

from tqdm.auto import tqdm

from ..config import BERT_READY_FILE
from ..utils.logging import get_logger
from ..utils.tokenizer_utils import load_json_tokenizer


LOGGER = get_logger(__name__)

SENT_RE = re.compile(r"//|/|\n")


def iter_documents(data_dir: Path) -> Iterable[str]:
    """Yield raw Wylie text for every document (line) in all jsonl files."""
    files = sorted(p for p in data_dir.glob("*.jsonl"))
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                try:
                    text = json.loads(line).get("text", "").strip()
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue


def split_sentences(doc: str) -> List[str]:
    """Split document on Tibetan shad markers and newlines; keep non-empty."""
    parts = [p.strip() for p in SENT_RE.split(doc) if p.strip()]
    return parts or [doc]


def build_sequences(sentences: List[str], tokenizer, max_len: int) -> List[str]:
    """Combine sentences into ≤max_len token strings without crossing docs."""
    sequences: List[str] = []
    current_tokens: List[str] = []
    current_len = 0

    for sent in sentences:
        tokens = tokenizer.encode(sent).tokens
        if len(tokens) > max_len:
            # long single sentence; split hard at max_len
            for i in range(0, len(tokens), max_len):
                chunk = tokens[i : i + max_len]
                if chunk:
                    sequences.append(" ".join(chunk))
            current_tokens = []
            current_len = 0
            continue

        if current_len + len(tokens) <= max_len:
            current_tokens.extend(tokens)
            current_len += len(tokens)
        else:
            # emit current sequence
            sequences.append(" ".join(current_tokens))
            current_tokens = list(tokens)
            current_len = len(tokens)

    if current_tokens:
        sequences.append(" ".join(current_tokens))
    return sequences


def prepare_corpus(
    input_dir: Path,
    tokenizer_json: Path,
    output_file: Path | None = None,
    max_len: int = 512,
) -> Path:
    """Prepare corpus into ≤512-token sequences and write to a text file."""
    output_path = Path(output_file or BERT_READY_FILE)
    tokenizer = load_json_tokenizer(tokenizer_json)

    sequences_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for doc in tqdm(iter_documents(input_dir), desc="Processing documents"):
            sent_list = split_sentences(doc)
            seqs = build_sequences(sent_list, tokenizer, max_len)
            for seq in seqs:
                out_f.write(seq + "\n")
                sequences_written += 1

    LOGGER.info("Wrote %d sequences to %s", sequences_written, output_path)
    return output_path





