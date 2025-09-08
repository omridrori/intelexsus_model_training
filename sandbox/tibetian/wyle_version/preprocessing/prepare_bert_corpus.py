#!/usr/bin/env python
"""Prepare Tibetan Wylie corpus for BERT continual pre-training.

Reads all *.jsonl files in a directory (each line → document), splits each
 document into "sentences" using Tibetan shad markers (// and /) or newlines,
 and emits sequences up to 512 tokens that are *document-contiguous* and do
 not break individual sentences.

Output: a plain-text file where each line is a chunk (≤512 tokens) encoded back
 to space-separated tokens.  This is what `datasets.load_dataset("text")` needs.

Usage
-----
python sandbox/tibetian/prepare_bert_corpus.py \
       --input_dir cleaned_data/step3_wylie_converted \
       --tokenizer tibetan-tokenizer/tokenizer.json \
       --max_len 512 \
       --output_file data/tibetan_bert_ready_512.txt
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer
from tqdm.auto import tqdm

SENT_RE = re.compile(r"//|/|\n")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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


def build_sequences(sentences: List[str], tokenizer: Tokenizer, max_len: int) -> List[str]:
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Tibetan BERT corpus")
    parser.add_argument("--input_dir", default="cleaned_data/step3_wylie_converted")
    parser.add_argument("--tokenizer", default="tibetan-tokenizer/tokenizer.json")
    parser.add_argument("--output_file", default="data/tibetan_bert_ready_512.txt")
    parser.add_argument("--max_len", type=int, default=512)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    tok_path = Path(args.tokenizer)
    out_path = Path(args.output_file)

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    if not tok_path.exists():
        raise FileNotFoundError(tok_path)

    tokenizer = Tokenizer.from_file(str(tok_path))

    sequences_written = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for doc in tqdm(iter_documents(input_dir), desc="Processing documents"):
            sent_list = split_sentences(doc)
            seqs = build_sequences(sent_list, tokenizer, args.max_len)
            for seq in seqs:
                out_f.write(seq + "\n")
                sequences_written += 1

    print(f"Done. Wrote {sequences_written:,} sequences to {out_path}")


if __name__ == "__main__":
    main()
