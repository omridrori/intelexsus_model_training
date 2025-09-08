#!/usr/bin/env python
"""Train a BPE tokenizer on Tibetan Wylie corpus (keeps + / . ' etc.)

Usage (from project root):
    python sandbox/tibetian/train_tokenizer_bpe.py \
           --input_dir cleaned_data/step3_wylie_converted \
           --output_path tibetan-tokenizer-bpe/tokenizer.json \
           --vocab_size 30000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import BpeTrainer
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Helpers --------------------------------------------------------------------
# -----------------------------------------------------------------------------

def iter_texts(data_dir: Path) -> Iterator[str]:
    for fn in sorted(data_dir.glob("*.jsonl")):
        with fn.open(encoding="utf-8") as f:
            for line in f:
                try:
                    txt = json.loads(line).get("text", "").strip()
                    if txt:
                        yield txt
                except json.JSONDecodeError:
                    continue


def count_lines(data_dir: Path) -> int:
    total = 0
    for fn in data_dir.glob("*.jsonl"):
        with fn.open(encoding="utf-8") as f:
            for _ in f:
                total += 1
    return total

# -----------------------------------------------------------------------------
# Main -----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tibetan BPE tokenizer")
    parser.add_argument("--input_dir", default="cleaned_data/step3_wylie_converted")
    parser.add_argument("--output_path", default="tibetan-tokenizer-bpe/tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=30000)
    args = parser.parse_args()

    data_dir = Path(args.input_dir)
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    # Init tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        initial_alphabet=list("'+.~")
    )

    total = count_lines(data_dir)
    print(f"Total lines: {total:,}. Starting BPE training (vocab {args.vocab_size}) …")

    tokenizer.train_from_iterator(iter_texts(data_dir), trainer=trainer, length=total)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    print(f"Tokenizer saved to {out_path}")

    # quick demo
    demo = "oM badz+ra sat+wa hU~M"
    enc = tokenizer.encode(demo)
    print("Demo:", demo)
    print("Tokens:", enc.tokens)


if __name__ == "__main__":
    main()
