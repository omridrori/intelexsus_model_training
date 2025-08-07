"""Pack cleaned textual chunks into ≤512-token sequences.

This utility loads the plain-text chunks produced by
`sanscrit.preprocessing.chunking.run_preprocessing`, encodes each chunk with a
BERT tokenizer, and concatenates consecutive chunks until the combined token
length (without special tokens) would exceed *max_tokens* (default 510).  The
resulting packed sequences are written *one per line* to `config.BERT_READY_FILE`.


"""
from __future__ import annotations

from pathlib import Path
from typing import List

from tqdm.auto import tqdm

from ..config import BERT_READY_FILE, CHUNKS_FILE, TOKENIZER_PATH
from ..utils.logging import get_logger
from ..utils.tokenizer_utils import load_tokenizer

LOGGER = get_logger(__name__)


def _token_len(text: str, tokenizer) -> int:  # helper
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_length=True,
        )["length"]
    )


def pack_chunks_to_sequences(
    chunks_input: Path | str | None = None,
    output_file: Path | str | None = None,
    tokenizer_path: Path | str | None = None,
    max_tokens: int = 510,
) -> None:
    """Main packing routine.

    Parameters
    ----------
    chunks_input: Path to input file with cleaned chunks (one per line).
                 Defaults to `config.CHUNKS_FILE`.
    output_file: Path where packed sequences will be written.
                 Defaults to `config.BERT_READY_FILE`.
    tokenizer_path: Which tokenizer to use for counting tokens.
                    Defaults to `config.TOKENIZER_PATH`.
    max_tokens: Maximum token length *before* adding special tokens.
    """
    chunks_input = Path(chunks_input or CHUNKS_FILE)
    output_file = Path(output_file or BERT_READY_FILE)
    tokenizer_path = Path(tokenizer_path or TOKENIZER_PATH)

    LOGGER.info("Packing chunks from %s → %s", chunks_input, output_file)

    tokenizer = load_tokenizer(tokenizer_path)

    # ------------------------------------------------------------------
    packed_sequences: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    with open(chunks_input, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Packing", unit="chunks"):
            chunk = line.strip()
            if not chunk:
                continue
            chunk_len = _token_len(chunk, tokenizer)

            # skip chunks that on their own exceed the limit (rare)
            if chunk_len > max_tokens:
                LOGGER.warning("Skipping over-length chunk (%d tokens)", chunk_len)
                continue

            if current_len + chunk_len <= max_tokens:
                current_parts.append(chunk)
                current_len += chunk_len
            else:
                # flush current buffer
                packed_sequences.append(" ".join(current_parts))
                current_parts = [chunk]
                current_len = chunk_len

    # flush any remainder
    if current_parts:
        packed_sequences.append(" ".join(current_parts))

    # ------------------------------------------------------------------
    with open(output_file, "w", encoding="utf-8") as f:
        for seq in packed_sequences:
            f.write(seq + "\n")

    LOGGER.info("Wrote %d packed sequences", len(packed_sequences))


if __name__ == "__main__":  # pragma: no cover
    pack_chunks_to_sequences()