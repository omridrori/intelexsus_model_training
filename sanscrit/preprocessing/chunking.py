"""Break Sanskrit raw documents into clean, fixed-length chunks.

This module is a **refactored** version of the original
`sandbox/sanskrit/preprocessing/chunking_preprocessor.py`, rewritten to be
import-friendly and reusable.  Usage:

>>> from sanscrit.preprocessing.chunking import run_preprocessing
>>> run_preprocessing()

It produces two artefacts inside ``sanscrit/../data/sanskrit_preprocessed``:
    1. ``sanskrit_chunks_final.txt`` – cleaned chunks, one per line.
    2. ``sanskrit_processing_report.txt`` – human-readable processing report.
"""
from __future__ import annotations

import datetime
import json
import re
import string
from pathlib import Path
from typing import List, Dict, Tuple

import nltk
from nltk.corpus import words as nltk_words
from tqdm import tqdm

from ..config import RAW_DATA_DIR, PROCESSED_DIR
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

# ---------------------------------------------------------------------------
# Settings (feel free to tweak from your own scripts) ------------------------

CHUNK_TARGET_WORDS_MIN = 40
ENGLISH_THRESHOLD = 0.5  # Prevent false positives like "Hare Krishna"
MAX_EXAMPLES_PER_CATEGORY = 5

OUTPUT_FILE = PROCESSED_DIR / "sanskrit_chunks_final.txt"
REPORT_FILE = PROCESSED_DIR / "sanskrit_processing_report.txt"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper classes / functions -------------------------------------------------

class ReportLogger:
    """Collect examples & write a markdown-like processing report."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.examples: Dict[str, List[dict]] = {}
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(f"Sanskrit Processing Report – {datetime.datetime.now()}\n")
            f.write("=" * 60 + "\n\n")

    # internal helper
    def _append(self, text: str) -> None:
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    # public helpers
    def header(self, title: str) -> None:
        self._append("\n" + "#" * 60)
        self._append(f"# {title.upper()}")
        self._append("#" * 60 + "\n")

    def log_example(self, category: str, data: dict) -> None:
        self.examples.setdefault(category, [])
        if len(self.examples[category]) < MAX_EXAMPLES_PER_CATEGORY:
            self.examples[category].append(data)

    def write_report(self, stats: dict) -> None:
        self.header("Final Statistics Summary")
        for key, value in stats.items():
            self._append(f"- {key.replace('_', ' ').title()}: {value:,}")

        self.header("Examples of Discarded Chunks")
        if "discarded_non_latin" in self.examples:
            self._append("\n--- REASON: Contains Non-Latin/IAST Characters ---\n")
            for ex in self.examples["discarded_non_latin"]:
                self._append(f"[ORIGINAL DOCUMENT CONTEXT]:\n{ex['original_doc'][:500]}...\n")
                self._append(f"[DISCARDED CHUNK]:\n{ex['chunk']}\n" + "-" * 40)

        if "discarded_english" in self.examples:
            self._append("\n--- REASON: High Percentage of English Words ---\n")
            for ex in self.examples["discarded_english"]:
                self._append(f"[ORIGINAL DOCUMENT CONTEXT]:\n{ex['original_doc'][:500]}...\n")
                self._append(f"[DISCARDED CHUNK]:\n{ex['chunk']}\n" + "-" * 40)

        self.header("Examples of Internal Cleaning")
        if "cleaning" in self.examples:
            for ex in self.examples["cleaning"]:
                self._append(f"[ORIGINAL DOCUMENT CONTEXT]:\n{ex['original_doc'][:500]}...\n")
                self._append(f"[CHUNK BEFORE CLEANING]:\n{ex['before']}\n")
                self._append(f"[CHUNK AFTER CLEANING]:\n{ex['after']}\n" + "-" * 40)

        LOGGER.info("Detailed report saved to: %s", self.file_path)

# ---------------------------------------------------------------------------
# Filtering helpers ----------------------------------------------------------

def _setup_nltk_words() -> set[str]:
    try:
        nltk.data.find("corpora/words")
    except LookupError:
        nltk.download("words", quiet=True)
    return set(nltk_words.words())

IAST_PATTERN = re.compile(r"[āīūṛṝḷḹṃḥṅñṭḍṇśṣĀĪŪṚṜḶḸṀḤṄÑṬḌṆŚṢ]")
IAST_CHARS = "āīūṛṝḷḹṃḥṅñṭḍṇśṣ"
ALLOWED_PUNCTUATION = ".,;!?()[]{}'\"-|=_/>"
ALLOWED_CHARS = set(string.ascii_lowercase + string.digits + ALLOWED_PUNCTUATION + IAST_CHARS + string.whitespace)


def _is_chunk_latin_only(chunk: str) -> bool:
    return all(char.lower() in ALLOWED_CHARS for char in chunk)


def _is_likely_english(word: str, english_vocab: set[str]) -> bool:
    if IAST_PATTERN.search(word):
        return False
    cleaned_word = word.strip(".,;!?()[]{}'\"-_")
    return cleaned_word.lower() in english_vocab

# ---------------------------------------------------------------------------
# Core logic -----------------------------------------------------------------


def _segment_documents_into_verses(all_docs: List[str]) -> List[dict]:
    """Break raw documents into individual verses (with context kept)."""
    all_verses: List[dict] = []
    for doc in tqdm(all_docs, desc="Segmenting documents into verses"):
        segment_pattern = r"\s*\|\|\s*|\s*//\s*|\||\."
        segmented_text = re.sub(segment_pattern, "\n", doc)
        verses = [v.strip() for v in segmented_text.splitlines() if v.strip()]
        for verse in verses:
            all_verses.append({"text": verse, "original_doc": doc})
    return all_verses


def _group_verses_into_chunks(verses: List[dict]) -> List[dict]:
    chunks: List[dict] = []
    current_chunk_verses: List[dict] = []
    current_word_count = 0
    for verse_data in tqdm(verses, desc="Grouping verses into chunks"):
        verse_words = verse_data["text"].split()
        if not verse_words:
            continue
        if current_word_count >= CHUNK_TARGET_WORDS_MIN:
            chunk_text = " ".join(v["text"] for v in current_chunk_verses)
            chunks.append({"text": chunk_text, "original_doc": current_chunk_verses[0]["original_doc"]})
            current_chunk_verses = []
            current_word_count = 0
        current_chunk_verses.append(verse_data)
        current_word_count += len(verse_words)
    # flush last chunk
    if current_chunk_verses:
        chunk_text = " ".join(v["text"] for v in current_chunk_verses)
        chunks.append({"text": chunk_text, "original_doc": current_chunk_verses[0]["original_doc"]})
    return chunks


def _process_chunks(chunks: List[dict], english_vocab: set[str], logger: ReportLogger) -> Tuple[List[str], dict]:
    stats = {k: 0 for k in [
        "total",
        "discarded_non_latin",
        "discarded_english",
        "discarded_url",
        "kept",
    ]}
    stats["total"] = len(chunks)

    final_chunks: List[str] = []
    for chunk_data in tqdm(chunks, desc="Filtering and Cleaning Chunks"):
        chunk = chunk_data["text"]
        original_doc = chunk_data["original_doc"]

        # 1) filter non-Latin characters (except IAST)
        if not _is_chunk_latin_only(chunk):
            stats["discarded_non_latin"] += 1
            logger.log_example("discarded_non_latin", {"chunk": chunk, "original_doc": original_doc})
            continue
        # 2) filter URLs
        if re.search(r"https?://\S+", chunk):
            stats["discarded_url"] += 1
            continue
        # 3) English ratio
        words = chunk.split()
        eng_count = sum(1 for w in words if _is_likely_english(w, english_vocab))
        if words and (eng_count / len(words)) > ENGLISH_THRESHOLD:
            stats["discarded_english"] += 1
            logger.log_example("discarded_english", {"chunk": chunk, "original_doc": original_doc})
            continue

        # 4) Cleaning regex passes
        before_clean = chunk
        cleaned = chunk.lower()
        cleaned = re.sub(r"\b\d+(\.\d+)*\b", "", cleaned)  # multi-part numeric refs
        cleaned = re.sub(r"\b[a-zāīūṛṝḷḹṃḥṅñṭḍṇśṣ]+\w*_\d+\s*//?", "", cleaned)
        cleaned = re.sub(r"\b\d+\b", "", cleaned)
        cleaned = re.sub(r"\(\d+\)", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if cleaned and len(cleaned.split()) > 1:
            final_chunks.append(cleaned)
            stats["kept"] += 1
            logger.log_example("cleaning", {"before": before_clean, "after": cleaned, "original_doc": original_doc})

    return final_chunks, stats

# ---------------------------------------------------------------------------
# Public API -----------------------------------------------------------------


def run_preprocessing(
    raw_data_path: Path | None = None,
    output_file: Path | None = None,
    report_file: Path | None = None,
) -> None:
    """Run end-to-end preprocessing pipeline and save artefacts."""
    raw_data_path = raw_data_path or RAW_DATA_DIR
    output_file = output_file or OUTPUT_FILE
    report_file = report_file or REPORT_FILE

    logger = ReportLogger(report_file)
    logger.header("Initialization")
    logger._append(f"Raw data path: {raw_data_path}")
    logger._append(f"Final output file: {output_file}")

    english_vocab = _setup_nltk_words()

    # 1) Load raw documents --------------------------------------------------
    all_docs: List[str] = []
    all_files = list(raw_data_path.rglob("*.jsonl"))
    for file_path in tqdm(all_files, desc="Loading raw documents"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    text = json.loads(line).get("text", "")
                    if text.strip():
                        all_docs.append(text)
                except json.JSONDecodeError:
                    continue

    # 2) Segment & chunk -----------------------------------------------------
    verses = _segment_documents_into_verses(all_docs)
    initial_chunks = _group_verses_into_chunks(verses)
    final_clean_chunks, stats = _process_chunks(initial_chunks, english_vocab, logger)

    # 3) Save ----------------------------------------------------------------
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in final_clean_chunks:
            f.write(chunk + "\n")

    stats["documents_loaded"] = len(all_docs)
    stats["verses_extracted"] = len(verses)
    logger.write_report(stats)
    LOGGER.info("Wrote %d cleaned chunks to %s", len(final_clean_chunks), output_file)

# ---------------------------------------------------------------------------
# CLI helper -----------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    run_preprocessing()