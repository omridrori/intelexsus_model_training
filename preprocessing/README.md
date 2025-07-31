# Corpus Preprocessing Pipeline

This directory contains the universal script for cleaning, normalizing, and preparing the raw corpus data for model training.

## `universal_preprocessor.py`

This is a modular script that performs a series of optional preprocessing steps on the raw `.jsonl` data. It allows for the creation of different versions of the corpus by enabling or disabling steps via command-line arguments.

### How It Works

The script reads all `.jsonl` files from the `data/sanskrit` directory and processes them through a user-defined pipeline. The output is a single, clean `.txt` file with a dynamic name reflecting the steps taken (e.g., `sanskrit_en-filtered_latin-only_verses.txt`).

### Pipeline Steps & Arguments

The processing steps are applied in a logical order to ensure data integrity. You can combine any of the following flags:

1.  `--filter-english`:
    -   **What it does:** Removes documents where the ratio of likely English words exceeds a threshold.
    -   **Why:** To filter out noise like log files, metadata, and English commentary that contaminated the raw data.
    -   **Threshold:** Can be adjusted with `--english-threshold` (default is 0.7).

2.  `--segment-verses`:
    -   **What it does:** Segments the corpus into individual verses (ślokas). It unifies `//` and `||` as delimiters and splits the text accordingly.
    -   **Why:** To create distinct training examples, which is ideal for models like BERT that operate on shorter sequences.

3.  `--filter-non-latin`:
    -   **What it does:** After segmentation, it removes any verse containing characters outside of the allowed Latin/IAST character set (e.g., it removes Devanagari script).
    -   **Why:** To ensure the entire corpus is in a single, consistent script (IAST transliteration) for the tokenizer and model.

4.  `--normalize`:
    -   **What it does:** Applies final cleaning rules: converts all text to lowercase, removes URLs, and strips editorial identifiers (e.g., `wal_1`).
    -   **Why:** To reduce vocabulary size and remove non-linguistic noise, helping the model focus on the language itself.

5.  `--diagnostic`:
    -   **What it does:** Creates log files (`log_removed_english.txt`, `log_removed_non_latin.txt`) that contain the exact lines/verses removed during the filtering steps.
    -   **Why:** For debugging and verifying that the filters are not accidentally removing valid data.

### Example Usage

To run the full recommended pipeline and generate a clean, verse-segmented, Latin-only corpus:
```bash
python preprocessing/universal_preprocessor.py --filter-english --segment-verses --filter-non-latin --normalize
``` 