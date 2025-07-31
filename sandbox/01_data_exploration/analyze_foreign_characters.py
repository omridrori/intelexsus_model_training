import json
from pathlib import Path
import re
import string
import time
from collections import Counter
from tqdm import tqdm
import argparse

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# This is our strict definition of what's allowed. Everything else is "foreign".
STRICT_IAST_CHARS = "āīūṛṝḷḹṃḥṅñṭḍṇśṣĀĪŪṚṜḶḸṀḤṄÑṬḌṆŚṢ"
ALLOWED_CHARS = set(
    string.ascii_letters + string.digits + string.punctuation + string.whitespace + STRICT_IAST_CHARS
)

def contains_foreign_char(word: str) -> bool:
    """Checks if a word contains any character not in the allowed set."""
    # Strip punctuation from the word before checking, but check the original word characters
    # This avoids issues where a valid word is flagged due to trailing punctuation.
    for char in word:
        if char not in ALLOWED_CHARS:
            return True
    return False

def analyze_corpus(corpus_name: str, files: list[Path]):
    """
    Analyzes a specific corpus for words containing foreign characters.
    1. Counts the occurrences of each foreign word.
    2. Extracts context for each unique foreign word.
    3. Prints a report and saves the context.
    """
    print(f"--- Analyzing corpus: {corpus_name} ---")
    if not files:
        print(f"No .jsonl files found for '{corpus_name}'. Skipping.")
        return

    foreign_word_counts = Counter()
    word_context_samples = {}  # To store one sample per unique foreign word

    for file_path in tqdm(files, desc=f"Analyzing {corpus_name} files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            prev_text = "[START]"
            curr_str = None
            
            try:
                curr_str = next(f)
            except StopIteration:
                continue # Empty file, go to the next one

            for next_str in f:
                # --- Process the 'current' line ---
                try:
                    curr_data = json.loads(curr_str)
                    curr_text = curr_data.get("text", "")

                    # Check for foreign words in the current line's text
                    if curr_text:
                        words = curr_text.split()
                        for word in words:
                            cleaned_word = word.strip(string.punctuation)
                            if cleaned_word and contains_foreign_char(cleaned_word):
                                foreign_word_counts[cleaned_word] += 1
                                # If it's a new foreign word, get its context
                                if cleaned_word not in word_context_samples:
                                    # We need the text from the next line for context
                                    next_text = json.loads(next_str).get("text", "[INVALID JSON]")
                                    word_context_samples[cleaned_word] = (prev_text, curr_text, next_text)
                
                except json.JSONDecodeError:
                    # If current line is broken, we can't process it.
                    # Its text will be '[INVALID JSON]' for the next iteration's 'prev_text'
                    curr_text = "[INVALID JSON]"

                # --- Slide the window for the next iteration ---
                prev_text = curr_text
                curr_str = next_str

            # --- Process the very last line of the file after the loop finishes ---
            try:
                curr_data = json.loads(curr_str)
                curr_text = curr_data.get("text", "")
                if curr_text:
                    words = curr_text.split()
                    for word in words:
                        cleaned_word = word.strip(string.punctuation)
                        if cleaned_word and contains_foreign_char(cleaned_word):
                            foreign_word_counts[cleaned_word] += 1
                            if cleaned_word not in word_context_samples:
                                # For the last line, there is no 'next' line
                                word_context_samples[cleaned_word] = (prev_text, curr_text, "[END]")
            except json.JSONDecodeError:
                # Ignore if the last line is broken
                pass

    # --- 1. Generate Report Table ---
    print(f"\n\n--- Foreign Word Analysis for '{corpus_name.upper()}' Corpus ---")
    total_unique_words = len(foreign_word_counts)
    if total_unique_words == 0:
        print("No foreign words found.")
        print("-" * 60)
        return

    print(f"Total unique foreign words found: {total_unique_words:,}")
    print("-" * 60)
    print(f"{'Word':<40} | {'Count':>12}")
    print("-" * 60)
    
    for word, count in foreign_word_counts.most_common(25):  # Print top 25 most common
        print(f"{word:<40} | {count:>12,}")
    print("-" * 60)

    # --- 2. Generate Context File ---
    context_file = OUTPUT_DIR / f"{corpus_name}_foreign_word_context.txt"
    print(f"\nWriting context samples for {len(word_context_samples)} unique words to {context_file}...")
    with open(context_file, 'w', encoding='utf-8') as f:
        # Sort for consistent output
        for word, (prev_s, curr_s, next_s) in sorted(word_context_samples.items()):
            f.write(f"--- Context for Word: '{word}' ---\n")
            f.write(f"  [Previous]: {prev_s.strip()}\n")
            f.write(f"  [Current]:  {curr_s.strip()}\n")
            f.write(f"  [Next]:     {next_s.strip()}\n")
            f.write("-" * 50 + "\n\n")
    
    print(f"--- Analysis complete for corpus: {corpus_name} ---\n")

def run_full_analysis(test_run: bool = False):
    """
    Main function to run the analysis on all predefined corpora.
    It assumes data is structured in subdirectories like 'data/sanskrit' and 'data/tibetan'.
    If test_run is True, it will only process a small subset of files.
    """
    print("--- Starting deep analysis of foreign words in corpora... ---")
    if test_run:
        print("\n--- RUNNING IN TEST MODE: Processing only a small subset of files. ---\n")
    start_time = time.time()
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # We assume a directory structure like 'data/sanskrit', 'data/tibetan'
    sanskrit_files = list((DATA_ROOT / "sanskrit").glob("**/*.jsonl"))
    tibetan_files = list((DATA_ROOT / "tibetan").glob("**/*.jsonl"))
    
    if test_run:
        sanskrit_files = sanskrit_files[:3]
        tibetan_files = tibetan_files[:3]
        
    # Fallback for when files are not in subdirectories
    if not sanskrit_files and not tibetan_files:
        print("\nWarning: No .jsonl files found in 'data/sanskrit/' or 'data/tibetan/'.")
        all_files = list(DATA_ROOT.glob("**/*.jsonl"))
        if all_files:
            print("Attempting to run analysis on all found .jsonl files in 'data/' as a single corpus.")
            analyze_corpus("all_data", all_files)
        else:
            print("Error: No .jsonl files found anywhere in the 'data/' directory. Exiting.")
        return

    if sanskrit_files:
        analyze_corpus("sanskrit", sanskrit_files)
    if tibetan_files:
        analyze_corpus("tibetan", tibetan_files)

    end_time = time.time()
    print(f"Full analysis complete in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze corpora for words with foreign characters.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode on a small subset of files."
    )
    args = parser.parse_args()
    
    run_full_analysis(test_run=args.test) 