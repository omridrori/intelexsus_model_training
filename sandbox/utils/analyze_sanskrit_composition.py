import json
import re
import string
from pathlib import Path
from tqdm import tqdm
import nltk
from nltk.corpus import words
import argparse

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "sanskrit"

# --- Character and Word Definitions (from universal_preprocessor.py) ---

# This ensures the analysis uses the exact same logic as the real preprocessor.
def setup_nltk_words():
    """
    Checks for NLTK 'words' corpus and downloads it if missing.
    """
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        print("NLTK 'words' corpus not found. Downloading...")
        nltk.download('words', quiet=True)
    return set(words.words())

# Pattern to find any IAST character
IAST_PATTERN = re.compile(r"[āīūṛṝḷḹṃḥṅñṭḍṇśṣĀĪŪṚṜḶḸṀḤṄÑṬḌṆŚṢ]")

def is_likely_english(word: str, english_vocab: set) -> bool:
    """
    Determines if a word is likely English. A word is NOT English if it contains
    any IAST characters. Otherwise, it's checked against the NLTK vocabulary.
    """
    if IAST_PATTERN.search(word):
        return False
    # Clean punctuation from word edges before checking
    cleaned_word = word.strip(string.punctuation + "–—_")
    return cleaned_word.lower() in english_vocab

# Definition of allowed characters for a "clean" line (Latin/IAST only)
ALLOWED_PUNCTUATION = ".,;!?()[]{}'\"-|=_/" 
ALLOWED_CHARS = set(
    string.ascii_letters +
    string.digits +
    ALLOWED_PUNCTUATION +
    "āīūṛṝḷḹṃḥṅñṭḍṇśṣĀĪŪṚṜḶḸṀḤṄÑṬḌṆŚṢ" + # Added uppercase IAST for completeness
    string.whitespace
)

def is_verse_latin_only(verse: str) -> bool:
    """
    Checks if all characters in a verse are within the allowed set.
    """
    return all(char in ALLOWED_CHARS for char in verse)

# --- Main Analysis Pipeline ---

def analyze_composition(english_threshold: float):
    """
    Main function to run the full analysis pipeline.
    """
    print("--- Starting Sanskrit Corpus Composition Analysis ---")
    
    # --- 1. Load all documents into a single text block ---
    all_files = list(RAW_DATA_PATH.glob("**/*.jsonl"))
    if not all_files:
        print(f"Error: No raw .jsonl files found in '{RAW_DATA_PATH}'.")
        return

    print(f"\nStep 1: Loading raw data from {len(all_files)} files...")
    full_text_content = []
    for file_path in tqdm(all_files, desc="Loading files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    full_text_content.append(json.loads(line).get("text", ""))
                except json.JSONDecodeError:
                    continue
    
    # --- 2. Segment into verses ---
    print("\nStep 2: Segmenting documents into verses...")
    # Join all documents, then unify delimiters ('//' to '||'), then split
    unified_text = " || ".join(doc for doc in full_text_content if doc)
    unified_text = unified_text.replace('//', '||')
    all_verses = [verse.strip() for verse in unified_text.split('||') if verse.strip()]
    total_verses = len(all_verses)
    print(f"-> Found {total_verses:,} total verses.")

    # --- 3. Classify verses (Latin vs. Non-Latin) ---
    print("\nStep 3: Classifying verses (Latin vs. Non-Latin)...")
    latin_verses = []
    non_latin_verses = []
    for verse in tqdm(all_verses, desc="Classifying verses"):
        if is_verse_latin_only(verse):
            latin_verses.append(verse)
        else:
            non_latin_verses.append(verse)
            
    # --- 4. Sub-classify Latin verses (English vs. Sanskrit) ---
    print("\nStep 4: Analyzing Latin verses for English content...")
    english_vocab = setup_nltk_words()
    english_verses = []
    sanskrit_verses = []
    
    for verse in tqdm(latin_verses, desc="Analyzing for English"):
        words_in_verse = verse.split()
        if not words_in_verse:
            sanskrit_verses.append(verse) # Empty verses are not English
            continue
        
        english_word_count = sum(1 for w in words_in_verse if is_likely_english(w, english_vocab))
        
        # Check if the ratio of English words meets the threshold
        if (english_word_count / len(words_in_verse)) >= english_threshold:
            english_verses.append(verse)
        else:
            sanskrit_verses.append(verse)

    # --- 5. Generate Final Report ---
    num_non_latin = len(non_latin_verses)
    num_latin = len(latin_verses)
    num_english = len(english_verses)
    num_sanskrit = len(sanskrit_verses)
    final_kept = num_sanskrit

    perc_non_latin = (num_non_latin / total_verses) * 100 if total_verses > 0 else 0
    perc_latin = (num_latin / total_verses) * 100 if total_verses > 0 else 0
    
    # Percentages for the sub-analysis are relative to the 'latin_verses' pool
    perc_english = (num_english / num_latin) * 100 if num_latin > 0 else 0
    perc_sanskrit_in_latin = (num_sanskrit / num_latin) * 100 if num_latin > 0 else 0
    perc_final_kept = (final_kept / total_verses) * 100 if total_verses > 0 else 0

    print("\n\n--- Sanskrit Corpus Composition Report ---")
    print("=" * 60)
    print(f"Total Verses Identified: {total_verses:,}")
    print("=" * 60)

    print("\n[Phase 1: Script-Based Filtering]")
    print(f"  - Non-Latin Verses (Discarded): {num_non_latin:>9,} ({perc_non_latin:>5.1f}%)")
    print(f"  - Latin/IAST Verses (Kept):     {num_latin:>9,} ({perc_latin:>5.1f}%)")
    
    print("\n[Phase 2: English Language Filtering (within Latin/IAST verses)]")
    print(f"  - Total Latin/IAST Verses Analyzed: {num_latin:,}")
    print(f"    - Likely English (Discarded):     {num_english:>9,} ({perc_english:>5.1f}%)")
    print(f"    - Likely Sanskrit (Kept):         {num_sanskrit:>9,} ({perc_sanskrit_in_latin:>5.1f}%)")

    print("\n--- Final Summary ---")
    print(f"Total verses remaining after all filtering: {final_kept:,}")
    print(f"Overall percentage of original verses kept: {perc_final_kept:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the composition of the Sanskrit corpus based on script and language.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--english-threshold", 
        type=float, 
        default=0.7, 
        help="Threshold of English words to classify a verse as English."
    )
    args = parser.parse_args()
    
    analyze_composition(english_threshold=args.english_threshold)
