import argparse
import json
import re
import string
from pathlib import Path

import nltk
import numpy as np
from nltk.corpus import words
from tqdm import tqdm

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "sanskrit"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# --- Helper Functions (from previous scripts) ---

def setup_nltk_words():
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        print("NLTK 'words' corpus not found. Downloading...")
        nltk.download('words', quiet=True)
    return set(words.words())

IAST_PATTERN = re.compile(r"[āīūṛṝḷḹṃḥṅñṭḍṇśṣĀĪŪṚṜḶḸṀḤṄÑṬḌṆŚṢ]")
def is_likely_english(word: str, english_vocab: set) -> bool:
    if IAST_PATTERN.search(word):
        return False
    cleaned_word = word.strip(".,;!?()[]{}'\"-_")
    return cleaned_word.lower() in english_vocab

IAST_CHARS = "āīūṛṝḷḹṃḥṅñṭḍṇśṣ"
# Note: The user wants to keep '//' and '_', so we add them to punctuation.
# Let's define the exact punctuation we want to keep, instead of the broad string.punctuation
ALLOWED_PUNCTUATION = ".,;!?()[]{}'\"-|=_/" 

ALLOWED_CHARS = set(
    string.ascii_lowercase +  # We work with lowercase text
    string.digits +
    ALLOWED_PUNCTUATION +
    IAST_CHARS +
    string.whitespace
)
def is_line_latin_only(line: str) -> bool:
    for char in line.lower():
        if char not in ALLOWED_CHARS:
            return False
    return True

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\b[a-zāīūṛṝḷḹṃḥṅñṭḍṇśṣ]+\w*_\d+\b', '', text)
    text = re.sub(r'\(\d+\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main Processing Pipeline ---

def main():
    parser = argparse.ArgumentParser(
        description="A universal preprocessor for the Sanskrit corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--filter-english", action="store_true", help="Filter out lines with a high ratio of English words.")
    parser.add_argument("--english-threshold", type=float, default=0.7, help="Threshold for filtering English lines.")
    parser.add_argument("--filter-non-latin", action="store_true", help="Filter out lines with non-Latin/IAST characters (e.g., Devanagari).")
    parser.add_argument("--normalize", action="store_true", help="Apply text normalization (lowercase, remove IDs, etc.).")
    parser.add_argument("--segment-verses", action="store_true", help="Segment the corpus into verses based on '||' and '//'.")
    parser.add_argument("--diagnostic", action="store_true", help="Save logs of removed lines for each filtering step.")
    
    args = parser.parse_args()
    
    # --- 1. Dynamic Output Filename ---
    output_filename_parts = ["sanskrit"]
    if args.filter_english:
        output_filename_parts.append("en-filtered")
    if args.filter_non_latin:
        output_filename_parts.append("latin-only")
    if args.normalize:
        output_filename_parts.append("normalized")
    if args.segment_verses:
        output_filename_parts.append("verses")
    
    if len(output_filename_parts) == 1: # No flags used
        output_filename_parts.append("raw_concatenated")
        
    output_filename = "_".join(output_filename_parts) + ".txt"
    output_file_path = OUTPUT_DIR / output_filename

    print("--- Universal Sanskrit Preprocessor ---")
    print(f"Active steps: {', '.join(output_filename_parts[1:]) or 'None'}")
    if args.diagnostic:
        print("Diagnostic mode: ON (logs will be saved)")
    print(f"Output will be saved to: {output_file_path}")
    
    # --- 2. Initial Data Loading ---
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_files = list(RAW_DATA_PATH.glob("*.jsonl"))
    if not all_files:
        print(f"Error: No raw .jsonl files found in '{RAW_DATA_PATH}'.")
        return
        
    print(f"\nLoading raw data from {len(all_files)} files...")
    # 'documents' will hold the original text content from each JSON line
    documents = []
    for file_path in tqdm(all_files, desc="Loading raw files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    documents.append(json.loads(line).get("text", ""))
                except json.JSONDecodeError:
                    continue

    # --- 3. Processing Pipeline (RE-CORRECTED Order) ---
    # The 'lines' variable will be transformed at each step.
    lines = documents
    print(f"-> Loaded {len(lines):,} initial documents.")

    # Step A: Filter original documents
    if args.filter_english:
        print("\nStep A: Filtering English documents...")
        english_vocab = setup_nltk_words()
        lines_after_filter = []
        removed_lines = []
        for line in tqdm(lines, desc="Filtering English"):
            if not line: continue
            words_in_line = line.split()
            if not words_in_line:
                lines_after_filter.append(line)
                continue
            english_word_count = sum(1 for w in words_in_line if is_likely_english(w, english_vocab))
            if (english_word_count / len(words_in_line)) <= args.english_threshold:
                lines_after_filter.append(line)
            else:
                removed_lines.append(line)
        
        print(f"-> Removed {len(removed_lines):,} lines in this step.")
        if args.diagnostic:
            log_path = OUTPUT_DIR / "log_removed_english.txt"
            print(f"-> Diagnostic log saved to: {log_path}")
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(removed_lines))
        
        lines = lines_after_filter
        print(f"-> Lines remaining: {len(lines):,}")
    
    # Step B: Segment into verses
    if args.segment_verses:
        print("\nStep B: Segmenting into verses...")
        full_text = " || ".join(line for line in lines if line)
        unified_text = full_text.replace('//', '||')
        lines = [verse.strip() for verse in unified_text.split('||') if verse.strip()]
        print(f"-> Segmented into {len(lines):,} verses.")

    # Step C: Filter non-Latin content from the resulting lines (which are now verses)
    if args.filter_non_latin:
        print("\nStep C: Filtering non-Latin/IAST lines/verses...")
        lines_after_filter = []
        removed_lines = []
        for line in tqdm(lines, desc="Filtering script"):
            if is_line_latin_only(line):
                lines_after_filter.append(line)
            else:
                removed_lines.append(line)

        print(f"-> Removed {len(removed_lines):,} lines in this step.")
        if args.diagnostic:
            log_path = OUTPUT_DIR / "log_removed_non_latin.txt"
            print(f"-> Diagnostic log saved to: {log_path}")
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(removed_lines))

        lines = lines_after_filter
        print(f"-> Lines remaining: {len(lines):,}")

    # Step D: Normalize as the final step
    if args.normalize:
        print("\nStep D: Normalizing text...")
        lines = [normalize_text(line) for line in tqdm(lines, desc="Normalizing")]
        print("-> Normalization complete.")

    # --- 4. Final Cleanup & Save ---
    final_lines = [line for line in lines if line]
    
    print(f"\nWriting {len(final_lines):,} final lines to {output_file_path}...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for line in final_lines:
            f.write(line + '\n')
            
    # --- 5. Final Statistics ---
    if not final_lines:
        print("\nNo data to generate statistics for.")
        return
        
    line_lengths = [len(line) for line in final_lines]
    print("\n--- Final Output Statistics ---")
    print(f"Total lines/verses: {len(final_lines):,}")
    print(f"Average length: {np.mean(line_lengths):.2f} characters")
    print(f"Median length:  {np.median(line_lengths):.0f} characters")
    print(f"Std. Dev of length: {np.std(line_lengths):.2f} characters")
    print(f"Min length: {min(line_lengths):,}")
    print(f"Max length: {max(line_lengths):,}")
    
    # --- 6. Verification Step ---
    if args.filter_non_latin:
        print("\n--- Verifying final output for non-Latin characters ---")
        offending_lines = []
        with open(output_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Verifying output"), 1):
                if not is_line_latin_only(line):
                    offending_lines.append((i, line.strip()))
                if len(offending_lines) >= 10: # Stop after finding 10 examples
                    break
        
        if not offending_lines:
            print("✅ Verification successful: Output file contains only allowed characters.")
        else:
            print("\n❌ VERIFICATION FAILED: The output file contains non-allowed characters.")
            print("This indicates a potential bug in the filtering logic.")
            print("Examples of offending lines:")
            for line_num, text in offending_lines:
                display_text = (text[:100] + '...') if len(text) > 100 else text
                print(f"  Line {line_num}: {display_text}")

    print("\n--- Preprocessing Complete ---")

if __name__ == "__main__":
    main() 