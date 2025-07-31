import json
from pathlib import Path
import time
import string
from tqdm import tqdm

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent 
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# --- DEFINITIVE, MANUALLY-CURATED CHARACTER SET ---
# Based on the previous output, we see the data is very noisy.
# Therefore, instead of deriving the vocab from the data, we define it strictly.
# This set contains standard characters plus a clean, known set for IAST.
# Any character NOT in this set will cause a word to be flagged.
STRICT_IAST_CHARS = "āīūṛṝḷḹṃḥṅñṭḍṇśṣĀĪŪṚṜḶḸṀḤṄÑṬḌṆŚṢ"
ALLOWED_CHARS = set(
    string.ascii_letters + 
    string.digits + 
    string.punctuation + 
    string.whitespace + 
    STRICT_IAST_CHARS
)

def contains_foreign_chars(word: str) -> bool:
    """
    Checks if a word contains ANY character that is NOT in our strict
    allowed set of transliteration characters.
    """
    for char in word:
        if char not in ALLOWED_CHARS:
            return True
    return False

def analyze_word_scripts():
    """
    Scans the entire corpus, classifies each word as 'Transliteration' or
    'Foreign/Original' and provides a final report on the distribution.
    """
    print("--- Analyzing word script distribution using STRICT vocabulary... ---")
    start_time = time.time()
    
    all_files = list(DATA_ROOT.glob("**/*.jsonl"))
    
    transliterated_word_count = 0
    foreign_word_count = 0
    
    # Wrap the file loop with tqdm for a progress bar
    for file_path in tqdm(all_files, desc="Analyzing word scripts"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    text = json.loads(line).get("text", "")
                    if text:
                        words = text.split()
                        for word in words:
                            if contains_foreign_chars(word):
                                foreign_word_count += 1
                            else:
                                transliterated_word_count += 1
                except json.JSONDecodeError:
                    continue
                    
    end_time = time.time()
    print("\n\n--- Processing Complete ---")
    print(f"Scanned {len(all_files)} files in {end_time - start_time:.2f} seconds.")

    # --- Generate Final Report ---
    total_words = foreign_word_count + transliterated_word_count
    print("\n--- Global Corpus Word-Script Ratio Report ---")
    
    if total_words == 0:
        print("Could not find any words in the corpus.")
        return
        
    foreign_perc = (foreign_word_count / total_words) * 100
    transliterated_perc = (transliterated_word_count / total_words) * 100
    
    print(f"Total Words in Corpus: {total_words:,}")
    print("-" * 50)
    print(f"Words with Foreign/Original Script: {foreign_word_count:,} ({foreign_perc:.4f}%)")
    print(f"Words in Pure Transliteration:    {transliterated_word_count:,} ({transliterated_perc:.4f}%)")
    print("-" * 50)

if __name__ == "__main__":
    analyze_word_scripts() 