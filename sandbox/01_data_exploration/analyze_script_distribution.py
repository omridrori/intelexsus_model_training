import json
from pathlib import Path
import string
from tqdm import tqdm

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

SANSKRIT_PATH = DATA_ROOT / "sanskrit"
TIBETAN_PATH = DATA_ROOT / "tibetan"

# --- Character Set Definition ---
# This is our strict definition of what's allowed for a word to be considered "transliterated".
STRICT_IAST_CHARS = "āīūṛṝḷḹṃḥṅñṭḍṇśṣĀĪŪṚṜḶḸṀḤṄÑṬḌṆŚṢ"
ALLOWED_CHARS = set(
    string.ascii_letters + 
    string.digits + 
    string.punctuation + 
    string.whitespace + 
    STRICT_IAST_CHARS
)

def is_word_transliterated(word: str) -> bool:
    """
    Checks if a word contains ONLY allowed transliteration characters.
    It ignores surrounding punctuation for the check.
    """
    # First, strip common punctuation from the edges of the word
    cleaned_word = word.strip(string.punctuation)
    if not cleaned_word:
        return True # A word that is only punctuation is not 'original script'
        
    # A word is transliterated if all its characters are in the allowed set.
    return all(char in ALLOWED_CHARS for char in cleaned_word)

def analyze_corpus_distribution(corpus_path: Path, language_name: str):
    """
    Analyzes a corpus to count words in transliteration vs. original script.
    """
    print(f"\n--- Analyzing Script Distribution for: {language_name} ---")
    
    all_files = list(corpus_path.glob("**/*.jsonl"))
    if not all_files:
        print(f"No .jsonl files found in '{corpus_path}'. Skipping.")
        return

    transliterated_word_count = 0
    original_script_word_count = 0
    total_words = 0

    for file_path in tqdm(all_files, desc=f"Processing {language_name}"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    text = json.loads(line).get("text", "")
                    if text:
                        words = text.split()
                        for word in words:
                            # We check the original word, not a cleaned one
                            if not word.strip():
                                continue
                            
                            total_words += 1
                            if is_word_transliterated(word):
                                transliterated_word_count += 1
                            else:
                                original_script_word_count += 1
                except json.JSONDecodeError:
                    # Ignore lines that are not valid JSON
                    continue

    print(f"\n--- Report for {language_name} ---")
    if total_words > 0:
        trans_perc = (transliterated_word_count / total_words) * 100
        orig_perc = (original_script_word_count / total_words) * 100
        print(f"Total words processed: {total_words:,}")
        print(f"Words in Transliteration: {transliterated_word_count:,} ({trans_perc:.2f}%)")
        print(f"Words in Original Script: {original_script_word_count:,} ({orig_perc:.2f}%)")
    else:
        print("No words found in this corpus.")
    print("-" * 50)


if __name__ == "__main__":
    analyze_corpus_distribution(SANSKRIT_PATH, "Sanskrit")
    analyze_corpus_distribution(TIBETAN_PATH, "Tibetan")
    print("\nAnalysis complete.") 