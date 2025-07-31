import json
from pathlib import Path
import string
import time
from tqdm import tqdm

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# --- Character Sets ---

# 1. Characters that are definitely part of an original script.
# Any line containing these will be ignored completely.
TIBETAN_CHARS = "ཤརཕྱོགསརྩེནསདཀརགསལཟླབ"
DEVANAGARI_CHARS = "अआइईउऊऋकखगघङचछजझञ"
ORIGINAL_SCRIPT_CHARS = set(TIBETAN_CHARS + DEVANAGARI_CHARS)

# 2. Standard characters that we want to filter out from the final report.
STANDARD_CHARS = set(
    string.ascii_letters + 
    string.digits + 
    string.punctuation + 
    string.whitespace
)

def generate_vocab_from_data():
    """
    Scans the corpus to produce a definitive list of all special characters
    used in transliterated texts.
    """
    print("--- Generating definitive transliteration vocabulary from data... ---")
    start_time = time.time()

    all_files = list(DATA_ROOT.glob("**/*.jsonl"))
    
    # This set will collect all characters from lines that are identified as transliterated.
    found_translit_chars = set()
    
    file_count = len(all_files)
    # Wrap the file loop with tqdm for a progress bar
    for file_path in tqdm(all_files, desc="Scanning files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    text = json.loads(line).get("text", "")
                    if not text:
                        continue
                    
                    # If the line contains any original script, we discard it completely.
                    if any(char in ORIGINAL_SCRIPT_CHARS for char in text):
                        continue
                    
                    # Otherwise, it's a transliteration, so we add all its characters to our set.
                    found_translit_chars.update(text)

                except json.JSONDecodeError:
                    continue
    
    # Now, filter out all the standard characters to isolate the special ones.
    special_chars = found_translit_chars - STANDARD_CHARS
    
    end_time = time.time()
    print("\n\n--- Analysis Complete ---")
    print(f"Scanned {file_count} files in {end_time - start_time:.2f} seconds.")

    # --- Generate Final Report ---
    print("\n--- Discovered Special Transliteration Characters ---")
    
    if not special_chars:
        print("No special characters were found. All transliterations use standard A-Z characters.")
    else:
        sorted_chars = sorted(list(special_chars))
        print(f"Found {len(sorted_chars)} unique special characters used in your data.")
        print("\nHere is the exact string to use in your other scripts:")
        print("-" * 60)
        # Print the characters as a clean, copy-pasteable string
        print("".join(sorted_chars))
        print("-" * 60)

if __name__ == "__main__":
    generate_vocab_from_data() 