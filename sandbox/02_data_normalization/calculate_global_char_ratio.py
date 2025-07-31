import json
from pathlib import Path
import time

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent 
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Characters that indicate an original, non-transliterated script
TIBETAN_CHARS = "ཤརཕྱོགསརྩེནསདཀརགསལཟླབ"
DEVANAGARI_CHARS = "अआइईउऊऋकखगघङचछजझञ"
ORIGINAL_SCRIPT_CHARS = set(TIBETAN_CHARS + DEVANAGARI_CHARS)

def calculate_global_ratio():
    """
    Scans the entire corpus to calculate the overall ratio of original script
    characters to transliterated characters across all documents combined.
    """
    print("--- Calculating global character ratio across all data... ---")
    start_time = time.time()
    
    all_files = list(DATA_ROOT.glob("**/*.jsonl"))
    
    total_chars = 0
    total_original_script_chars = 0
    
    file_count = len(all_files)
    for i, file_path in enumerate(all_files):
        print(f"Processing file {i+1}/{file_count}: {file_path.name}", end='\r')
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    text = json.loads(line).get("text", "")
                    if text:
                        total_chars += len(text)
                        total_original_script_chars += sum(1 for char in text if char in ORIGINAL_SCRIPT_CHARS)
                except json.JSONDecodeError:
                    continue
                    
    end_time = time.time()
    print("\n--- Processing Complete ---")
    print(f"Scanned {file_count} files in {end_time - start_time:.2f} seconds.")

    # --- Generate Final Report ---
    print("\n--- Global Corpus Character Ratio Report ---")
    
    if total_chars == 0:
        print("Could not find any text in the corpus.")
        return
        
    total_transliterated_chars = total_chars - total_original_script_chars
    
    original_perc = (total_original_script_chars / total_chars) * 100
    transliterated_perc = (total_transliterated_chars / total_chars) * 100
    
    print(f"Total Characters in Corpus: {total_chars:,}")
    print("-" * 40)
    print(f"Characters in Original Script: {total_original_script_chars:,} ({original_perc:.4f}%)")
    print(f"Characters in Transliteration: {total_transliterated_chars:,} ({transliterated_perc:.4f}%)")
    print("-" * 40)

if __name__ == "__main__":
    calculate_global_ratio() 