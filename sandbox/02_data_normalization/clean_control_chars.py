import json
import unicodedata
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "processed"

CORPORA = ["sanskrit", "tibetan"]

def clean_text(text: str) -> str:
    """
    Removes Unicode control characters from a string.
    
    Args:
        text: The input string, potentially containing control characters.

    Returns:
        A cleaned string with control characters removed.
    """
    if not isinstance(text, str):
        return ""
        
    # Keeps characters that are not in the "Control characters" (Cc) category
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != 'C')

def process_and_clean_corpus():
    """
    Reads raw data from specified corpora, cleans control characters from the
    'text' field in each JSON line, and writes the clean data to a
    'processed' directory, preserving the original file structure.
    """
    print("--- Starting low-level cleaning: Removing control characters... ---")
    
    for corpus_name in CORPORA:
        raw_corpus_path = RAW_DATA_ROOT / corpus_name
        processed_corpus_path = PROCESSED_DATA_ROOT / corpus_name
        
        if not raw_corpus_path.exists():
            print(f"\nWarning: Raw data directory not found for '{corpus_name}' at: {raw_corpus_path}")
            print("Skipping this corpus.")
            continue
            
        print(f"\nProcessing corpus: '{corpus_name}'")
        processed_corpus_path.mkdir(parents=True, exist_ok=True)
        
        all_files = list(raw_corpus_path.glob("**/*.jsonl"))
        if not all_files:
            print(f"  No .jsonl files found in {raw_corpus_path}.")
            continue
            
        for input_path in tqdm(all_files, desc=f"Cleaning {corpus_name} files"):
            output_path = processed_corpus_path / input_path.name
            
            with open(input_path, 'r', encoding='utf-8') as f_in, \
                 open(output_path, 'w', encoding='utf-8') as f_out:
                
                for line in f_in:
                    try:
                        data = json.loads(line)
                        if 'text' in data:
                            # Clean the text and update the dictionary
                            data['text'] = clean_text(data['text'])
                        
                        # Write the cleaned (or original, if no 'text' field) JSON back to the new file
                        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                        
                    except json.JSONDecodeError:
                        # If a line isn't valid JSON, we can't process it.
                        # We can choose to log this or simply skip it.
                        # For now, we skip it to keep the process running.
                        continue
                        
    print("\n--- Cleaning complete. ---")
    print(f"Cleaned data is located in: {PROCESSED_DATA_ROOT}")

if __name__ == "__main__":
    process_and_clean_corpus() 