import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

SANSKRIT_PATH = DATA_ROOT / "sanskrit"
TIBETAN_PATH = DATA_ROOT / "tibetan"

# --- Parameters ---
LINES_TO_SAMPLE = 250000  # Increased sample size for better accuracy

def analyze_corpus_bytes(corpus_path: Path, language_name: str):
    """
    Analyzes a sample of lines from a corpus to determine the average
    number of bytes used per character in UTF-8 encoding.
    It now shuffles files to ensure a more representative sample.
    """
    print(f"\n--- Analyzing Byte-per-Character Distribution for: {language_name} ---")
    
    all_files = list(corpus_path.glob("**/*.jsonl"))
    if not all_files:
        print(f"No .jsonl files found in '{corpus_path}'. Skipping.")
        return

    # Shuffle the list of files to ensure we sample across the entire corpus
    print(f"Found {len(all_files)} files. Shuffling them for representative sampling...")
    random.shuffle(all_files)

    ratios = []
    lines_processed = 0

    # We use a generator to avoid listing all files if we hit the sample limit early
    file_generator = (file for file in all_files)

    with tqdm(total=LINES_TO_SAMPLE, desc=f"Sampling {language_name}") as pbar:
        for file_path in file_generator:
            if lines_processed >= LINES_TO_SAMPLE:
                break
                
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if lines_processed >= LINES_TO_SAMPLE:
                        break
                    
                    try:
                        text = json.loads(line).get("text", "")
                        if text:
                            char_count = len(text)
                            byte_count = len(text.encode('utf-8'))
                            
                            if char_count > 0:
                                ratios.append(byte_count / char_count)
                                
                            lines_processed += 1
                            pbar.update(1)
                            
                    except json.JSONDecodeError:
                        continue
    
    print(f"\n--- Report for {language_name} ---")
    if ratios:
        avg_ratio = np.mean(ratios)
        std_dev = np.std(ratios)
        min_ratio = np.min(ratios)
        max_ratio = np.max(ratios)
        
        print(f"Sampled {len(ratios):,} lines.")
        print(f"Average Bytes per Character: {avg_ratio:.4f}")
        print(f"Standard Deviation: {std_dev:.4f}")
        print(f"Min/Max Ratio in Sample: {min_ratio:.4f} / {max_ratio:.4f}")

    else:
        print("Could not sample any text lines from this corpus.")
    print("-" * 60)


if __name__ == "__main__":
    # First, let's run the cleaning script to make sure we analyze the 'real' data
    # Note: This assumes the cleaning script exists and is runnable.
    # For simplicity here, we will comment this out and assume data is ready.
    # print("Ensuring data is clean first...")
    # import sandbox.c01_initial_cleaning.clean_control_chars as cleaner
    # cleaner.process_and_clean_corpus()

    # Important: We will analyze the RAW data to explain the original size difference.
    print("Analyzing RAW data to explain the original file size differences.")
    analyze_corpus_bytes(SANSKRIT_PATH, "Sanskrit (Raw)")
    analyze_corpus_bytes(TIBETAN_PATH, "Tibetan (Raw)")
    
    # We could also analyze the processed data to see if cleaning changed the ratio.
    # PROCESSED_SANSKRIT_PATH = DATA_ROOT / "processed" / "sanskrit"
    # PROCESSED_TIBETAN_PATH = DATA_ROOT / "processed" / "tibetan"
    # print("\nAnalyzing PROCESSED data to see the effect of cleaning.")
    # analyze_corpus_bytes(PROCESSED_SANSKRIT_PATH, "Sanskrit (Processed)")
    # analyze_corpus_bytes(PROCESSED_TIBETAN_PATH, "Tibetan (Processed)")

    print("\nAnalysis complete.") 