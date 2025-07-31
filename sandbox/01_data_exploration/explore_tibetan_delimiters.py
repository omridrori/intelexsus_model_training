import json
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
TIBETAN_PATH = DATA_ROOT / "tibetan"

# --- Parameters ---
FILES_TO_SCAN = 5  # Limit how many files we check to keep the output readable
SAMPLES_PER_FILE = 3 # Limit how many context samples we print from each file

def explore_delimiters():
    """
    Scans a subset of the Tibetan corpus to find and display context around
    potential sentence delimiters like '།' (Shad) and '.' (period).
    """
    print("--- Exploring potential sentence delimiters in Tibetan data... ---\n")
    
    all_files = list(TIBETAN_PATH.glob("**/*.jsonl"))
    if not all_files:
        print(f"Error: No .jsonl files found in '{TIBETAN_PATH}'. Cannot run analysis.")
        return

    delimiter_counts = Counter()
    total_lines_scanned = 0

    files_to_process = all_files[:FILES_TO_SCAN]
    print(f"Scanning the first {len(files_to_process)} files...\n")

    for file_path in tqdm(files_to_process, desc="Scanning files"):
        print(f"\n--- Samples from: {file_path.name} ---")
        samples_found_in_file = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines_scanned += 1
                
                # We only want to print a few samples to not flood the console
                if samples_found_in_file >= SAMPLES_PER_FILE:
                    # But we still continue counting delimiters in the rest of the file
                    text = line.strip()
                    delimiter_counts['།'] += text.count('།')
                    delimiter_counts['.'] += text.count('.')
                    continue

                # Check if the line contains a Shad and we still need samples
                if '།' in line:
                    # Split by the Shad to show context
                    segments = line.split('།')
                    
                    # The part before the first Shad is a good candidate for a sentence
                    context = segments[0]
                    
                    print(f"  Context for '།': \"{context.strip()}།\"")
                    samples_found_in_file += 1

                # We can also check for periods for comparison
                elif '.' in line and samples_found_in_file < SAMPLES_PER_FILE:
                    segments = line.split('.')
                    context = segments[0]
                    print(f"  Context for '.': \"{context.strip()}.\"")
                    samples_found_in_file += 1
                
                # After checking for samples, count all delimiters in the line
                text = line.strip()
                delimiter_counts['།'] += text.count('།')
                delimiter_counts['.'] += text.count('.')

    # --- Final Report ---
    print("\n\n--- Delimiter Frequency Report ---")
    print(f"Scanned {total_lines_scanned:,} lines across {len(files_to_process)} files.")
    print("-" * 40)
    if not delimiter_counts:
        print("No '།' or '.' delimiters were found.")
    else:
        for delimiter, count in delimiter_counts.items():
            print(f"  Delimiter '{delimiter}': Found {count:,} times")
    print("-" * 40)
    print("\nPlease review the 'Context' samples above to determine the correct sentence terminator.")


if __name__ == "__main__":
    explore_delimiters() 