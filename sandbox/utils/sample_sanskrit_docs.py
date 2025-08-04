import json
from pathlib import Path
import random

def sample_sanskrit_documents():
    """
    Samples 10 documents from the Sanskrit corpus with specific word count constraints.
    """
    # --- Configuration ---
    try:
        # This structure assumes the script is in sandbox/utils
        SCRIPT_DIR = Path(__file__).parent
        PROJECT_ROOT = SCRIPT_DIR.parent.parent
    except NameError:
        # Fallback for interactive environments
        PROJECT_ROOT = Path.cwd()

    SANSKRIT_DATA_PATH = PROJECT_ROOT / "data" / "sanskrit"
    OUTPUT_FILE = SCRIPT_DIR / "sanskrit_samples.txt"
    NUM_SAMPLES = 10
    MIN_WORDS = 20
    MAX_WORDS = 40

    print(f"--- Starting Sanskrit Document Sampling ---")
    print(f"Source directory: {SANSKRIT_DATA_PATH}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Constraints: Min words={MIN_WORDS}, Max words={MAX_WORDS}")
    print(f"Number of samples to find: {NUM_SAMPLES}")
    print("-" * 50)

    if not SANSKRIT_DATA_PATH.exists():
        print(f"❌ ERROR: Data directory not found at '{SANSKRIT_DATA_PATH}'")
        return

    all_files = list(SANSKRIT_DATA_PATH.glob("**/*.jsonl"))
    if not all_files:
        print(f"❌ ERROR: No .jsonl files found in '{SANSKRIT_DATA_PATH}'")
        return

    random.shuffle(all_files)

    sampled_docs = []
    docs_found = 0

    # Use a file generator to avoid listing all files if we find enough samples early
    file_generator = (file for file in all_files)

    for file_path in file_generator:
        if docs_found >= NUM_SAMPLES:
            break

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # To avoid reading huge files entirely, we'll read lines and shuffle them
                lines = f.readlines()
                random.shuffle(lines)

                for line in lines:
                    if docs_found >= NUM_SAMPLES:
                        break
                    
                    try:
                        text = json.loads(line).get("text", "")
                        if not text:
                            continue

                        words = text.split()
                        word_count = len(words)

                        if word_count >= MIN_WORDS:
                            # Truncate if necessary
                            final_words = words[:MAX_WORDS]
                            document = " ".join(final_words)
                            sampled_docs.append(document)
                            docs_found += 1

                    except (json.JSONDecodeError, AttributeError):
                        # Ignore malformed JSON lines or lines without 'text'
                        continue
        except Exception as e:
            print(f"Could not process file {file_path}: {e}")


    print(f"\n--- Found {len(sampled_docs)} Documents ---")
    if sampled_docs:
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
                for i, doc in enumerate(sampled_docs, 1):
                    f_out.write(f"[{i:02d}] {doc}\n\n")
            print(f"✅ Successfully wrote {len(sampled_docs)} samples to {OUTPUT_FILE}")
        except Exception as e:
            print(f"❌ ERROR: Could not write to output file: {e}")
    else:
        print("Could not find any documents matching the criteria.")
    print("--- Sampling Complete ---")

if __name__ == "__main__":
    sample_sanskrit_documents()
