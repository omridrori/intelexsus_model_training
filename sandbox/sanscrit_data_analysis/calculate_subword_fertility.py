import os
import random
from pathlib import Path
from transformers import AutoTokenizer, logging
from tqdm import tqdm

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress tokenizer warnings
logging.set_verbosity(40)

# ---------------------------------------------------------------------------
# --- Settings ---
# The pre-processed, clean text file to be analyzed
PROCESSED_DATA_FILE = Path("data/sanskrit_preprocessed/sanskrit_bert_ready_512.txt")
CUSTOM_TOKENIZER_PATH = Path("sanscrit/sanskrit-bert-tokenizer")
MBERT_TOKENIZER_NAME = "bert-base-multilingual-cased"
# Number of random lines to sample from the file.
NUM_SAMPLES = 50000
# ---------------------------------------------------------------------------


def calculate_fertility_on_processed_data():
    """
    Calculates Subword Fertility on a large, random sample of the 
    pre-processed Sanskrit data file.
    """
    if not PROCESSED_DATA_FILE.exists():
        print(f"Error: Processed data file not found at '{PROCESSED_DATA_FILE}'")
        return

    # --- Load all lines from the processed file ---
    print(f"--- Loading lines from '{PROCESSED_DATA_FILE}' ---")
    with open(PROCESSED_DATA_FILE, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    if not all_lines:
        print("No data found to analyze.")
        return
        
    print(f"Loaded {len(all_lines):,} total lines.")

    # --- Take a random sample ---
    sample_size = min(NUM_SAMPLES, len(all_lines))
    print(f"--- Taking a random sample of {sample_size:,} lines ---")
    # Strip whitespace/newlines from each line in the sample
    texts_to_process = [line.strip() for line in random.sample(all_lines, sample_size) if line.strip()]
    
    if not texts_to_process:
        print("Sample is empty after cleaning. Nothing to process.")
        return

    # Count total words in the final sample
    total_words = sum(len(text.split()) for text in texts_to_process)

    # --- Load Tokenizers ---
    print("--- Loading Tokenizers ---")
    try:
        custom_tokenizer = AutoTokenizer.from_pretrained(str(CUSTOM_TOKENIZER_PATH))
        mbert_tokenizer = AutoTokenizer.from_pretrained(MBERT_TOKENIZER_NAME)
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
        return

    # --- Batch process the sample in mini-batches to show progress ---
    print(f"\n--- Analyzing Sampled Corpus ({len(texts_to_process):,} documents) ---")
    
    BATCH_SIZE = 1000 
    total_custom_tokens = 0
    total_mbert_tokens = 0

    # Process in mini-batches with a progress bar
    for i in tqdm(range(0, len(texts_to_process), BATCH_SIZE), desc="Tokenizing batches"):
        batch = texts_to_process[i:i + BATCH_SIZE]
        
        custom_token_results = custom_tokenizer(batch, verbose=False)
        mbert_token_results = mbert_tokenizer(batch, verbose=False)

        total_custom_tokens += sum(len(ids) for ids in custom_token_results['input_ids'])
        total_mbert_tokens += sum(len(ids) for ids in mbert_token_results['input_ids'])

    # --- Calculate and Print Results ---
    if total_words == 0:
        print("No words were found in the sample.")
        return

    fertility_custom = total_custom_tokens / total_words
    fertility_mbert = total_mbert_tokens / total_words

    print("\n" + "="*50)
    print("--- Subword Fertility Results (from Processed Data Sample) ---")
    print(f"Sampled {len(texts_to_process):,} lines with {total_words:,} words.\n")

    print(f"Our Custom Tokenizer:")
    print(f"  Total Tokens: {total_custom_tokens:,}")
    print(f"  Subword Fertility: {fertility_custom:.4f}\n")

    print(f"mBERT Tokenizer:")
    print(f"  Total Tokens: {total_mbert_tokens:,}")
    print(f"  Subword Fertility: {fertility_mbert:.4f}\n")
    
    print("="*50)

if __name__ == "__main__":
    calculate_fertility_on_processed_data()
