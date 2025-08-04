from transformers import BertTokenizerFast
from pathlib import Path

# --- 1. Settings ---

# Path to the trained tokenizer directory
TOKENIZER_DIR = Path("sanskrit-bert-tokenizer")

# Input file: our clean and unified data
INPUT_DATA_FILE = Path("data/sanskrit_preprocessed/sanskrit_chunks_final_v2.txt")

# How many samples to take from the beginning of the file
NUM_SAMPLES_TO_TEST = 5

# Sequence length for demonstration. You can play with this to see how padding and truncation work.
MAX_LENGTH = 512


# --- 2. Debug logic ---

def debug_tokenizer_on_real_data():
    """
    Loads real examples from the data file and demonstrates how the tokenizer processes them.
    """
    # Validation checks
    if not TOKENIZER_DIR.exists():
        print(f"❌ ERROR: Tokenizer directory not found at '{TOKENIZER_DIR}'")
        return
    if not INPUT_DATA_FILE.exists():
        print(f"❌ ERROR: Input data file not found at '{INPUT_DATA_FILE}'")
        return

    # Loading samples from the file
    print(f"Reading {NUM_SAMPLES_TO_TEST} samples from '{INPUT_DATA_FILE}'...")
    with open(INPUT_DATA_FILE, 'r', encoding='utf-8') as f:
        sample_texts = [f.readline().strip() for _ in range(NUM_SAMPLES_TO_TEST)]

    # Loading the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    print(f"✅ Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
    print(f"--- Visualizing output for MAX_LENGTH = {MAX_LENGTH} ---\n")

    for i, text in enumerate(sample_texts):
        if not text: continue # Skip empty lines if any
        
        print(f"=============== SAMPLE #{i+1} ================")
        print(f"Original Text (from file):\n'{text}'")
        print("-" * 25)

        # The tokenization itself
        encoding = tokenizer(
            text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )

        # Converting IDs back to tokens for display
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])

        print("Tokenization Result (what BERT sees):\n")
        print(f"Tokens ({len(tokens)} total):\n{tokens}\n")
        print(f"Token IDs ({len(encoding['input_ids'])} total):\n{encoding['input_ids']}\n")
        print(f"Attention Mask ({len(encoding['attention_mask'])} total):\n{encoding['attention_mask']}\n")
        print("===========================================\n")


if __name__ == "__main__":
    debug_tokenizer_on_real_data() 