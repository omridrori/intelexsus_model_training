from pathlib import Path
from tokenizers import BertWordPieceTokenizer
import os
from tqdm import tqdm

# --- 1. Settings ---

# Input file: the clean chunks file we created in the previous step
INPUT_FILE = Path("data/sanskrit_preprocessed/sanskrit_chunks_final_v2.txt")

# Where to save the trained tokenizer
# Important: this will be a directory, not a single file
OUTPUT_DIR = "sanskrit-bert-tokenizer"

# --- 2. Validation checks ---

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Corpus file not found at: {INPUT_FILE}. Please run the chunk processing script first.")

print("--- Starting Tokenizer Training ---")
print(f"Reading corpus from: {INPUT_FILE}")
print(f"Tokenizer will be saved to: '{OUTPUT_DIR}' directory")

# --- 3. Creating and training the tokenizer ---

print("Creating BERT WordPiece tokenizer...")

# Creating an empty tokenizer with settings suitable for BERT
# Important: strip_accents=False preserves our diacritical marks
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True
)

print("Training tokenizer on our data...")

# Training the tokenizer on our data file
tokenizer.train(
    files=[str(INPUT_FILE)],
    vocab_size=32000,
    min_frequency=2,
    show_progress=True,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
)

# --- 4. Saving the tokenizer ---

print("Saving tokenizer...")

# Creating output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Saving the tokenizer. This will create a vocab.txt file and other configuration files
tokenizer.save_model(OUTPUT_DIR)

print("\n--- Tokenizer Training Complete! ---")
print(f"Tokenizer files saved to the '{OUTPUT_DIR}' directory.")
print(f"The main vocabulary file is: {os.path.join(OUTPUT_DIR, 'vocab.txt')}")

# --- 5. Additional information ---
print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")
print(f"Special tokens: {tokenizer.token_to_id('[PAD]')}, {tokenizer.token_to_id('[UNK]')}, {tokenizer.token_to_id('[CLS]')}, {tokenizer.token_to_id('[SEP]')}, {tokenizer.token_to_id('[MASK]')}") 