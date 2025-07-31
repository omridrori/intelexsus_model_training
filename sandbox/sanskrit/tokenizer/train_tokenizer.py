
from tokenizers import BertWordPieceTokenizer
import os

# Step 1: Preparation and settings

# Define paths
corpus_file_path = r"C:\Users\omrid\Desktop\university\intelexsus\sandbox\sanskrit\outputs\sanskrit_latin_only.txt"
output_dir = "sanskrit-bert-tokenizer"

# Ensure the corpus path exists
if not os.path.exists(corpus_file_path):
    raise FileNotFoundError(f"Corpus file not found at: {corpus_file_path}")

print("Step 1 completed: Paths are set.")
print(f"Corpus file path: {corpus_file_path}")
print(f"Output directory for the tokenizer: {output_dir}")

# Step 2: Create an empty tokenizer
print("\nStep 2: Creating an empty tokenizer...")
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,  # Crucial for languages with diacritics
    lowercase=True
)
print("Step 2 completed: Empty tokenizer created with the required settings.")

# Step 3: Train the tokenizer
print("\nStep 3: Training the tokenizer...")
tokenizer.train(
    files=[corpus_file_path],
    vocab_size=32000,
    min_frequency=2,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
)
print("Step 3 completed: Tokenizer training is finished.")

# Step 4: Save the trained tokenizer
print("\nStep 4: Saving the tokenizer...")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
tokenizer.save_model(output_dir)
print(f"Step 4 completed: Tokenizer saved to {os.path.abspath(output_dir)}")
print(f"The vocabulary file (vocab.txt) can be found in the '{output_dir}' directory.") 