import os
from tokenizers import BertWordPieceTokenizer
import random
from tqdm import tqdm

# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Configuration ---
# The tokenizer output directory is expected to be in the same directory as this script.
TOKENIZER_DIR = os.path.join(script_dir, "sanskrit-bert-tokenizer")
VOCAB_FILE = os.path.join(TOKENIZER_DIR, "vocab.txt")
CORPUS_FILE_PATH = r"C:\Users\omrid\Desktop\university\intelexsus\sandbox\sanskrit\outputs\sanskrit_latin_only.txt"
SAMPLE_SIZE = 1000  # Number of lines to use for analysis

# --- 1. Load the Tokenizer ---
print(f"Loading tokenizer from: {VOCAB_FILE}")
if not os.path.exists(VOCAB_FILE):
    raise FileNotFoundError(f"Tokenizer vocabulary file not found at: {VOCAB_FILE}. Please ensure the path is correct and you have run the training script.")

# Load the tokenizer from the saved vocab file.
tokenizer = BertWordPieceTokenizer(VOCAB_FILE, lowercase=True)

# --- 2. Print Vocabulary Size ---
vocab_size = tokenizer.get_vocab_size()
print(f"\n--- Tokenizer Analysis ---")
print(f"Vocabulary Size: {vocab_size}")

# --- 3. Prepare Test Data ---
print(f"\nPreparing test data by sampling {SAMPLE_SIZE} lines from '{os.path.basename(CORPUS_FILE_PATH)}'...")
if not os.path.exists(CORPUS_FILE_PATH):
    raise FileNotFoundError(f"Corpus file not found at: {CORPUS_FILE_PATH}")

with open(CORPUS_FILE_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()

if len(lines) == 0:
    raise ValueError("Corpus file is empty. Cannot perform analysis.")

if len(lines) < SAMPLE_SIZE:
    print(f"Warning: Corpus has fewer than {SAMPLE_SIZE} lines. Using all {len(lines)} lines for analysis.")
    test_lines = lines
else:
    test_lines = random.sample(lines, SAMPLE_SIZE)

print("Test data prepared.")

# --- 4. Calculate Unknown Token ([UNK]) Percentage and Average Length ---
print(f"\nAnalyzing {len(test_lines)} sample lines...")
unk_token_id = tokenizer.token_to_id('[UNK]')
total_tokens = 0
unk_tokens = 0
sequence_lengths = []

for line in tqdm(test_lines, desc="Analyzing tokens"):
    line = line.strip()
    if not line:
        continue
    
    encoding = tokenizer.encode(line)
    sequence_lengths.append(len(encoding.ids))
    total_tokens += len(encoding.ids)
    
    # Count UNK tokens in the current line
    unk_tokens += encoding.ids.count(unk_token_id)

# --- 5. Display Results ---
if total_tokens > 0:
    unk_percentage = (unk_tokens / total_tokens) * 100
    print(f"\nUnknown ([UNK]) Token Percentage: {unk_percentage:.4f}%")
else:
    print("\nCould not calculate UNK percentage, no tokens found in sample.")

if sequence_lengths:
    avg_seq_length = sum(sequence_lengths) / len(sequence_lengths)
    print(f"Average Sequence Length (Tokens per Verse): {avg_seq_length:.2f}")
else:
    print("Could not calculate average sequence length, no sequences found in sample.") 