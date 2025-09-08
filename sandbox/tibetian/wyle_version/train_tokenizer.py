import os
import json
import argparse
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit

# -----------------------------------------------------------------------------
# Utility to count total lines for progress bar --------------------------------
# -----------------------------------------------------------------------------


def count_lines(path: str) -> int:
    """Return the total number of non-empty lines across all .jsonl files."""
    total = 0
    for filename in os.listdir(path):
        if filename.endswith(".jsonl"):
            fp = os.path.join(path, filename)
            with open(fp, "r", encoding="utf-8") as f:
                for _ in f:
                    total += 1
    return total

def get_text_iterator(data_dir: str):
    """
    A generator that yields text from all .jsonl files in a directory.
    This is memory-efficient as it doesn't load all data at once.
    """
    print(f"Reading files from: {data_dir}")
    for filename in os.listdir(data_dir):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        text = json.loads(line).get("text")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        continue

def train_tibetan_tokenizer(input_dir: str, output_path: str, vocab_size: int):
    """
    Trains a WordPiece tokenizer on the Tibetan Wylie corpus.
    
    Args:
        input_dir (str): Directory containing the cleaned .jsonl files.
        output_path (str): Path to save the trained tokenizer file (e.g., 'tokenizer.json').
        vocab_size (int): The desired size of the vocabulary.
    """
    # 1. Initialize a tokenizer
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    
    # 2. Set up the pre-tokenizer – Whitespace *only*, keep '+' and "'" inside tokens
    tokenizer.pre_tokenizer = WhitespaceSplit()
    
    # 3. Initialize a trainer
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        initial_alphabet=list("'+.~")
    )
    
    # 4. Get the text iterator
    text_iterator = get_text_iterator(input_dir)

    total_lines = count_lines(input_dir)
    
    # 5. Train the tokenizer
    print(f"Starting tokenizer training with vocab size: {vocab_size}...")
    tokenizer.train_from_iterator(text_iterator, trainer=trainer, length=total_lines)
    print("Training complete.")
    
    # 6. Save the tokenizer
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    tokenizer.save(output_path)
    print(f"Tokenizer saved to: {output_path}")
    
    # 7. Demonstrate usage
    print("\n--- Tokenizer Usage Example ---")
    encoded = tokenizer.encode("slob dpon pad+ma'i sku la phyag 'tshal lo")
    print("Original text: slob dpon pad+ma'i sku la phyag 'tshal lo")
    print(f"Encoded tokens: {encoded.tokens}")
    print(f"Encoded IDs: {encoded.ids}")
    print("-----------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a WordPiece tokenizer on Tibetan Wylie text.")
    parser.add_argument(
        "--input_dir", 
        default=os.path.join("cleaned_data", "step3_wylie_converted"), 
        help="Directory containing the input .jsonl files."
    )
    parser.add_argument(
        "--output_path", 
        default=os.path.join("tibetan-tokenizer", "tokenizer.json"), 
        help="Path to save the trained tokenizer."
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=30000, 
        help="Size of the vocabulary to train."
    )
    args = parser.parse_args()
    
    train_tibetan_tokenizer(args.input_dir, args.output_path, args.vocab_size)


