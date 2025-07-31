import os
import argparse
import logging
from datasets import load_dataset, enable_progress_bar
from transformers import BertTokenizerFast

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def tokenize_and_save(raw_dataset_path, tokenizer_path, output_dir, max_length=512):
    """
    Tokenizes a raw text dataset and saves it to disk for faster loading.
    """
    if not os.path.exists(raw_dataset_path):
        raise FileNotFoundError(f"Raw dataset file not found at {raw_dataset_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer directory not found at {tokenizer_path}")

    logging.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    logging.info("Tokenizer loaded.")

    logging.info(f"Loading raw dataset from: {raw_dataset_path}")
    raw_dataset = load_dataset("text", data_files={"train": raw_dataset_path}, split="train")
    logging.info(f"Raw dataset loaded with {len(raw_dataset)} examples.")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_special_tokens_mask=True
        )

    logging.info("Tokenizing dataset... This may take a while but only needs to be done once.")
    enable_progress_bar()
    
    # Use multiple processes to speed up tokenization.
    # On Windows, this needs to be handled carefully, but the `datasets` library manages it.
    num_procs = 4 if os.name == 'nt' else 8 # A safe number for Windows
    
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_procs,
        remove_columns=["text"]
    )
    logging.info("Dataset tokenized successfully.")

    logging.info(f"Saving tokenized dataset to: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tokenized_dataset.save_to_disk(output_dir)
    logging.info(f"Dataset saved successfully to {output_dir}")
    logging.info(f"You can now point your training script to this directory.")

def main():
    parser = argparse.ArgumentParser(description="Tokenize a text dataset and save it for faster reuse.")
    
    # Define paths relative to the script's location within the sandbox
    script_dir = os.path.dirname(__file__)
    sanskrit_dir = os.path.abspath(os.path.join(script_dir, '..'))
    base_dir = os.path.abspath(os.path.join(sanskrit_dir, '..', '..')) # This should be the project root

    # Keep original data path, but ensure output is within the sandbox
    default_dataset_path = os.path.join(base_dir, 'data', 'sanskrit_preprocessed', 'sanskrit_latin_only.txt')
    default_tokenizer_path = os.path.join(sanskrit_dir, 'tokenizer', 'sanskrit-bert-tokenizer')
    default_output_dir = os.path.join(sanskrit_dir, 'data', 'tokenized_dataset') # Output inside sandbox/sanskrit/data

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=default_dataset_path,
        help=f"Path to the raw text dataset file. Default: {default_dataset_path}"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=default_tokenizer_path,
        help=f"Path to the pretrained tokenizer directory. Default: {default_tokenizer_path}"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir,
        help=f"Directory to save the tokenized dataset. Default: {default_output_dir}"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length for tokenization."
    )

    args = parser.parse_args()

    tokenize_and_save(args.dataset_path, args.tokenizer_path, args.output_dir, args.max_length)

if __name__ == "__main__":
    main() 