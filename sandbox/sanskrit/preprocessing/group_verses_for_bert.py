from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizerFast
import numpy as np

# --- 1. Settings ---

# Path to your trained tokenizer
TOKENIZER_PATH = Path("sanskrit-bert-tokenizer")

# Input file: the clean output from the previous script (after filtering and cleaning)
INPUT_FILE = Path("data/sanskrit_preprocessed/sanskrit_chunks_final_v2.txt")

# Output file: the final data, ready for BERT training
OUTPUT_FILE = Path("data/sanskrit_preprocessed/sanskrit_bert_ready_512.txt")

# Token limit. We aim for slightly less than 512
# to leave room for special tokens like [CLS] and [SEP] that BERT will add during training
MAX_TOKENS = 510

# Ensure the output file directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# --- 2. Verse grouping logic ---

def group_verses_by_token_length():
    print("--- Starting Verse Grouping by Token Length for BERT ---")
    
    # Validation checks
    if not INPUT_FILE.exists():
        print(f"❌ ERROR: Input file not found at '{INPUT_FILE}'. Please run the cleaning script first.")
        return
    if not TOKENIZER_PATH.exists():
        print(f"❌ ERROR: Tokenizer not found at '{TOKENIZER_PATH}'. Please run train_tokenizer.py first.")
        return

    # Loading the tokenizer
    print(f"Loading tokenizer from '{TOKENIZER_PATH}'...")
    tokenizer = BertTokenizerFast.from_pretrained(str(TOKENIZER_PATH))

    # Reading all verses from the file
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        verses = [line.strip() for line in f if line.strip()]
    
    print(f"Read {len(verses):,} clean verses from input file.")

    grouped_chunks = []
    current_chunk_verses = []
    current_token_count = 0

    for verse in tqdm(verses, desc="Grouping into ~512 token chunks"):
        # Important: we tokenize without adding special tokens, just to measure length
        verse_tokens = tokenizer.encode(verse, add_special_tokens=False)
        verse_token_count = len(verse_tokens)

        # If adding the current verse would exceed the limit
        if current_token_count + verse_token_count > MAX_TOKENS:
            # If there's something in the current chunk, save it
            if current_chunk_verses:
                grouped_chunks.append(" ".join(current_chunk_verses))
            # Start a new chunk with the current verse
            current_chunk_verses = [verse]
            current_token_count = verse_token_count
        else:
            # Otherwise, just add the verse to the chunk
            current_chunk_verses.append(verse)
            current_token_count += verse_token_count

    # Save the last chunk if there's anything left
    if current_chunk_verses:
        grouped_chunks.append(" ".join(current_chunk_verses))

    # --- 3. Saving results and statistics ---
    print(f"\nWriting {len(grouped_chunks):,} grouped lines to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for chunk in grouped_chunks:
            f.write(chunk + '\n')
            
    # Calculate statistics on the output
    print("Calculating final statistics...")
    output_token_lengths = [len(tokenizer.encode(chunk, add_special_tokens=False)) for chunk in tqdm(grouped_chunks, desc="Analyzing output")]
    
    print("\n--- Final Statistics ---")
    print(f"Input verses: {len(verses):,}")
    print(f"Output grouped chunks: {len(grouped_chunks):,}")
    if output_token_lengths:
        print(f"Average token length per chunk: {np.mean(output_token_lengths):.2f}")
        print(f"Median token length per chunk: {np.median(output_token_lengths):.0f}")
        print(f"Min / Max token length: {np.min(output_token_lengths)} / {np.max(output_token_lengths)}")
        
    print(f"\n✅ Done! The file '{OUTPUT_FILE}' is now the final dataset ready for training.")

if __name__ == "__main__":
    group_verses_by_token_length() 