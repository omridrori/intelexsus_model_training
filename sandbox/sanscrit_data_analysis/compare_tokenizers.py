from pathlib import Path
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# --- Settings ---
# Path to our custom-trained Sanskrit tokenizer
CUSTOM_TOKENIZER_PATH = Path("sanscrit/sanskrit-bert-tokenizer")

# Name of the standard multilingual BERT model on Hugging Face Hub
MBERT_TOKENIZER_NAME = "bert-base-multilingual-cased"

# Sentences to compare
SENTENCES_TO_TEST = [
    "dharma",
    "devānām",
    "maharṣiruvāca",  # A compound word (sandhi)
    "sa maharṣir uvāca anena vacanena", # A full phrase
    "atha yogānuśāsanam" # Another classic phrase
]
# ---------------------------------------------------------------------------


def compare_tokenizers(custom_path: Path, hf_name: str, sentences: list):
    """
    Loads our custom tokenizer and a standard mBERT tokenizer to compare
    their tokenization of Sanskrit text.
    """
    print("--- Loading Tokenizers ---")
    try:
        # Load our custom tokenizer
        if not custom_path.exists():
            print(f"Error: Custom tokenizer not found at '{custom_path}'")
            return
        custom_tokenizer = AutoTokenizer.from_pretrained(str(custom_path))
        print(f"Successfully loaded custom tokenizer from '{custom_path}'")

        # Load standard mBERT tokenizer from Hugging Face
        mbert_tokenizer = AutoTokenizer.from_pretrained(hf_name)
        print(f"Successfully loaded mBERT tokenizer ('{hf_name}')")
        
    except Exception as e:
        print(f"An error occurred while loading tokenizers: {e}")
        return

    print("\n" + "="*60)
    print("--- Tokenization Comparison ---")
    print("="*60 + "\n")

    for sentence in sentences:
        print(f"Original Sentence:\n  '{sentence}'\n")
        
        # Tokenize with our custom tokenizer
        custom_tokens = custom_tokenizer.tokenize(sentence)
        print(f"Our Tokenizer ({len(custom_tokens)} tokens):")
        print(f"  {custom_tokens}\n")

        # Tokenize with mBERT
        mbert_tokens = mbert_tokenizer.tokenize(sentence)
        print(f"mBERT Tokenizer ({len(mbert_tokens)} tokens):")
        print(f"  {mbert_tokens}")
        
        print("\n" + "-"*60 + "\n")

 
if __name__ == "__main__":
    compare_tokenizers(CUSTOM_TOKENIZER_PATH, MBERT_TOKENIZER_NAME, SENTENCES_TO_TEST)



