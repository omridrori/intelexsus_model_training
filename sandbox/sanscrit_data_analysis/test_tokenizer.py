from pathlib import Path
from transformers import BertTokenizerFast

# ---------------------------------------------------------------------------
# --- Settings ---
# Path to the pre-trained tokenizer directory
TOKENIZER_PATH = Path("sanscrit/sanskrit-bert-tokenizer")
# ---------------------------------------------------------------------------


def test_tokenizer(tokenizer_path: Path):
    """
    Loads a pre-trained tokenizer and shows how it tokenizes specific
    Sanskrit words to demonstrate its understanding of morphology.
    """
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer directory not found at '{tokenizer_path}'")
        print("Please ensure the path is correct.")
        return

    print(f"Loading tokenizer from: '{tokenizer_path}'...")
    try:
        # Load the fast tokenizer from the specified path
        tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_path))
        print("Tokenizer loaded successfully.\n")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    # --- Words to Test ---
    # A list of tuples: (description, word_or_phrase)
    words_to_test = [
        ("Simple Word (Root)", "dharma"),
        ("Word with Grammatical Ending", "devānām"),
        ("Another word with Ending", "gacchanti"),
        ("Compound Word (Sandhi)", "maharṣir"),
        ("Slightly more complex Sandhi", "ityuvāca"), # iti + uvāca
        ("A short phrase", "sa maharṣir uvāca anena vacanena")
    ]

    print("--- Tokenization Examples ---")
    
    for description, word in words_to_test:
        # Use the .tokenize() method to get the list of tokens
        tokens = tokenizer.tokenize(word)
        
        print(f"Original ({description}):")
        print(f"  '{word}'")
        print("Tokenized:")
        # The '##' prefix indicates a subword that is part of a previous word
        print(f"  {tokens}\n" + "-"*40)

    print("\nAnalysis:")
    print("Notice how the tokenizer splits words like 'devānām' into a root ('devā') and an ending ('##nām').")
    print("This is crucial because it shows the tokenizer has learned the morphological structure of Sanskrit,")
    print("rather than just memorizing a large vocabulary of full words.")
    print("The '##' prefix indicates a subword that continues the previous token.")


if __name__ == "__main__":
    test_tokenizer(TOKENIZER_PATH)



