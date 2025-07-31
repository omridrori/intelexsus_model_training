import json
from pathlib import Path
import re
import string
from tqdm import tqdm

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

SANSKRIT_PATH = DATA_ROOT / "sanskrit"
TIBETAN_PATH = DATA_ROOT / "tibetan"

SANSKRIT_OUTPUT_FILE = OUTPUT_DIR / "sanskrit_cleaned.txt"
TIBETAN_OUTPUT_FILE = OUTPUT_DIR / "tibetan_cleaned.txt"

# This is our strict definition of what's allowed.
STRICT_IAST_CHARS = "āīūṛṝḷḹṃḥṅñṭḍṇśṣĀĪŪṚṜḶḸṀḤṄÑṬḌṆŚṢ"
ALLOWED_CHARS = set(
    string.ascii_letters + string.digits + string.punctuation + string.whitespace + STRICT_IAST_CHARS
)

def is_word_clean(word: str) -> bool:
    """Checks if a word contains ONLY allowed transliteration characters."""
    return all(char in ALLOWED_CHARS for char in word)

def is_sentence_clean(sentence: str) -> bool:
    """Checks if all words in a sentence are clean."""
    words = sentence.split()
    if not words:
        return True # An empty sentence is considered clean
    return all(is_word_clean(word) for word in words)

def process_corpus(corpus_path: Path, output_file: Path, language_name: str, sentence_delimiter_regex: str):
    """
    Reads a corpus, filters it sentence by sentence using a specific delimiter,
    and writes the clean text to an output file.
    """
    print(f"\n--- Processing {language_name} corpus... ---")
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_files = list(corpus_path.glob("*.jsonl"))

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for file_path in tqdm(all_files, desc=f"Filtering {language_name}"):
            with open(file_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        if not text:
                            continue

                        # Split text into sentences using the language-specific delimiter
                        sentences = re.split(sentence_delimiter_regex, text)
                        
                        clean_sentences = []
                        for sentence in sentences:
                            if sentence and is_sentence_clean(sentence):
                                clean_sentences.append(sentence.strip())
                        
                        # Join the clean sentences back together and write to file
                        if clean_sentences:
                            clean_text = " ".join(clean_sentences)
                            f_out.write(clean_text + "\n")
                            
                    except json.JSONDecodeError:
                        continue

    print(f"✅ Finished processing {language_name}. Cleaned corpus saved to: {output_file}")


if __name__ == "__main__":
    # For Sanskrit, we assume standard Western punctuation is used in transliteration.
    sanskrit_delimiters = r'(?<=[.|!?])\s*'
    process_corpus(SANSKRIT_PATH, SANSKRIT_OUTPUT_FILE, "Sanskrit", sanskrit_delimiters)

    # For Tibetan, the primary sentence delimiter is the Shad character '།'.
    tibetan_delimiters = r'(?<=།)\s*'
    process_corpus(TIBETAN_PATH, TIBETAN_OUTPUT_FILE, "Tibetan", tibetan_delimiters)
    
    print("\nAll tasks complete.") 