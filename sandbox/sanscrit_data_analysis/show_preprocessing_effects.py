import json
import re
import string
from pathlib import Path

# --- Logic copied from sanscrit/preprocessing/chunking.py ---

# Note: In a real package, we would refactor to import these.
# For this script, copying is simpler to demonstrate the logic directly.

try:
    import nltk
    from nltk.corpus import words as nltk_words
    nltk.data.find("corpora/words")
    ENGLISH_VOCAB = set(nltk_words.words())
except LookupError:
    nltk.download("words", quiet=True)
    from nltk.corpus import words as nltk_words_after_download
    ENGLISH_VOCAB = set(nltk_words_after_download.words())


IAST_PATTERN = re.compile(r"[āīūṛṝḷḹṃḥṅñṭḍṇśṣĀĪŪṚṜḶḸṀḤṄÑṬḌṆŚṢ]")
IAST_CHARS = "āīūṛṝḷḹṃḥṅñṭḍṇśṣ"
ALLOWED_PUNCTUATION = ".,;!?()[]{}'\"-|=_/>"
ALLOWED_CHARS = set(string.ascii_lowercase + string.digits + ALLOWED_PUNCTUATION + IAST_CHARS + string.whitespace)
ENGLISH_THRESHOLD = 0.5

def _is_chunk_latin_only(chunk: str) -> bool:
    """Checks if a chunk contains only allowed Latin/IAST characters."""
    return all(char.lower() in ALLOWED_CHARS for char in chunk)

def find_first_non_latin_char(chunk: str) -> tuple[int, str] | None:
    """Finds the index and character of the first non-allowed char."""
    for i, char in enumerate(chunk):
        if char.lower() not in ALLOWED_CHARS:
            return i, char
    return None

def _is_likely_english(word: str) -> bool:
    """Checks if a single word is likely English."""
    if IAST_PATTERN.search(word):
        return False
    cleaned_word = word.strip(".,;!?()[]{}'\"-_")
    return cleaned_word.lower() in ENGLISH_VOCAB

def get_english_ratio(chunk: str) -> float:
    """Calculates the ratio of English words in a chunk."""
    words = chunk.split()
    if not words:
        return 0.0
    eng_count = sum(1 for w in words if _is_likely_english(w))
    return eng_count / len(words)

def apply_cleaning_regex(chunk: str) -> str:
    """Applies the series of cleaning regexes."""
    cleaned = chunk.lower()
    # Each tuple is (pattern, description)
    regex_rules = [
        (r"https?://\S+", "URL"),
        (r"\b\d+(\.\d+)*\b", "Multi-part numeric reference"),
        (r"\b[a-zāīūṛṝḷḹṃḥṅñṭḍṇśṣ]+\w*_\d+\s*//?", "ID-like pattern"),
        (r"\b\d+\b", "Standalone number"),
        (r"\(\d+\)", "Number in parentheses"),
    ]
    for pattern, _ in regex_rules:
        cleaned = re.sub(pattern, "", cleaned)
    
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

# --- End of copied logic ---


def display_snippet(text: str, index: int, max_len: int = 200) -> str:
    """Creates a snippet of text centered around a specific index."""
    start = max(0, index - max_len // 2)
    end = min(len(text), index + max_len // 2)
    snippet = text[start:end]
    if start > 0:
        snippet = "... " + snippet
    if end < len(text):
        snippet = snippet + " ..."
    return snippet.replace("\n", " ")


def find_examples():
    """
    Finds and prints examples of preprocessing changes on the raw Sanskrit data.
    """
    raw_data_dir = Path("data/sanskrit")
    if not raw_data_dir.exists():
        print(f"Directory not found: {raw_data_dir}")
        return

    examples_to_find = {
        "non_latin": None,
        "high_english_ratio": None,
        "url": None,
        "cleaning": None,
    }
    
    print("Searching for examples of preprocessing changes...\n")

    files_to_check = sorted(list(raw_data_dir.glob("*.jsonl")))
    for data_file in files_to_check:
        if all(examples_to_find.values()):
            break
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                if all(examples_to_find.values()):
                    break
                try:
                    text = json.loads(line).get("text", "")
                    if not text or len(text) < 20: # Skip very short texts
                        continue

                    # 1. Check for Non-Latin characters
                    if not examples_to_find["non_latin"] and not _is_chunk_latin_only(text):
                        char_info = find_first_non_latin_char(text)
                        if char_info:
                            index, char = char_info
                            examples_to_find["non_latin"] = {
                                "reason": f"Contains non-Latin/IAST characters (e.g., '{char}')",
                                "before": display_snippet(text, index),
                                "after": "CHUNK DISCARDED"
                            }
                            continue

                    # 2. Check for URLs
                    url_match = re.search(r"https?://\S+", text)
                    if not examples_to_find["url"] and url_match:
                         examples_to_find["url"] = {
                            "reason": "Contains a URL",
                            "before": display_snippet(text, url_match.start()),
                            "after": "CHUNK DISCARDED"
                        }
                         continue

                    # 3. Check for high English ratio
                    if not examples_to_find["high_english_ratio"]:
                        ratio = get_english_ratio(text)
                        if ratio > ENGLISH_THRESHOLD:
                            # Find first English word to center the snippet
                            first_english_word_idx = -1
                            english_words_found = []
                            words = text.split()
                            for w in words:
                                if _is_likely_english(w):
                                    english_words_found.append(w)
                                    try:
                                        first_english_word_idx = text.index(w)
                                        break
                                    except ValueError:
                                        continue
                            
                            snippet_center = first_english_word_idx if first_english_word_idx != -1 else 0
                            examples_to_find["high_english_ratio"] = {
                                "reason": f"High English word ratio ({ratio:.2f}). Found words: {english_words_found[:5]}",
                                "before": display_snippet(text, snippet_center),
                                "after": "CHUNK DISCARDED"
                            }
                            continue
                    
                    # 4. Check for general cleaning
                    cleaned_text = apply_cleaning_regex(text)
                    if not examples_to_find["cleaning"] and cleaned_text != text.lower().strip():
                        # Find first difference to center the snippet
                        first_diff = -1
                        for i, (c1, c2) in enumerate(zip(text.lower(), cleaned_text)):
                            if c1 != c2:
                                first_diff = i
                                break
                        if first_diff == -1 and len(cleaned_text) < len(text.lower()):
                            first_diff = len(cleaned_text)
                            
                        examples_to_find["cleaning"] = {
                            "reason": "General cleaning (removes numbers, normalizes space, etc.)",
                            "before": display_snippet(text, first_diff),
                            "after": display_snippet(cleaned_text, first_diff)
                        }

                except (json.JSONDecodeError, TypeError):
                    continue
    
    # --- Print found examples ---
    for key, data in examples_to_find.items():
        print(f"--- Example for: {key.replace('_', ' ').title()} ---")
        if data:
            print(f"Reason: {data['reason']}")
            print(f"Before: {data['before']}")
            print(f"After:  {data['after']}")
        else:
            print("No example found in the dataset.")
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    find_examples()
