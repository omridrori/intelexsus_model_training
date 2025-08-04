# sanskrit preprocessing

from pathlib import Path
import json
import re
import string
from tqdm import tqdm
import nltk
from nltk.corpus import words
import datetime

# --- 1. Settings (with fixes) ---
RAW_DATA_PATH = Path("data/sanskrit") 
OUTPUT_FILE = Path("data/sanskrit_preprocessed/sanskrit_chunks_final_v2.txt")
REPORT_FILE = Path("data/sanskrit_preprocessed/sanskrit_processing_report_v2.txt")

CHUNK_TARGET_WORDS_MIN = 40
ENGLISH_THRESHOLD = 0.5 # ✨ Fix: Increased threshold to prevent false positives like "Hare Krishna"
MAX_EXAMPLES_PER_CATEGORY = 5

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# --- 2. Report Management Class (Logger) ---

class ReportLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.examples = {}
        # Open the report file in write mode with UTF-8 encoding
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(f"Sanskrit Processing Report - {datetime.datetime.now()}\n")
            f.write("="*60 + "\n\n")

    def _write(self, text):
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def header(self, title):
        self._write("\n" + "#" * 60)
        self._write(f"# {title.upper()}")
        self._write("#" * 60 + "\n")

    def log_example(self, category, data):
        if category not in self.examples:
            self.examples[category] = []
        if len(self.examples[category]) < MAX_EXAMPLES_PER_CATEGORY:
            self.examples[category].append(data)

    def write_report(self, stats):
        self.header("Final Statistics Summary")
        for key, value in stats.items():
            self._write(f"- {key.replace('_', ' ').title()}: {value:,}")

        self.header("Examples of Discarded Chunks")
        if 'discarded_non_latin' in self.examples:
            self._write("\n--- REASON: Contains Non-Latin/IAST Characters ---\n")
            for ex in self.examples['discarded_non_latin']:
                self._write(f"[ORIGINAL DOCUMENT CONTEXT]:\n{ex['original_doc'][:500]}...\n")
                self._write(f"[DISCARDED CHUNK]:\n{ex['chunk']}\n" + "-"*40)
        
        if 'discarded_english' in self.examples:
            self._write("\n--- REASON: High Percentage of English Words ---\n")
            for ex in self.examples['discarded_english']:
                self._write(f"[ORIGINAL DOCUMENT CONTEXT]:\n{ex['original_doc'][:500]}...\n")
                self._write(f"[DISCARDED CHUNK]:\n{ex['chunk']}\n" + "-"*40)

        self.header("Examples of Internal Cleaning")
        if 'cleaning' in self.examples:
            for ex in self.examples['cleaning']:
                self._write(f"[ORIGINAL DOCUMENT CONTEXT]:\n{ex['original_doc'][:500]}...\n")
                self._write(f"[CHUNK BEFORE CLEANING]:\n{ex['before']}\n")
                self._write(f"[CHUNK AFTER CLEANING]:\n{ex['after']}\n" + "-"*40)

        print(f"\n✅ Detailed report saved to: {self.file_path}")


# --- 3. Filtering Helper Functions (with fixes) ---
def setup_nltk_words():
    try: nltk.data.find('corpora/words')
    except LookupError: nltk.download('words', quiet=True)
    return set(words.words())

IAST_PATTERN = re.compile(r"[āīūṛṝḷḹṃḥṅñṭḍṇśṣĀĪŪṚṜḶḸṀḤṄÑṬḌṆŚṢ]")
IAST_CHARS = "āīūṛṝḷḹṃḥṅñṭḍṇśṣ"
# ✨ Fix: Added the '>' character and other common symbols
ALLOWED_PUNCTUATION = ".,;!?()[]{}'\"-|=_/>" 
ALLOWED_CHARS = set(string.ascii_lowercase + string.digits + ALLOWED_PUNCTUATION + IAST_CHARS + string.whitespace)

def is_chunk_latin_only(chunk: str) -> bool:
    for char in chunk.lower():
        if char not in ALLOWED_CHARS: return False
    return True

def is_likely_english(word: str, english_vocab: set) -> bool:
    if IAST_PATTERN.search(word): return False
    cleaned_word = word.strip(".,;!?()[]{}'\"-_")
    return cleaned_word.lower() in english_vocab


# --- 4. Main Processing Logic (no changes) ---

def get_all_verses_with_context(all_docs):
    """Breaks down documents into verses while preserving the original document context."""
    all_verses = []
    for doc in tqdm(all_docs, desc="Segmenting documents into verses"):
        full_text = doc
        segment_pattern = r'\s*\|\|\s*|\s*//\s*|\||\.'
        segmented_text = re.sub(segment_pattern, '\n', full_text)
        verses = [v.strip() for v in segmented_text.splitlines() if v.strip()]
        for verse in verses:
            all_verses.append({'text': verse, 'original_doc': doc})
    return all_verses

def group_into_chunks(verses):
    """Merges verses into chunks, while preserving context."""
    chunks = []
    current_chunk_verses = []
    current_word_count = 0
    for verse_data in tqdm(verses, desc="Grouping verses into chunks"):
        verse_words = verse_data['text'].split()
        if not verse_words: continue
        if current_word_count >= CHUNK_TARGET_WORDS_MIN:
            chunk_text = " ".join([v['text'] for v in current_chunk_verses])
            original_doc_context = current_chunk_verses[0]['original_doc']
            chunks.append({'text': chunk_text, 'original_doc': original_doc_context})
            current_chunk_verses = []
            current_word_count = 0
        current_chunk_verses.append(verse_data)
        current_word_count += len(verse_words)
    if current_chunk_verses:
        chunk_text = " ".join([v['text'] for v in current_chunk_verses])
        original_doc_context = current_chunk_verses[0]['original_doc']
        chunks.append({'text': chunk_text, 'original_doc': original_doc_context})
    return chunks

def process_chunks(chunks, english_vocab, logger):
    """Filters and cleans each chunk, and logs the actions in the report."""
    final_chunks = []
    stats = {k: 0 for k in ['total', 'discarded_non_latin', 'discarded_english', 'discarded_url', 'kept']}
    stats['total'] = len(chunks)

    for chunk_data in tqdm(chunks, desc="Filtering and Cleaning Chunks"):
        chunk = chunk_data['text']
        original_doc = chunk_data['original_doc']
        
        # Filtering
        if not is_chunk_latin_only(chunk):
            stats['discarded_non_latin'] += 1
            logger.log_example('discarded_non_latin', {'chunk': chunk, 'original_doc': original_doc})
            continue
        if re.search(r'https?://\S+', chunk):
            stats['discarded_url'] += 1
            logger.log_example('discarded_url', {'chunk': chunk, 'original_doc': original_doc})
            continue
        
        words = chunk.split()
        if not words: continue
        eng_count = sum(1 for w in words if is_likely_english(w, english_vocab))
        if (eng_count / len(words)) > ENGLISH_THRESHOLD:
            stats['discarded_english'] += 1
            logger.log_example('discarded_english', {'chunk': chunk, 'original_doc': original_doc})
            continue

        # Internal cleaning
        chunk_before_cleaning = chunk
        cleaned_chunk = chunk.lower()
        cleaned_chunk = re.sub(r'\b\d+(\.\d+)*\b', '', cleaned_chunk)
        cleaned_chunk = re.sub(r'\b[a-zāīūṛṝḷḹṃḥṅñṭḍṇśṣ]+\w*_\d+\s*//?', '', cleaned_chunk)
        cleaned_chunk = re.sub(r'\b\d+\b', '', cleaned_chunk)
        cleaned_chunk = re.sub(r'\(\d+\)', '', cleaned_chunk)
        cleaned_chunk = re.sub(r'\s+', ' ', cleaned_chunk).strip()

        if cleaned_chunk and len(cleaned_chunk.split()) > 1:
            final_chunks.append(cleaned_chunk)
            stats['kept'] += 1
            logger.log_example('cleaning', {'before': chunk_before_cleaning, 'after': cleaned_chunk, 'original_doc': original_doc})
            
    return final_chunks, stats

def main():
    logger = ReportLogger(REPORT_FILE)
    logger.header("Initialization")
    logger._write(f"Raw data path: {RAW_DATA_PATH}")
    logger._write(f"Final output file: {OUTPUT_FILE}")
    
    english_vocab = setup_nltk_words()

    all_docs = []
    all_files = list(RAW_DATA_PATH.rglob("*.jsonl"))
    for file_path in tqdm(all_files, desc="Loading raw documents"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    text = json.loads(line).get("text", "")
                    if text.strip(): all_docs.append(text)
                except json.JSONDecodeError: continue
    
    verses = get_all_verses_with_context(all_docs)
    initial_chunks = group_into_chunks(verses)
    final_clean_chunks, stats = process_chunks(initial_chunks, english_vocab, logger)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in final_clean_chunks: f.write(chunk + "\n")
    
    stats['documents_loaded'] = len(all_docs)
    stats['verses_extracted'] = len(verses)
    logger.write_report(stats)
    
if __name__ == "__main__":
    main()