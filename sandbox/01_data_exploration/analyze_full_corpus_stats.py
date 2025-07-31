import json
from pathlib import Path
from tqdm import tqdm
import unicodedata

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

CORPORA = {
    "Sanskrit": DATA_ROOT / "sanskrit",
    "Tibetan": DATA_ROOT / "tibetan"
}

def get_corpus_stats(corpus_path: Path):
    """
    Performs a comprehensive analysis of a corpus, calculating various metrics
    related to size, content, and overhead.
    """
    stats = {
        "total_files": 0,
        "total_size_bytes": 0,
        "total_lines": 0,
        "total_words": 0,
        "total_chars_in_text": 0,
        "total_encoded_bytes_of_text": 0,
        "total_control_chars_in_text": 0,
        "total_whitespace_chars_in_text": 0,
    }

    all_files = list(corpus_path.glob("**/*.jsonl"))
    if not all_files:
        return None
    
    stats["total_files"] = len(all_files)

    for file_path in tqdm(all_files, desc=f"Analyzing {corpus_path.name}"):
        stats["total_size_bytes"] += file_path.stat().st_size
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stats["total_lines"] += 1
                try:
                    data = json.loads(line)
                    text = data.get("text")
                    if not text or not isinstance(text, str):
                        continue

                    stats["total_chars_in_text"] += len(text)
                    stats["total_words"] += len(text.split())
                    stats["total_encoded_bytes_of_text"] += len(text.encode('utf-8'))
                    
                    for char in text:
                        if unicodedata.category(char)[0] == 'C':
                            stats["total_control_chars_in_text"] += 1
                        elif char.isspace():
                            stats["total_whitespace_chars_in_text"] += 1
                            
                except (json.JSONDecodeError, TypeError):
                    continue
    return stats

def print_report(sanskrit_stats, tibetan_stats):
    """Prints a detailed, side-by-side comparison report."""
    
    def format_bytes(byte_count):
        if byte_count is None: return "N/A"
        if byte_count > 1024**3:
            return f"{byte_count / 1024**3:.2f} GB"
        if byte_count > 1024**2:
            return f"{byte_count / 1024**2:.2f} MB"
        if byte_count > 1024:
            return f"{byte_count / 1024:.2f} KB"
        return f"{byte_count} Bytes"

    print("\n--- Comprehensive Corpus Analysis Report ---")
    print("-" * 80)
    print(f"{'Metric':<35} | {'Sanskrit':>20} | {'Tibetan':>20}")
    print("=" * 80)

    # File System Metrics
    print(f"{'Total Files':<35} | {sanskrit_stats['total_files']:>20,} | {tibetan_stats['total_files']:>20,}")
    print(f"{'Total Size on Disk':<35} | {format_bytes(sanskrit_stats['total_size_bytes']):>20} | {format_bytes(tibetan_stats['total_size_bytes']):>20}")
    
    # Content Metrics
    print("-" * 80)
    print(f"{'Total Lines (JSON objects)':<35} | {sanskrit_stats['total_lines']:>20,} | {tibetan_stats['total_lines']:>20,}")
    print(f"{'Total Words in Text':<35} | {sanskrit_stats['total_words']:>20,} | {tibetan_stats['total_words']:>20,}")
    print(f"{'Total Chars in Text':<35} | {sanskrit_stats['total_chars_in_text']:>20,} | {tibetan_stats['total_chars_in_text']:>20,}")

    # Byte & Encoding Metrics
    print("-" * 80)
    s_text_bytes = sanskrit_stats['total_encoded_bytes_of_text']
    t_text_bytes = tibetan_stats['total_encoded_bytes_of_text']
    print(f"{'Size of Text Content (UTF-8)':<35} | {format_bytes(s_text_bytes):>20} | {format_bytes(t_text_bytes):>20}")
    
    # Derived Ratios & Percentages
    print("-" * 80)
    s_bpc = s_text_bytes / sanskrit_stats['total_chars_in_text'] if sanskrit_stats['total_chars_in_text'] else 0
    t_bpc = t_text_bytes / tibetan_stats['total_chars_in_text'] if tibetan_stats['total_chars_in_text'] else 0
    print(f"{'Avg. Bytes per Character':<35} | {s_bpc:>20.2f} | {t_bpc:>20.2f}")

    s_cpw = sanskrit_stats['total_chars_in_text'] / sanskrit_stats['total_words'] if sanskrit_stats['total_words'] else 0
    t_cpw = tibetan_stats['total_chars_in_text'] / tibetan_stats['total_words'] if tibetan_stats['total_words'] else 0
    print(f"{'Avg. Chars per Word':<35} | {s_cpw:>20.2f} | {t_cpw:>20.2f}")

    # THE KEY METRIC: OVERHEAD
    s_overhead = (sanskrit_stats['total_size_bytes'] - s_text_bytes) / sanskrit_stats['total_size_bytes'] * 100 if sanskrit_stats['total_size_bytes'] else 0
    t_overhead = (tibetan_stats['total_size_bytes'] - t_text_bytes) / tibetan_stats['total_size_bytes'] * 100 if tibetan_stats['total_size_bytes'] else 0
    print(f"{'File Size Overhead %':<35} | {s_overhead:>19.2f}% | {t_overhead:>19.2f}%")
    
    # Junk Data Metrics
    s_junk = sanskrit_stats['total_control_chars_in_text'] / sanskrit_stats['total_chars_in_text'] * 100 if sanskrit_stats['total_chars_in_text'] else 0
    t_junk = tibetan_stats['total_control_chars_in_text'] / tibetan_stats['total_chars_in_text'] * 100 if tibetan_stats['total_chars_in_text'] else 0
    print(f"{'Control (Junk) Chars in Text %':<35} | {s_junk:>19.4f}% | {t_junk:>19.4f}%")
    print("=" * 80)


if __name__ == "__main__":
    sanskrit_stats = get_corpus_stats(CORPORA["Sanskrit"])
    tibetan_stats = get_corpus_stats(CORPORA["Tibetan"])

    if sanskrit_stats and tibetan_stats:
        print_report(sanskrit_stats, tibetan_stats)
    else:
        print("Could not generate report due to missing data.") 