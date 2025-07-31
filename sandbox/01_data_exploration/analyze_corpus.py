import json
from pathlib import Path
from collections import Counter
import pandas as pd

# --- Configuration ---
# Get the absolute path of the script file itself
SCRIPT_DIR = Path(__file__).parent
# Get the project root directory (which is two levels up from the script's parent)
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# Define paths relative to the project root
DATA_ROOT = PROJECT_ROOT / "data"
SANSKRIT_PATH = DATA_ROOT / "sanskrit"
TIBETAN_PATH = DATA_ROOT / "tibetan"
OUTPUT_PATH = SCRIPT_DIR / "outputs"

# --- Main Analysis Logic ---

def analyze_corpus(corpus_path: Path, language_name: str) -> dict:
    """
    Analyzes all .jsonl files in a given directory.

    Args:
        corpus_path: Path to the directory containing .jsonl files.
        language_name: Name of the language for reporting.

    Returns:
        A dictionary containing aggregated statistics for the corpus.
    """
    print(f"--- Analyzing {language_name} corpus at: {corpus_path} ---")
    
    all_files = list(corpus_path.glob("*.jsonl"))
    if not all_files:
        print(f"Warning: No .jsonl files found in {corpus_path}")
        return None

    total_docs = 0
    total_chars = 0
    char_counter = Counter()
    doc_lengths = []

    for file_path in all_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # IMPORTANT: Change 'text' if the key containing the text is different in your JSONL files.
                    # For example, if the text is under a key named "content", change it to:
                    # text = json.loads(line).get("content", "")
                    data = json.loads(line)
                    text = data.get("text", "") 
                    
                    if text:
                        total_docs += 1
                        doc_length = len(text)
                        total_chars += doc_length
                        doc_lengths.append(doc_length)
                        char_counter.update(text)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON line in file {file_path}")
                    continue
    
    stats = {
        "language": language_name,
        "file_count": len(all_files),
        "total_docs": total_docs,
        "total_chars": total_chars,
        "char_counter": char_counter,
        "doc_lengths": doc_lengths,
    }
    
    print(f"Analysis complete for {language_name}.")
    return stats

def generate_report(all_stats: list):
    """Generates and prints a summary report and saves outputs."""
    
    print("\n\n--- AGGREGATED CORPUS REPORT ---")
    
    combined_char_counter = Counter()
    df_data = []

    for stats in all_stats:
        if not stats: continue
        
        combined_char_counter.update(stats["char_counter"])
        
        # Calculate descriptive statistics for document lengths
        lengths_series = pd.Series(stats["doc_lengths"])
        length_stats = lengths_series.describe()
        
        report_line = {
            "Language": stats["language"],
            "Files": stats["file_count"],
            "Documents": f"{stats['total_docs']:,}",
            "Characters": f"{stats['total_chars']:,}",
            "Unique Chars": f"{len(stats['char_counter']):,}",
            "Avg. Doc Length": f"{length_stats.get('mean', 0):.2f}",
            "Max Doc Length": f"{int(length_stats.get('max', 0)):,}",
        }
        df_data.append(report_line)

    # Print summary table
    summary_df = pd.DataFrame(df_data)
    print(summary_df.to_string(index=False))

    # Combined stats
    total_unique_chars = len(combined_char_counter)
    print(f"\nTotal unique characters across all corpora: {total_unique_chars:,}")

    # Save unique characters to a file
    output_char_file = OUTPUT_PATH / "unique_characters.txt"
    sorted_chars = sorted(combined_char_counter.keys())
    with open(output_char_file, 'w', encoding='utf-8') as f:
        f.write("".join(sorted_chars))
    print(f"\nSaved all unique characters to: {output_char_file}")


if __name__ == "__main__":
    # Ensure output directory exists
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    sanskrit_stats = analyze_corpus(SANSKRIT_PATH, "Sanskrit")
    tibetan_stats = analyze_corpus(TIBETAN_PATH, "Tibetan")
    
    all_stats_list = [s for s in [sanskrit_stats, tibetan_stats] if s]
    
    if all_stats_list:
        generate_report(all_stats_list)
    else:
        print("\nNo data was analyzed. Please check your paths and file locations.") 