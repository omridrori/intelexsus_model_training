import json
import re
from pathlib import Path
import statistics
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# --- Settings ---
# !!! Update this path to your raw Sanskrit data directory !!!
RAW_DATA_PATH = Path("./data/sanskrit")
OUTPUT_IMAGE_FILE = "verse_length_histogram.png"
# ---------------------------------------------------------------------------


def plot_histogram(lengths: list, median_val: float, filename: str):
    """
    Generates and saves a histogram of the verse lengths.
    """
    plt.figure(figsize=(12, 6))
    
    # Create the histogram. Using a log scale for the y-axis helps visualize
    # the distribution when there are many short verses.
    plt.hist(lengths, bins=100, color='skyblue', edgecolor='black', alpha=0.7, range=(0, 200))
    plt.yscale('log')

    # Add a vertical line for the median
    plt.axvline(median_val, color='red', linestyle='dashed', linewidth=2)
    plt.text(median_val + 5, plt.ylim()[1]*0.5, f'Median: {median_val}', color = 'red')

    # Add titles and labels for clarity
    plt.title('Distribution of Sanskrit Verse Lengths (Log Scale)', fontsize=16)
    plt.xlabel('Number of Words per Verse', fontsize=12)
    plt.ylabel('Frequency (Log Scale)', fontsize=12)
    plt.grid(axis='y', alpha=0.5)

    # Save the plot to a file
    plt.savefig(filename)
    print(f"\nHistogram saved successfully as '{filename}'")


def analyze_verses(data_path: Path):
    """
    Scans JSONL files, segments them into verses, and analyzes their length distribution.
    """
    if not data_path.exists():
        print(f"Error: Directory not found at {data_path}")
        return

    all_verse_lengths = []
    all_files = list(data_path.rglob("*.jsonl"))

    print(f"Found {len(all_files)} files to analyze...")

    for file_path in tqdm(all_files, desc="Processing files"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    text = json.loads(line).get("text", "")
                    if not text or not text.strip():
                        continue

                    segment_pattern = r"\s*\|\|\s*|\s*//\s*|\||\."
                    segmented_text = re.sub(segment_pattern, "\n", text)
                    verses = [v.strip() for v in segmented_text.splitlines() if v.strip()]

                    for verse in verses:
                        word_count = len(verse.split())
                        if word_count > 0:
                            all_verse_lengths.append(word_count)
                except json.JSONDecodeError:
                    continue

    if not all_verse_lengths:
        print("No verses found to analyze.")
        return

    # Calculate statistics
    total_verses = len(all_verse_lengths)
    min_len = min(all_verse_lengths)
    max_len = max(all_verse_lengths)
    mean_len = statistics.mean(all_verse_lengths)
    median_len = statistics.median(all_verse_lengths)

    short_verses_under_5 = sum(1 for length in all_verse_lengths if length < 5)
    short_verses_under_10 = sum(1 for length in all_verse_lengths if length < 10)

    percent_under_5 = (short_verses_under_5 / total_verses) * 100
    percent_under_10 = (short_verses_under_10 / total_verses) * 100

    # Print results
    print("\n--- Verse Length Analysis Results ---")
    print(f"Total Verses Analyzed: {total_verses:,}")
    print("-" * 35)
    print(f"Min length:    {min_len} words")
    print(f"Max length:    {max_len:,} words")
    print(f"Mean length:   {mean_len:.2f} words")
    print(f"Median length: {median_len} words")
    print("-" * 35)
    print(f"Percentage of verses with < 5 words:  {percent_under_5:.2f}%")
    print(f"Percentage of verses with < 10 words: {percent_under_10:.2f}%")
    print("-------------------------------------\n")

    # Generate and save the histogram
    plot_histogram(all_verse_lengths, median_len, OUTPUT_IMAGE_FILE)


if __name__ == "__main__":
    analyze_verses(RAW_DATA_PATH)


