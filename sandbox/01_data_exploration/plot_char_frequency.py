import argparse
import collections
import logging
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_character_frequency_report(input_file: Path, output_file: Path):
    """
    Analyzes the character frequency in a text file and saves it to a TSV file.

    Args:
        input_file: Path to the input text file.
        output_file: Path to save the output TSV report.
    """
    logging.info(f"Starting character frequency analysis for {input_file}...")

    if not input_file.exists():
        logging.error(f"Input file not found: {input_file}")
        return

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Use a Counter to store character frequencies
    char_counts = collections.Counter()

    # Get file size for tqdm progress bar
    file_size = os.path.getsize(input_file)
    
    logging.info("Reading file and counting characters...")
    with open(input_file, 'r', encoding='utf-8') as f, tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Processing {input_file.name}") as pbar:
        for line in f:
            char_counts.update(line)
            pbar.update(len(line.encode('utf-8')))

    logging.info(f"Found {len(char_counts)} unique characters.")

    # Get all characters, sorted by frequency
    all_common_chars = char_counts.most_common()
    
    if not all_common_chars:
        logging.warning("No characters found to save.")
        return
        
    df = pd.DataFrame(all_common_chars, columns=['Character', 'Frequency'])

    # Handle special whitespace characters for clarity in the output file
    df['Character'] = df['Character'].replace({
        ' ': '<space>',
        '\n': '<newline>',
        '\t': '<tab>'
    })

    logging.info(f"Saving character frequency report to {output_file}...")
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    logging.info(f"Report saved successfully to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate a character frequency report from a text file.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input text file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the output .tsv report.")
    
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    save_character_frequency_report(input_path, output_path)

if __name__ == "__main__":
    main() 