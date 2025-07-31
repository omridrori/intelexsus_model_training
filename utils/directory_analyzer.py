import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_directory(target_dir: str, output_dir_name: str = "contents"):
    """
    Analyzes a directory, collecting content from Python files and paths from other files.

    Args:
        target_dir (str): The path to the directory to analyze.
        output_dir_name (str, optional): The name of the subdirectory to save the output. Defaults to "contents".
    """
    target_path = Path(target_dir)
    if not target_path.is_dir():
        logging.error(f"Error: Directory not found at '{target_dir}'")
        return

    # Create the output directory inside the target directory
    output_path = target_path / output_dir_name
    try:
        output_path.mkdir(exist_ok=True)
        logging.info(f"Created output directory at: {output_path}")
    except OSError as e:
        logging.error(f"Failed to create output directory: {e}")
        return

    all_files = list(target_path.rglob("*"))
    files_to_process = [f for f in all_files if f.is_file()]
    
    if not files_to_process:
        logging.warning(f"No files found in '{target_dir}'.")
        return

    output_content = []

    logging.info(f"Analyzing directory: {target_dir}")
    for file_path in tqdm(files_to_process, desc="Analyzing files", unit="file"):
        relative_path = file_path.relative_to(target_path)

        if file_path.suffix == '.py':
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                output_content.append(f"--- Python File: {relative_path} ---\n")
                output_content.append(content)
                output_content.append("\n" + "-"*40 + "\n")
            except Exception as e:
                logging.warning(f"Could not read Python file {relative_path}: {e}")
        else:
            output_content.append(f"****{relative_path}****\n")

    # Write the collected content to a text file
    output_file_path = output_path / 'directory_contents.txt'
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_content))
        logging.info(f"Successfully wrote analysis to {output_file_path}")
    except IOError as e:
        logging.error(f"Failed to write to output file: {e}")


def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze a directory and collect file contents.")
    parser.add_argument("target_dir", type=str, help="The path to the directory to analyze.")
    args = parser.parse_args()

    analyze_directory(args.target_dir)

if __name__ == "__main__":
    main() 