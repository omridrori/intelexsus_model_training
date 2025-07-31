import logging
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_analysis_for_language(language: str, input_file: Path, output_dir: Path, script_path: Path):
    """
    Runs the character frequency analysis script for a specific language.

    Args:
        language: The name of the language (e.g., "Sanskrit").
        input_file: Path to the language-specific corpus file.
        output_dir: The directory to save the report in.
        script_path: Path to the analysis script to run.
    """
    logging.info(f"--- Running analysis for {language} ---")
    
    if not input_file.exists():
        logging.error(f"Input file for {language} not found at: {input_file}")
        return

    output_file = output_dir / f"{language.lower()}_char_frequency.tsv"
    
    command = [
        sys.executable,  # Use the same python interpreter that is running this script
        str(script_path),
        "--input-file", str(input_file),
        "--output-file", str(output_file),
    ]

    try:
        logging.info(f"Executing command: {' '.join(command)}")
        # Using capture_output=True to get stdout/stderr for logging
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        logging.info(f"Successfully generated report for {language}. Output at: {output_file}")
        # Log stdout from the script which contains progress info
        if result.stdout:
            logging.info(f"--- Script output for {language} ---\n{result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to generate report for {language}.")
        logging.error(f"Return code: {e.returncode}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {language}: {e}")


def main():
    """
    Main function to orchestrate the analysis for all specified languages.
    """
    # Assuming this script is in sandbox/03_... and project root is 3 levels up
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Path to the analysis script
    analysis_script = project_root / "sandbox/03_character_frequency_analysis/plot_char_frequency.py"
    
    # Base directory for the cleaned data
    data_dir = project_root / "sandbox/02_data_normalization/outputs"
    
    # Directory for saving the reports
    output_dir = project_root / "sandbox/03_character_frequency_analysis/outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the languages and their corresponding data files
    languages = {
        "Sanskrit": data_dir / "sanskrit_cleaned.txt",
        "Tibetan": data_dir / "tibetan_cleaned.txt",
    }

    for lang, file_path in languages.items():
        run_analysis_for_language(lang, file_path, output_dir, analysis_script)

    logging.info("--- All analyses complete. ---")


if __name__ == "__main__":
    main() 