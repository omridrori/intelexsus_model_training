
import os
import re
from nltk.corpus import words
import nltk
from tqdm import tqdm

def setup_nltk_words():
    """
    Checks if the NLTK 'words' corpus is available, and if not, downloads it.
    """
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        print("Downloading NLTK 'words' corpus...")
        nltk.download('words')
    return set(words.words())

def contains_translit_chars(word: str) -> bool:
    """
    Checks if a word contains specific transliteration characters for Sanskrit/Tibetan.

    Args:
        word: The word to check.

    Returns:
        True if the word contains at least one transliteration character, False otherwise.
    """
    # This regex looks for common diacritics and special characters in IAST and other systems.
    # ā, ī, ū, ṛ, ṝ, ḷ, ḹ, ṃ, ḥ, ś, ṣ, ñ, ṭ, ḍ, ṇ, '
    translit_pattern = re.compile(r"[āīūṛṝḷḹṃḥśṣñṭḍṇ']")
    return translit_pattern.search(word) is not None

def filter_english_from_file(input_path: str, output_path: str, english_words: set, threshold: float):
    """
    Reads a file, filters out lines that are predominantly English, and writes the
    result to a new file.

    Args:
        input_path: Path to the source text file.
        output_path: Path to write the filtered text file.
        english_words: A set of English words for checking.
        threshold: The ratio of English words above which a line is removed.
    """
    print(f"\nProcessing file: {os.path.basename(input_path)}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    removed_lines_count = 0
    total_lines = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        lines = infile.readlines()
        total_lines = len(lines)

        for line in tqdm(lines, desc="Filtering lines"):
            line = line.strip()
            if not line:
                continue

            words_in_line = line.split()
            total_words = len(words_in_line)
            if total_words == 0:
                continue

            english_word_count = 0
            for word in words_in_line:
                # A word is not English if it has transliteration characters.
                if contains_translit_chars(word):
                    continue
                # Otherwise, check if its lowercase version is in the English dictionary.
                if word.lower() in english_words:
                    english_word_count += 1
            
            english_ratio = english_word_count / total_words

            if english_ratio >= threshold:
                removed_lines_count += 1
            else:
                outfile.write(line + '\n')

    print(f"Processing complete for {os.path.basename(input_path)}.")
    print(f"Total lines processed: {total_lines}")
    print(f"Lines identified as predominantly English and removed: {removed_lines_count}")
    print(f"Filtered content saved to: {output_path}")


def main():
    """
    Main function to orchestrate the filtering process.
    """
    print("Starting the English line filtering process...")
    
    # Setup NLTK
    english_words = setup_nltk_words()

    # Define paths
    base_dir = os.path.join("sandbox", "02_data_normalization")
    input_dir = os.path.join(base_dir, "outputs")
    output_dir = input_dir # Save to the same directory

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    files_to_process = {
        "sanskrit": "sanskrit_cleaned.txt",
        "tibetan": "tibetan_cleaned.txt"
    }
    
    # Define the threshold for removing a line
    ENGLISH_RATIO_THRESHOLD = 0.8

    for lang, filename in files_to_process.items():
        input_file_path = os.path.join(input_dir, filename)
        output_filename = f"{lang}_no_english.txt"
        output_file_path = os.path.join(output_dir, output_filename)
        
        filter_english_from_file(
            input_path=input_file_path,
            output_path=output_file_path,
            english_words=english_words,
            threshold=ENGLISH_RATIO_THRESHOLD
        )

if __name__ == "__main__":
    main() 