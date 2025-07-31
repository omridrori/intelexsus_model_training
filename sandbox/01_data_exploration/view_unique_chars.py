from pathlib import Path

# Define the path to the file we want to read
# It's in the 'outputs' subdirectory relative to this script
SCRIPT_DIR = Path(__file__).parent
file_to_read = SCRIPT_DIR / "outputs" / "unique_characters.txt"

def view_characters(file_path: Path):
    """
    Reads and prints the content of a given text file.

    Args:
        file_path: The path to the text file.
    """
    if not file_path.exists():
        print(f"Error: File not found at '{file_path}'")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print("--- Content of unique_characters.txt ---")
            print(content)
            print("------------------------------------------")
            print(f"Total unique characters found: {len(content)}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    view_characters(file_to_read) 