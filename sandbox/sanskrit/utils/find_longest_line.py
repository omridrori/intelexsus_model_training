from pathlib import Path

# --- Settings ---
# Path to the final data file
DATA_FILE = Path("data/sanskrit_preprocessed/sanskrit_bert_ready_512.txt")

def find_and_print_longest_line(file_path):
    """
    Finds the longest line in a text file and prints it.
    """
    if not file_path.exists():
        print(f"❌ ERROR: File not found at '{file_path}'")
        return

    print(f"Scanning '{file_path}' to find the longest line...")

    longest_line = ""
    max_length = 0
    line_number = 0
    current_line_number = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            current_line_number += 1
            # Use len() to measure the line length in characters
            if len(line) > max_length:
                max_length = len(line)
                longest_line = line
                line_number = current_line_number
    
    print("\n--- Longest Line Found ---")
    print(f"Line Number: {line_number}")
    print(f"Length (in characters): {max_length:,}")
    print("-" * 30)
    print("Content:")
    print(longest_line)


if __name__ == "__main__":
    find_and_print_longest_line(DATA_FILE) 