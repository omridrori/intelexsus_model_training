from pathlib import Path

# --- 1. Settings ---

# Path to the BERT-ready data file
DATA_FILE = Path("data/sanskrit_preprocessed/sanskrit_bert_ready_512.txt")

# Number of lines to print
NUM_LINES_TO_PRINT = 10

# --- 2. Print function ---

def print_bert_data_sample():
    """
    Prints the first n lines from the BERT-ready data file to inspect the format.
    """
    print("--- Printing BERT Data Sample ---")
    
    # Check if file exists
    if not DATA_FILE.exists():
        print(f"❌ ERROR: Data file not found at '{DATA_FILE}'")
        print("Please run the group_verses_for_bert.py script first to create this file.")
        return
    
    # Get file size for information
    file_size = DATA_FILE.stat().st_size
    print(f"File size: {file_size:,} bytes")
    
    # Count total lines in file
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total lines in file: {total_lines:,}")
    print(f"Printing first {NUM_LINES_TO_PRINT} lines:\n")
    print("=" * 80)
    
    # Print the first n lines
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i > NUM_LINES_TO_PRINT:
                break
            
            # Clean the line and show line number
            clean_line = line.strip()
            print(f"Line {i}:")
            print(f"Length: {len(clean_line)} characters")
            print(f"Content: {clean_line}")
            print("-" * 40)
    
    print("=" * 80)
    print(f"✅ Printed {min(NUM_LINES_TO_PRINT, total_lines)} lines from '{DATA_FILE}'")

if __name__ == "__main__":
    print_bert_data_sample() 