import os

def print_file_head(file_path, num_lines=200):
    """
    Prints the first few lines of a file.

    Args:
        file_path (str): The path to the file.
        num_lines (int): The number of lines to print.
    """
    try:
        # Construct the absolute path to the file
        # The script is in sandbox/utils, so we need to go up two directories
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        absolute_file_path = os.path.join(base_dir, file_path)

        with open(absolute_file_path, 'r', encoding='utf-8') as f:
            print(f"--- First {num_lines} lines of {absolute_file_path} ---")
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                print(line.strip())
            print("--- End of sample ---")
    except FileNotFoundError:
        print(f"Error: File not found at {absolute_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # The file path is relative to the project root
    file_to_read = os.path.join('data', 'sanskrit_preprocessed', 'sanskrit_latin_only.txt')
    print_file_head(file_to_read) 