import json
import os

def print_first_n_examples(file_path, n=5):
    """
    Reads and prints the first n JSON objects from a JSONL file.

    Args:
        file_path (str): The path to the JSONL file.
        n (int): The number of examples to print.
    """
    print(f"Reading data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                try:
                    data = json.loads(line)
                    print(f"--- Example {i+1} ---")
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                    print("\\n")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON on line {i+1}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Assuming the script is run from the root of the project
    file_to_read = os.path.join("data", "tibetan", "Tibetan_1.jsonl")
    print_first_n_examples(file_to_read)

