import json
from pathlib import Path

def explore_raw_data(min_char_length=500, max_char_length=1000, num_samples=3):
    """
    Reads and prints a few examples from the raw Sanskrit data.

    Args:
        min_char_length (int): Minimum character length for a sample to be considered.
        max_char_length (int): Maximum character length to print for each sample.
        num_samples (int): The number of samples to print.
    """
    # Assuming the script is run from the root of the project.
    # The path is relative to the project root.
    raw_data_dir = Path("data/sanskrit")
    
    if not raw_data_dir.exists():
        print(f"Directory not found: {raw_data_dir}")
        print("Please ensure you are running this script from the project root,")
        print("and the raw data is located in 'data/sanskrit/'.")
        return

    sample_count = 0
    files_to_check = sorted(list(raw_data_dir.glob("*.jsonl")))

    if not files_to_check:
        print(f"No .jsonl files found in {raw_data_dir}")
        return

    print(f"--- Exploring raw data from: {files_to_check[0]} ---\n")

    for data_file in files_to_check:
        if sample_count >= num_samples:
            break
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    if sample_count >= num_samples:
                        break
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                        if text and len(text) >= min_char_length:
                            print(f"Example {sample_count + 1}:")
                            print(text[:max_char_length] + "..." if len(text) > max_char_length else text)
                            print("-" * 20)
                            sample_count += 1
                    except json.JSONDecodeError:
                        # Ignore lines that are not valid JSON
                        continue
        except Exception as e:
            print(f"Error reading file {data_file}: {e}")


if __name__ == "__main__":
    explore_raw_data()
