import os

def find_character_in_context(file_path, char_to_find, limit=10):
    """
    Finds a specific character in a file and prints the line containing it,
    along with the preceding and succeeding lines.

    Args:
        file_path (str): The path to the text file.
        char_to_find (str): The character to search for.
        limit (int): The maximum number of occurrences to find.
    """
    print(f"Searching for '{char_to_find}' in '{file_path}'...\n")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    found_count = 0
    for i, line in enumerate(lines):
        if char_to_find in line:
            if found_count >= limit:
                break
            
            print(f"--- Occurrence #{found_count + 1} ---")
            
            # Print the previous line if it exists
            if i > 0:
                print(f"Line {i}: {lines[i-1].strip()}")
            
            # Print the current line
            print(f"Line {i+1} (match): {line.strip()}")
            
            # Print the next line if it exists
            if i < len(lines) - 1:
                print(f"Line {i+2}: {lines[i+1].strip()}")
            
            print("-" * (len(str(found_count + 1)) + 20))
            print() # Add a blank line for readability
            
            found_count += 1

    if found_count == 0:
        print(f"The character '{char_to_find}' was not found in the file.")
    else:
        print(f"\nFinished. Found {found_count} occurrences.")


if __name__ == "__main__":
    # The user wants to search within the sandbox directory structure
    # The project root is the parent of the 'sandbox' directory.
    # We construct the path relative to the script's location.
    # This script is in: sandbox/01_data_exploration/
    # The target file is in: sandbox/02_data_normalization/outputs/
    
    # .. moves up to sandbox/, .. moves up to project root
    base_path = os.path.join(os.path.dirname(__file__), '..', '..')
    
    # Construct the final path
    input_file = os.path.join(base_path, 'sandbox', '02_data_normalization', 'outputs', 'tibetan_cleaned.txt')
    
    # Normalize path for Windows
    input_file = os.path.normpath(input_file)
    
    character_to_find = 'ā'
    find_character_in_context(input_file, character_to_find, limit=10) 