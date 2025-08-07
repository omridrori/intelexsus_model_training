import os
import argparse

def write_python_files_to_file(root_dir, output_file_handle):
    """
    Recursively finds all Python files in a directory, writes their path and content to a file.

    Args:
        root_dir (str): The path to the directory to search.
        output_file_handle: The file handle to write the output to.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                print(f"{file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        output_file_handle.write(f"{content}\n")
                except Exception as e:
                    output_file_handle.write(f"Error reading file {file_path}: {e}\n")
                output_file_handle.write("****\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write all python files' content from a directory to an output file.")
    parser.add_argument('directory', type=str, nargs='?', default='.',
                        help='The directory to scan for python files. Defaults to the current directory.')
    parser.add_argument('--output', type=str, default='sandbox/utils/python_files_content.txt',
                        help='The path for the output file. Defaults to sandbox/utils/python_files_content.txt')

    args = parser.parse_args()

    target_directory = args.directory
    output_file_path = args.output

    if not os.path.isdir(target_directory):
        print(f"Error: Directory '{target_directory}' not found.")
    else:
        try:
            # Create the directory for the output file if it doesn't exist
            output_dir = os.path.dirname(output_file_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                write_python_files_to_file(target_directory, f_out)
            print(f"Successfully wrote content to {output_file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
