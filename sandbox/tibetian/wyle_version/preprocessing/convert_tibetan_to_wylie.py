import sys
import os
import json
import re
from datetime import datetime
import pyewts

# --- Configuration: Path to the conversion tool ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERT_TOOL_PATH = os.path.join(SCRIPT_DIR, "preprocessing", "detect_and_convert")

if CONVERT_TOOL_PATH not in sys.path:
    sys.path.insert(0, CONVERT_TOOL_PATH)

try:
    from conversion.convertor import converter, conversionError as ConversionError
except ImportError as e:
    print(f"Error: Could not import the conversion tool from '{CONVERT_TOOL_PATH}'.")
    print(f"Please ensure the path is correct and dependencies are installed.")
    sys.exit(1)

# --- Regular Expressions ---
TIBETAN_UNICODE_RE = re.compile(r"[\u0F00-\u0FFF]")
TIBETAN_RATIO_THRESHOLD = 0.5  # Keep lines that are at least 50% Tibetan script
MINIMUM_LINE_LENGTH = 5        # Ignore very short lines

# WYLIE_RE has been removed to disable the strict quality check.

def extract_tibetan_lines(full_text: str) -> str:
    """Extract only lines that are primarily Tibetan based on ratio and length."""
    if not isinstance(full_text, str):
        return ""
    tibetan_lines = []
    for line in full_text.split('\n'):
        stripped = line.strip()
        if len(stripped) < MINIMUM_LINE_LENGTH:
            continue
        tib_chars = len(TIBETAN_UNICODE_RE.findall(line))
        if tib_chars / max(1, len(line)) >= TIBETAN_RATIO_THRESHOLD:
            tibetan_lines.append(line)
    return "\n".join(tibetan_lines)

def cleanup_wylie(text: str) -> str:
    """
    Cleans the raw Wylie output, preserving the shad (/) but removing
    decorative marks (@, #, _), brackets, page-number patterns, and
    standalone numeric tokens (Latin and Tibetan digits).
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove decorative symbols and underscores; KEEP shad (/)
    text = re.sub(r"[@#_]+", " ", text)

    # Remove bracketed content markers (but keep inside text already exposed)
    text = text.replace("[", "").replace("]", "")

    # Remove parenthesized numbers like (147) possibly with spaces
    text = re.sub(r"\(\s*\d+\s*\)", " ", text)

    # Remove standalone Latin-digit tokens
    text = re.sub(r"\b\d+\b", " ", text)

    # Remove standalone Tibetan-digit tokens (U+0F20–U+0F29) possibly repeated
    text = re.sub(r"(?<!\S)[\u0F20-\u0F29]+(?!\S)", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_tibetan_unicode(text: str) -> bool:
    """Check if the string contains a significant portion of Tibetan Unicode characters."""
    if not text:
        return False
    tibetan_chars = len(TIBETAN_UNICODE_RE.findall(text))
    return (tibetan_chars / len(text)) > 0.5


def render_progress(current: int, total: int, width: int = 40) -> None:
    if total <= 0: return
    fraction = min(1.0, current / total)
    filled = int(fraction * width)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rProgress: [{bar}] {current}/{total} ({fraction:.0%})", end="", flush=True)


def process_file(in_path: str, out_path: str, cv: pyewts.pyewts, stats: dict, tibetan_lines_counter: int):
    """Process one JSONL file, converting Tibetan Unicode to Wylie."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines_in_file = 0
    with open(in_path, 'r', encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        
        lines = fin.readlines()
        lines_in_file = len(lines)

        for i, line in enumerate(lines):
            try:
                obj = json.loads(line)
                text = obj.get("text", "")

                # Extract only Tibetan-heavy lines to avoid metadata
                extracted_text = extract_tibetan_lines(text)

                if extracted_text.strip():
                    stats['lines_processed'] += 1
                    tibetan_lines_counter += 1
                    try:
                        # Convert extracted Tibetan only
                        converted_text = cv.toWylie(extracted_text)
                        cleaned_text = cleanup_wylie(converted_text)

                        if tibetan_lines_counter % 10000 == 0:
                            print(f"\n--- Sample #{tibetan_lines_counter} (from {os.path.basename(in_path)}, line {i+1}) ---")
                            preview_src = extracted_text if len(extracted_text) <= 300 else extracted_text[:300] + "..."
                            preview_out = cleaned_text if len(cleaned_text) <= 300 else cleaned_text[:300] + "..."
                            print(f"  Original (extracted): {preview_src}")
                            print(f"  Converted          : {preview_out}")

                        obj['text'] = cleaned_text
                        stats['lines_converted'] += 1

                    except Exception as e:
                        stats['conversion_errors'] += 1
                        if len(stats['error_examples']) < 5:
                            stats['error_examples'].append({'file': in_path, 'line': i+1, 'text': extracted_text[:100], 'error': str(e)})
                else:
                    # No Tibetan content extracted; zero it out to avoid metadata
                    obj['text'] = ""

                line_out = json.dumps(obj, ensure_ascii=False) + "\n"
                fout.write(line_out)

            except json.JSONDecodeError:
                stats['json_errors'] += 1
                fout.write(line) # Write original line if JSON is invalid
    return lines_in_file, tibetan_lines_counter


def main():
    start_time = datetime.now()
    print("--- Starting Step 3: Convert Tibetan Unicode to Wylie ---")

    input_dir = os.path.join("cleaned_data", "step1_no_garbage")
    output_dir = os.path.join("cleaned_data", "step3_wylie_converted")
    report_path = os.path.join("cleaned_data", "reports", "step3_wylie_conversion_report.txt")

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: '{input_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    try:
        # Use the pyewts converter directly for more control
        cv = pyewts.pyewts()
    except Exception as e:
        print(f"Error initializing the converter: {e}")
        return

    files_to_process = [f for f in sorted(os.listdir(input_dir)) if f.endswith(".jsonl")]
    total_files = len(files_to_process)
    
    stats = {
        'total_lines': 0,
        'lines_processed': 0, # Lines identified as Tibetan Unicode
        'lines_converted': 0,
        'conversion_errors': 0,
        'json_errors': 0,
        'error_examples': []
    }

    tibetan_lines_processed_so_far = 0

    print(f"Found {total_files} files to process...")
    render_progress(0, total_files)

    for i, file_name in enumerate(files_to_process):
        in_path = os.path.join(input_dir, file_name)
        out_path = os.path.join(output_dir, file_name)
        
        lines_in_file, tibetan_lines_processed_so_far = process_file(in_path, out_path, cv, stats, tibetan_lines_processed_so_far)
        stats['total_lines'] += lines_in_file
        render_progress(i + 1, total_files)

    print("\n\n--- Conversion Complete ---")

    # --- Generate Report ---
    duration = datetime.now() - start_time
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=====================================================\n")
        f.write("Step 3: Tibetan Unicode to Wylie Conversion Report\n")
        f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {str(duration)}\n")
        f.write("=====================================================\n\n")
        f.write(f"Files Processed: {total_files}\n")
        f.write(f"Total Lines Scanned: {stats['total_lines']}\n\n")
        f.write(f"Lines Identified as Tibetan Unicode: {stats['lines_processed']}\n")
        f.write(f"Lines Successfully Converted to Wylie: {stats['lines_converted']}\n")
        f.write(f"Conversion Errors: {stats['conversion_errors']}\n")
        f.write(f"JSON Decoding Errors: {stats['json_errors']}\n\n")

        if stats['error_examples']:
            f.write("--- Examples of Conversion Errors ---\n")
            for ex in stats['error_examples']:
                f.write(f"File: {ex['file']}, Line: {ex['line']}\n")
                f.write(f"  Input Text: {ex.get('text', 'N/A')}\n")
                if 'error' in ex:
                    f.write(f"  Error: {ex['error']}\n")
                if 'output' in ex:
                    f.write(f"  Invalid Output: {ex['output']}\n")
                f.write("-" * 20 + "\n")

    print(f"Report generated at: {report_path}")
    print(f"Converted files are in: {output_dir}")

if __name__ == "__main__":
    main()
