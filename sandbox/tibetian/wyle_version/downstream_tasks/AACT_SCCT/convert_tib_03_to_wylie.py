import os
import json
from pathlib import Path
import pyewts
import re
from datetime import datetime

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
# Source dataset directory (tib_03_task inside central downstream tasks folder)
SRC_DIR = Path(__file__).resolve().parents[4] / 'data' / 'downstream_tasks' / 'Tibetan' / 'tib_03_task'
# Target directory will be next to source, suffixed with _wylie
DST_DIR = SRC_DIR.parent / f"{SRC_DIR.name}_wylie"
RESULTS_DIR = HERE / 'results' / 'convert_tib_03_to_wylie_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
TIBETAN_UNICODE_RE = re.compile(r"[\u0F00-\u0FFF]")

def cleanup_wylie(text: str) -> str:
    """Basic clean-up: remove decorative marks, brackets, page numbers, and collapse spaces."""
    if not text or not isinstance(text, str):
        return ""
    # Remove decorative symbols and underscores; keep shad (/)
    text = re.sub(r"[@#_]+", " ", text)
    # Remove brackets but keep inside text
    text = text.replace("[", "").replace("]", "")
    # Remove parenthesised numbers e.g. (147)
    text = re.sub(r"\(\s*\d+\s*\)", " ", text)
    # Remove standalone Latin digits
    text = re.sub(r"\b\d+\b", " ", text)
    # Remove standalone Tibetan digits (0F20–0F29)
    text = re.sub(r"(?<!\S)[\u0F20-\u0F29]+(?!\S)", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def convert_file(src_path: Path, dst_path: Path, cv: pyewts.pyewts, stats: dict):
    """Convert one JSONL file from Tibetan Unicode to Wylie."""
    os.makedirs(dst_path.parent, exist_ok=True)
    with src_path.open('r', encoding='utf-8') as fin, dst_path.open('w', encoding='utf-8') as fout:
        for line_no, line in enumerate(fin, start=1):
            try:
                obj = json.loads(line)
                text = obj.get('text', '')
                if text and TIBETAN_UNICODE_RE.search(text):
                    converted = cleanup_wylie(cv.toWylie(text))
                    obj['text'] = converted
                    stats['converted'] += 1
                else:
                    # keep as-is if no Tibetan detected
                    stats['skipped'] += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except Exception as e:
                stats['errors'] += 1
                if len(stats['error_examples']) < 5:
                    stats['error_examples'].append({'file': str(src_path), 'line': line_no, 'error': str(e)})
                fout.write(line)  # write original line to keep alignment


def main():
    print("--- Converting tib_03_task to Wylie ---")
    print(f"Source dir: {SRC_DIR}")
    print(f"Destination dir: {DST_DIR}\n")

    if not SRC_DIR.exists():
        print(f"Source directory not found: {SRC_DIR}")
        return

    files = ['train.jsonl', 'test.jsonl']
    cv = pyewts.pyewts()

    overall = {
        'converted': 0,
        'skipped': 0,
        'errors': 0,
        'error_examples': []
    }

    start = datetime.now()
    for fname in files:
        src_path = SRC_DIR / fname
        if not src_path.exists():
            print(f"Warning: {src_path} not found, skipping")
            continue
        dst_path = DST_DIR / fname
        stats = {'converted': 0, 'skipped': 0, 'errors': 0, 'error_examples': []}
        convert_file(src_path, dst_path, cv, stats)
        print(f"Converted {fname}: {stats['converted']} lines (skipped {stats['skipped']}, errors {stats['errors']})")
        overall['converted'] += stats['converted']
        overall['skipped'] += stats['skipped']
        overall['errors'] += stats['errors']
        overall['error_examples'].extend(stats['error_examples'])

    duration = datetime.now() - start
    # Write simple report
    report_path = RESULTS_DIR / 'tib_03_conversion_report.txt'
    with report_path.open('w', encoding='utf-8') as rep:
        rep.write("Tibetan Unicode to Wylie conversion report for tib_03_task\n")
        rep.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        rep.write(f"Duration: {duration}\n")
        rep.write(f"Lines converted: {overall['converted']}\n")
        rep.write(f"Lines skipped (no Tibetan): {overall['skipped']}\n")
        rep.write(f"Errors: {overall['errors']}\n")
        if overall['error_examples']:
            rep.write("\nError examples:\n")
            for ex in overall['error_examples']:
                rep.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("\nConversion complete.")
    print(f"Converted files saved to: {DST_DIR}")
    print(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
