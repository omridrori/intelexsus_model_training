import argparse
import os
import re
import sys
from pathlib import Path


def default_results_dir(script_file: Path) -> Path:
    parent = script_file.parent
    results_root = parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    out_dir = results_root / "examples_preview_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def extract_examples(text: str) -> list[str]:
    """Extract example payloads from a free-form examples file.
    Strategies:
      1) Quoted: text= "..." or text= “...”, multiline until matching quote
      2) Block fallback: text= (newline) ... until next 'text=' key or EOF
    """
    examples: list[str] = []
    s = text.replace("\r\n", "\n")

    # Find all key positions
    keys = list(re.finditer(r"(?mi)^\s*(?:text|Text|txt)\s*=", s))
    if not keys:
        return []

    for idx, m in enumerate(keys):
        seg_start = m.end()
        seg_end = keys[idx + 1].start() if idx + 1 < len(keys) else len(s)

        k = seg_start
        # skip spaces
        while k < seg_end and s[k] in " \t":
            k += 1

        payload = None
        if k < seg_end and s[k] in ('"', '\u201C', '\u201D'):
            q = s[k]
            close_q = '"' if q == '"' else '\u201D'
            k += 1
            j = s.find(close_q, k, seg_end)
            if j != -1:
                payload = s[k:j]
            else:
                payload = s[k:seg_end]
        else:
            payload = s[k:seg_end]

        ex = (payload or "").strip()
        if ex:
            examples.append(ex)

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing on examples_to_test and write before/after report")
    parser.add_argument("--examples_file", default=str(Path(__file__).with_name("examples_to_test.txt")))
    parser.add_argument("--max_chars", type=int, default=1500)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # Wire import for process_line
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    try:
        from make_all_unicode import process_line
    except Exception as e:
        print(f"Failed importing process_line: {e}")
        sys.exit(1)

    src_path = Path(args.examples_file)
    if not src_path.exists():
        print(f"Examples file not found: {src_path}")
        sys.exit(1)

    content = src_path.read_text(encoding="utf-8", errors="ignore")
    examples = extract_examples(content)
    if not examples:
        print("No examples found (looking for text=/Text=/txt= with quotes).")
        sys.exit(0)

    results_dir = default_results_dir(Path(__file__).resolve())
    out_file = results_dir / "before_after.txt"

    max_chars = max(200, int(args.max_chars))

    with out_file.open("w", encoding="utf-8") as out:
        for idx, ex in enumerate(examples, start=1):
            before = ex[:max_chars]
            after = process_line(ex)[:max_chars]
            out.write(f"--- Example {idx} ---\n")
            out.write("BEFORE:\n")
            out.write(before + "\n\n")
            out.write("AFTER:\n")
            out.write(after + "\n\n")

    print(f"Wrote before/after for {len(examples)} examples to: {out_file}")


if __name__ == "__main__":
    main()



import re
import sys
from pathlib import Path


def default_results_dir(script_file: Path) -> Path:
    parent = script_file.parent
    results_root = parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    out_dir = results_root / "examples_preview_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def extract_examples(text: str) -> list[str]:
    """Extract example payloads from a free-form examples file.
    Strategies:
      1) Quoted: text= "..." or text= “...”, multiline until matching quote
      2) Block fallback: text= (newline) ... until next 'text=' key or EOF
    """
    examples: list[str] = []
    s = text.replace("\r\n", "\n")

    # Find all key positions
    keys = list(re.finditer(r"(?mi)^\s*(?:text|Text|txt)\s*=", s))
    if not keys:
        return []

    for idx, m in enumerate(keys):
        seg_start = m.end()
        seg_end = keys[idx + 1].start() if idx + 1 < len(keys) else len(s)

        k = seg_start
        # skip spaces
        while k < seg_end and s[k] in " \t":
            k += 1

        payload = None
        if k < seg_end and s[k] in ('"', '\u201C', '\u201D'):
            q = s[k]
            close_q = '"' if q == '"' else '\u201D'
            k += 1
            j = s.find(close_q, k, seg_end)
            if j != -1:
                payload = s[k:j]
            else:
                payload = s[k:seg_end]
        else:
            payload = s[k:seg_end]

        ex = (payload or "").strip()
        if ex:
            examples.append(ex)

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing on examples_to_test and write before/after report")
    parser.add_argument("--examples_file", default=str(Path(__file__).with_name("examples_to_test.txt")))
    parser.add_argument("--max_chars", type=int, default=1500)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # Wire import for process_line
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    try:
        from make_all_unicode import process_line
    except Exception as e:
        print(f"Failed importing process_line: {e}")
        sys.exit(1)

    src_path = Path(args.examples_file)
    if not src_path.exists():
        print(f"Examples file not found: {src_path}")
        sys.exit(1)

    content = src_path.read_text(encoding="utf-8", errors="ignore")
    examples = extract_examples(content)
    if not examples:
        print("No examples found (looking for text=/Text=/txt= with quotes).")
        sys.exit(0)

    results_dir = default_results_dir(Path(__file__).resolve())
    out_file = results_dir / "before_after.txt"

    max_chars = max(200, int(args.max_chars))

    with out_file.open("w", encoding="utf-8") as out:
        for idx, ex in enumerate(examples, start=1):
            before = ex[:max_chars]
            after = process_line(ex)[:max_chars]
            out.write(f"--- Example {idx} ---\n")
            out.write("BEFORE:\n")
            out.write(before + "\n\n")
            out.write("AFTER:\n")
            out.write(after + "\n\n")

    print(f"Wrote before/after for {len(examples)} examples to: {out_file}")


if __name__ == "__main__":
    main()



import re
import sys
from pathlib import Path


def default_results_dir(script_file: Path) -> Path:
    parent = script_file.parent
    results_root = parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    out_dir = results_root / "examples_preview_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def extract_examples(text: str) -> list[str]:
    """Extract example payloads from a free-form examples file.
    Strategies:
      1) Quoted: text= "..." or text= “...”, multiline until matching quote
      2) Block fallback: text= (newline) ... until next 'text=' key or EOF
    """
    examples: list[str] = []
    s = text.replace("\r\n", "\n")

    # Find all key positions
    keys = list(re.finditer(r"(?mi)^\s*(?:text|Text|txt)\s*=", s))
    if not keys:
        return []

    for idx, m in enumerate(keys):
        seg_start = m.end()
        seg_end = keys[idx + 1].start() if idx + 1 < len(keys) else len(s)

        k = seg_start
        # skip spaces
        while k < seg_end and s[k] in " \t":
            k += 1

        payload = None
        if k < seg_end and s[k] in ('"', '\u201C', '\u201D'):
            q = s[k]
            close_q = '"' if q == '"' else '\u201D'
            k += 1
            j = s.find(close_q, k, seg_end)
            if j != -1:
                payload = s[k:j]
            else:
                payload = s[k:seg_end]
        else:
            payload = s[k:seg_end]

        ex = (payload or "").strip()
        if ex:
            examples.append(ex)

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing on examples_to_test and write before/after report")
    parser.add_argument("--examples_file", default=str(Path(__file__).with_name("examples_to_test.txt")))
    parser.add_argument("--max_chars", type=int, default=1500)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # Wire import for process_line
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    try:
        from make_all_unicode import process_line
    except Exception as e:
        print(f"Failed importing process_line: {e}")
        sys.exit(1)

    src_path = Path(args.examples_file)
    if not src_path.exists():
        print(f"Examples file not found: {src_path}")
        sys.exit(1)

    content = src_path.read_text(encoding="utf-8", errors="ignore")
    examples = extract_examples(content)
    if not examples:
        print("No examples found (looking for text=/Text=/txt= with quotes).")
        sys.exit(0)

    results_dir = default_results_dir(Path(__file__).resolve())
    out_file = results_dir / "before_after.txt"

    max_chars = max(200, int(args.max_chars))

    with out_file.open("w", encoding="utf-8") as out:
        for idx, ex in enumerate(examples, start=1):
            before = ex[:max_chars]
            after = process_line(ex)[:max_chars]
            out.write(f"--- Example {idx} ---\n")
            out.write("BEFORE:\n")
            out.write(before + "\n\n")
            out.write("AFTER:\n")
            out.write(after + "\n\n")

    print(f"Wrote before/after for {len(examples)} examples to: {out_file}")


if __name__ == "__main__":
    main()


