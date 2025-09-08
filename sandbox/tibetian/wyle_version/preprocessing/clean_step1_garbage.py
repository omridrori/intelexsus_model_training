import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, Tuple, List


TIBETAN_BLOCK_RE = re.compile(r"[\u0F00-\u0FFF]")
CONTROL_CHARS_RE = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]")
INVISIBLES_RE = re.compile(r"[\u200B\u200C\u200D\u2060\uFEFF]")
NBSP_LIKE_RE = re.compile(r"[\u00A0\u202F\u2007]")


def compute_tibetan_ratio(text: str) -> float:
    if not text:
        return 0.0
    tibetan_chars = len(TIBETAN_BLOCK_RE.findall(text))
    return tibetan_chars / max(1, len(text))


def looks_like_utf16_mojibake(text: str) -> bool:
    # Heuristics: presence of many NULs, starts with 'ÿþ' (UTF-16 BOM shown as Latin-1), or high control char density
    if not text:
        return False
    nul_count = text.count("\x00")
    control_density = len(CONTROL_CHARS_RE.findall(text)) / max(1, len(text))
    return text.startswith("ÿþ") or nul_count > 0 or control_density > 0.02


def try_fix_utf16_mojibake(text: str) -> Tuple[str, bool, float, float]:
    """Attempt to recover text that looks like UTF-16 mis-decoded as Latin-1/UTF-8.
    Strategy: encode to bytes via latin1, then decode as utf-16le/utf-16. Accept if Tibetan ratio improves.
    Returns: (best_text, changed, before_ratio, after_ratio)
    """
    before_ratio = compute_tibetan_ratio(text)
    try:
        raw = text.encode("latin1", errors="strict")
    except Exception:
        return text, False, before_ratio, before_ratio
    candidates: List[str] = []
    for codec in ("utf-16le", "utf-16"):
        try:
            candidates.append(raw.decode(codec, errors="ignore"))
        except Exception:
            continue
    best_text = text
    best_ratio = before_ratio
    for cand in candidates:
        ratio = compute_tibetan_ratio(cand)
        if ratio > best_ratio + 0.02:  # require meaningful improvement
            best_text = cand
            best_ratio = ratio
    
    changed = best_text is not text
    final_ratio = best_ratio if changed else before_ratio

    return (best_text, changed, before_ratio, final_ratio)


def clean_text(text: str) -> Tuple[str, Dict[str, int], Dict[str, str]]:
    """Clean garbage/control characters while preserving Tibetan content.
    Returns: (cleaned_text, counters, debug_samples)
    counters keys: control_removed, invisibles_removed, nbsp_normalized, utf16_fixed
    debug_samples may include 'utf16_before'/'utf16_after' for first fix.
    """
    counters = {"control_removed": 0, "invisibles_removed": 0, "nbsp_normalized": 0, "utf16_fixed": 0}
    debug: Dict[str, str] = {}

    s = text

    # Normalize BOM/invisible marks
    invisibles = INVISIBLES_RE.findall(s)
    if invisibles:
        counters["invisibles_removed"] += len(invisibles)
        s = INVISIBLEs_RE_SUB(s)

    # Normalize NBSP-like spaces to regular spaces
    nbsp_count = len(NBSP_LIKE_RE.findall(s))
    if nbsp_count:
        counters["nbsp_normalized"] += nbsp_count
        s = NBSP_LIKE_RE.sub(" ", s)

    # Attempt UTF-16 mojibake recovery if heuristic triggers
    if looks_like_utf16_mojibake(s):
        original_for_debug = s
        fixed, changed, ratio_before, ratio_after = try_fix_utf16_mojibake(s)
        if changed:
            debug.setdefault("utf16_before", original_for_debug[:2000])
            s = fixed
            debug["utf16_after"] = s[:2000]
            debug["ratio_before"] = f"{ratio_before:.2%}"
            debug["ratio_after"] = f"{ratio_after:.2%}"
            counters["utf16_fixed"] += 1

    # Remove control characters (keep tabs/newlines)
    controls = CONTROL_CHARS_RE.findall(s)
    if controls:
        counters["control_removed"] += len(controls)
        s = CONTROL_CHARS_RE.sub("", s)

    return s, counters, debug


# Precompile a safe substitute to remove invisibles quickly
def _remove_invisibles(m: re.Match) -> str:
    return ""


INVISIBLEs_RE_SUB = lambda s: INVISIBLES_RE.sub(_remove_invisibles, s)


def process_file(in_path: str, out_path: str, max_examples: int = 3) -> Dict[str, object]:
    """Process one JSONL file and write cleaned output. Returns per-file stats."""
    stats = {
        "file": in_path,
        "lines": 0,
        "lines_modified": 0,
        "control_removed": 0,
        "invisibles_removed": 0,
        "nbsp_normalized": 0,
        "utf16_fixed_lines": 0,
        "examples": [],  # up to max_examples dicts: {i, before, after}
        "errors": 0,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            stats["lines"] += 1
            line_out = line
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                cleaned, counters, debug = clean_text(text)
                # Aggregate counters
                stats["control_removed"] += counters["control_removed"]
                stats["invisibles_removed"] += counters["invisibles_removed"]
                stats["nbsp_normalized"] += counters["nbsp_normalized"]
                if counters["utf16_fixed"]:
                    stats["utf16_fixed_lines"] += 1
                if cleaned != text:
                    stats["lines_modified"] += 1
                    if len(stats["examples"]) < max_examples:
                        stats["examples"].append({
                            "line": i,
                            "before": text[:3000],
                            "after": cleaned[:3000],
                            **({
                                "utf16_before": debug.get("utf16_before", ""), 
                                "utf16_after": debug.get("utf16_after", ""),
                                "ratio_before": debug.get("ratio_before"),
                                "ratio_after": debug.get("ratio_after")
                            } if debug else {}),
                        })
                obj["text"] = cleaned
                line_out = json.dumps(obj, ensure_ascii=False) + "\n"
            except Exception:
                # Keep original line if anything goes wrong, but count error
                stats["errors"] += 1
            fout.write(line_out)

    return stats


def render_progress(current: int, total: int, width: int = 40) -> None:
    if total <= 0:
        return
    current = min(max(current, 0), total)
    fraction = current / total
    filled = int(fraction * width)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rProgress: [{bar}] {current}/{total} ({fraction:.0%})", end="", flush=True)


def write_report(report_path: str, run_stats: Dict[str, object], per_file: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as rep:
        rep.write("""================================================================================\n""")
        rep.write("Step 1: Garbage Character Cleaning Report\n")
        rep.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        rep.write("""================================================================================\n""")
        rep.write(f"Files Processed: {run_stats['files']}\n")
        rep.write(f"Total Lines: {run_stats['lines']}\n")
        rep.write(f"Lines Modified: {run_stats['lines_modified']} ({run_stats['lines_modified'] / max(1, run_stats['lines']):.2%})\n")
        rep.write(f"UTF-16 Fixes (lines): {run_stats['utf16_fixed_lines']}\n")
        rep.write(f"Control Chars Removed: {run_stats['control_removed']}\n")
        rep.write(f"Invisible Marks Removed: {run_stats['invisibles_removed']}\n")
        rep.write(f"NBSP Normalized: {run_stats['nbsp_normalized']}\n")
        rep.write(f"Errors: {run_stats['errors']}\n")
        rep.write("\n")

        for st in per_file:
            rep.write("--------------------------------------------------------------------------------\n")
            rep.write(f"File: {st['file']}\n")
            rep.write(f"Lines: {st['lines']} | Modified: {st['lines_modified']} ({st['lines_modified'] / max(1, st['lines']):.2%}) | Errors: {st['errors']}\n")
            rep.write(f"UTF-16 Fixes: {st['utf16_fixed_lines']} | Control Removed: {st['control_removed']} | Invisibles Removed: {st['invisibles_removed']} | NBSP Normalized: {st['nbsp_normalized']}\n")
            if st["examples"]:
                rep.write("Examples (truncated):\n")
                for ex in st["examples"]:
                    rep.write(f"  - Line {ex['line']}:\n")
                    if ex.get("utf16_before") and ex.get("utf16_after"):
                        rep.write("    utf16_before: " + ex["utf16_before"][:200] + "\n")
                        rep.write("    utf16_after:  " + ex["utf16_after"][:200] + "\n")
                        if ex.get("ratio_before") is not None:
                            rep.write(f"    (Validation: Tibetan Ratio changed from {ex['ratio_before']} to {ex['ratio_after']})\n")
                    rep.write("    before: " + ex["before"][:200] + "\n")
                    rep.write("    after:  " + ex["after"][:200] + "\n")
            rep.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean garbage/binary characters from Tibetan JSONL files.")
    parser.add_argument("--input_dir", default=os.path.join("data", "tibetan"), help="Directory containing source .jsonl files")
    parser.add_argument("--output_dir", default=os.path.join("cleaned_data", "step1_no_garbage"), help="Directory to write cleaned .jsonl files")
    parser.add_argument("--report", default=os.path.join("cleaned_data", "reports", "step1_garbage_filter_report.txt"), help="Path for cleaning report")
    parser.add_argument("--no_progress", action="store_true", help="Disable console progress bar")
    parser.add_argument("--max_examples", type=int, default=3, help="Max changed-line examples per file in report")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: input_dir not found: {args.input_dir}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    files = [f for f in sorted(os.listdir(args.input_dir)) if f.endswith(".jsonl")]
    if not files:
        print(f"No .jsonl files found in {args.input_dir}")
        return 0

    per_file_stats: List[Dict[str, object]] = []
    run_stats = {
        "files": 0,
        "lines": 0,
        "lines_modified": 0,
        "utf16_fixed_lines": 0,
        "control_removed": 0,
        "invisibles_removed": 0,
        "nbsp_normalized": 0,
        "errors": 0,
    }

    if not args.no_progress:
        render_progress(0, len(files))

    for idx, name in enumerate(files, start=1):
        in_path = os.path.join(args.input_dir, name)
        out_path = os.path.join(args.output_dir, name)
        st = process_file(in_path, out_path, max_examples=args.max_examples)
        per_file_stats.append(st)
        run_stats["files"] += 1
        run_stats["lines"] += st["lines"]
        run_stats["lines_modified"] += st["lines_modified"]
        run_stats["utf16_fixed_lines"] += st["utf16_fixed_lines"]
        run_stats["control_removed"] += st["control_removed"]
        run_stats["invisibles_removed"] += st["invisibles_removed"]
        run_stats["nbsp_normalized"] += st["nbsp_normalized"]
        run_stats["errors"] += st["errors"]
        if not args.no_progress:
            render_progress(idx, len(files))

    if not args.no_progress:
        print()  # newline after progress bar

    write_report(args.report, run_stats, per_file_stats)
    print(f"Cleaning complete. Files: {run_stats['files']}, Lines: {run_stats['lines']}, Modified: {run_stats['lines_modified']}")
    print(f"Report written to: {args.report}")
    print(f"Cleaned files in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


