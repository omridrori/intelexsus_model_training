import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# --- Wire to cloned detect_and_convert (use pyewts + ACIP only to avoid circular imports) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECT_CONVERT_PATH = os.path.join(
    SCRIPT_DIR, "..", "wyle_version", "preprocessing", "detect_and_convert"
)
DETECT_CONVERT_PATH = os.path.normpath(DETECT_CONVERT_PATH)

if DETECT_CONVERT_PATH not in sys.path:
    sys.path.insert(0, DETECT_CONVERT_PATH)

try:
    import pyewts  # direct use for EWTS<->Unicode
    from importlib.machinery import SourceFileLoader
    _ACIP_MODULE = SourceFileLoader(
        "_acip_module",
        os.path.join(DETECT_CONVERT_PATH, "conversion", "ACIP.py"),
    ).load_module()
    ACIPtoEWTS = getattr(_ACIP_MODULE, "ACIPtoEWTS")
except Exception as e:
    print(f"Failed setting up converters from: {DETECT_CONVERT_PATH}")
    print(f"Error: {e}")
    sys.exit(1)


# --- Regexes ---
TIBETAN_UNICODE_RE = re.compile(r"[\u0F00-\u0FFF]")
ASCII_LETTER_RE = re.compile(r"^[A-Za-z]+$")
HAS_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
LATIN_NUMBER_RE = re.compile(r"^\d+$")
TIBETAN_DIGIT_TOKEN_RE = re.compile(r"^[\u0F20-\u0F29]+$")
URL_RE = re.compile(
    r"(?i)\b((?:https?://|www\.)[\w\-._~:/?#\[\]@!$&'()*+,;=%]+)"
)
PAGE_MARKER_INLINE_RE = re.compile(r"@(?:[0-9]+[A-Za-z]?|[A-Za-z]+)")


def contains_tibetan(text: str) -> bool:
    if not text:
        return False
    return bool(TIBETAN_UNICODE_RE.search(text))


def tibetan_ratio(text: str) -> float:
    if not text:
        return 0.0
    tib = len(TIBETAN_UNICODE_RE.findall(text))
    return tib / max(1, len(text))


def remove_links_with_context(text: str, window: int = 20) -> str:
    """Remove URLs and ±window words around each URL occurrence."""
    if not text:
        return text
    tokens: List[str] = text.split()
    n = len(tokens)
    if n == 0:
        return text

    to_drop = [False] * n
    for i, tok in enumerate(tokens):
        if URL_RE.search(tok):
            start = max(0, i - window)
            end = min(n, i + window + 1)
            for j in range(start, end):
                to_drop[j] = True

    kept = [t for i, t in enumerate(tokens) if not to_drop[i]]
    return " ".join(kept)


def remove_page_markers(tokens: List[str]) -> List[str]:
    """(Kept for backward compat) token-level drop; now superseded by inline removal."""
    return [t for t in tokens if not t.startswith("@")]


def remove_page_markers_in_text(text: str) -> str:
    if not text:
        return text
    return PAGE_MARKER_INLINE_RE.sub(" ", text)


def remove_non_tibetan_metadata_segments(text: str) -> str:
    """Drop bracketed segments ([], (), <>) that do not contain Tibetan."""
    if not text:
        return text

    def _drop_if_non_tib(m: re.Match) -> str:
        inner = m.group(1)
        return "" if not contains_tibetan(inner) else m.group(0)

    # [] and () and <>
    s = re.sub(r"\[(.*?)\]", _drop_if_non_tib, text)
    s = re.sub(r"\((.*?)\)", _drop_if_non_tib, s)
    s = re.sub(r"<(.*?)>", _drop_if_non_tib, s)
    return s


def strip_english_and_numbers(text: str) -> str:
    """Remove tokens that are ASCII letters or numbers (Latin/Tibetan)."""
    if not text:
        return text
    kept: List[str] = []
    for tok in text.split():
        if LATIN_NUMBER_RE.match(tok):
            continue
        if TIBETAN_DIGIT_TOKEN_RE.match(tok):
            continue
        # Remove tokens with ASCII letters, unless they also contain Tibetan
        if HAS_ASCII_LETTER_RE.search(tok) and not contains_tibetan(tok):
            continue
        kept.append(tok)
    return " ".join(kept)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def render_progress(current: int, total: int, width: int = 40, label: str | None = None) -> None:
    if total <= 0:
        return
    current = min(max(current, 0), total)
    fraction = current / total
    filled = int(fraction * width)
    bar = "#" * filled + "-" * (width - filled)
    prefix = f"{label} " if label else ""
    print(f"\r{prefix}[{bar}] {current}/{total} ({fraction:.0%})", end="", flush=True)


def _ewts_to_unicode(text: str) -> Tuple[str, List[str]]:
    """Try strict EWTS→Unicode, fallback to sloppy if warnings, return best.
    """
    conv = pyewts.pyewts(check=True, check_strict=True, print_warnings=False, fix_spacing=True)
    warns_strict: List[str] = []
    out_strict = conv.toUnicode(text, warns_strict, sloppy=False)
    if contains_tibetan(out_strict) and not warns_strict:
        return out_strict, warns_strict
    # Fallback: sloppy mode often covers Wylie-like inputs
    warns_sloppy: List[str] = []
    out_sloppy = conv.toUnicode(text, warns_sloppy, sloppy=True)
    # Prefer output with Tibetan and fewer warnings
    if contains_tibetan(out_sloppy) and (len(warns_sloppy) <= len(warns_strict) or not contains_tibetan(out_strict)):
        return out_sloppy, warns_sloppy
    return out_strict, warns_strict


def convert_to_unicode_text(text: str) -> str:
    """Best-effort line-level conversion to preserve proper tsheg placement.
    Strategy:
    - If already has Tibetan, keep as-is
    - Try EWTS strict; if Tibetan, use it
    - Try EWTS sloppy; if Tibetan, use it
    - Try ACIP→EWTS strict → Unicode strict; if Tibetan, use it
    - Otherwise, return original (foreign terms remain)
    """
    if not text:
        return ""
    if contains_tibetan(text):
        return text
    conv = pyewts.pyewts(check=True, check_strict=True, print_warnings=False, fix_spacing=True)
    # EWTS strict
    warns: List[str] = []
    out = conv.toUnicode(text, warns, sloppy=False)
    if contains_tibetan(out):
        return out
    # EWTS sloppy
    warns2: List[str] = []
    out2 = conv.toUnicode(text, warns2, sloppy=True)
    if contains_tibetan(out2) and tibetan_ratio(out2) >= 0.2:
        return out2
    # ACIP pipeline
    try:
        ewts_text = ACIPtoEWTS(text)
        warns3: List[str] = []
        out3 = conv.toUnicode(ewts_text, warns3, sloppy=False)
        if contains_tibetan(out3):
            return out3
    except Exception:
        pass
    return text


def remove_all_digits(text: str) -> str:
    """Remove all ASCII and Tibetan digits anywhere in the text."""
    if not text:
        return text
    return re.sub(r"[0-9\u0F20-\u0F29]+", "", text)


# Whitelist filter: keep Tibetan block + minimal punctuation/whitespace
_WHITELIST_EXTRA = {"\u300A", "\u300B", "\n", " "}


def whitelist_tibetan(text: str) -> str:
    if not text:
        return text
    kept_chars: List[str] = []
    for ch in text:
        code = ord(ch)
        if 0x0F00 <= code <= 0x0FFF:
            kept_chars.append(ch)
            continue
        if ch in ("\u300A", "\u300B"):
            kept_chars.append(ch)
            continue
        if ch == "\n" or ch == " ":
            kept_chars.append(ch)
    return "".join(kept_chars)


def remove_standalone_slash_runs(text: str) -> str:
    """Globally remove runs of '/' and '*' (e.g., *//, //, /**) before conversion."""
    if not text:
        return text
    return re.sub(r"[*/]{2,}", " ", text)


def process_line(text: str) -> str:
    # 1) Remove URLs ±20 words
    s = remove_links_with_context(text, window=20)
    # 2) Drop non-Tibetan bracketed metadata
    s = remove_non_tibetan_metadata_segments(s)

    # 3) Remove page markers and numbers
    s = remove_page_markers_in_text(s)
    s = remove_all_digits(s)

    # 3b) Remove slash/star delimiter runs like *// before conversion
    s = remove_standalone_slash_runs(s)

    # 4) Convert the whole line to preserve tsheg; attempt foreign terms too
    s = convert_to_unicode_text(s)

    # 4a) Fix tsheg after shad/double-shad (remove ་ immediately after །/༎)
    s = re.sub(r"([།༎])་+", r"\1", s)
    # 4a2) Ensure a space after shad/double-shad when followed by Tibetan letter
    s = re.sub(r"([།༎])(?=[\u0F00-\u0FFF])", r"\1 ", s)

    # 4b) Keep only lines with sufficient Tibetan content (drop English/other-script lines)
    lines = [ln for ln in s.splitlines() if tibetan_ratio(ln) >= 0.2]
    s = "\n".join(lines)

    # 4c) Remove remaining non‑Tibetan tokens (keep only tokens containing U+0F00–U+0FFF)
    if s:
        s = " ".join(tok for tok in s.split() if contains_tibetan(tok))

    # 4d) Apply strict whitelist to purge any leftover foreign glyphs
    s = whitelist_tibetan(s)

    # 5) Normalize whitespace
    s = normalize_spaces(s)
    return s


def _first_tibetan_token_index(tokens: List[str]) -> int:
    for idx, tok in enumerate(tokens):
        if contains_tibetan(tok):
            return idx
    return -1


def _snippet_around_first_tibetan(text: str, total_words: int = 300) -> str:
    tokens = text.split()
    if not tokens:
        return ""
    center = _first_tibetan_token_index(tokens)
    if center == -1:
        return " ".join(tokens[:total_words])
    half = max(1, total_words // 2)
    start = max(0, center - half)
    end = min(len(tokens), start + total_words)
    start = max(0, end - total_words)
    return " ".join(tokens[start:end])


def _first_n_words(text: str, n: int = 300) -> str:
    return " ".join(text.split()[:n])


def process_file(in_path: Path, out_path: Path, max_examples: int = 0) -> Dict[str, object]:
    stats = {
        "lines": 0,
        "json_errors": 0,
        "changed": 0,
        "kept_unicode": 0,
        "converted_from_wylie": 0,
        "link_segments_removed": 0,  # heuristic: count url matches
        "examples": [],  # list of {line, before_snippet, after_snippet}
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Count lines first for accurate progress bar
    try:
        total_lines = sum(1 for _ in in_path.open("r", encoding="utf-8"))
    except Exception:
        total_lines = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            stats["lines"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                fout.write(line)
                continue

            original = obj.get("text", "")

            # quick count of links for stats
            stats["link_segments_removed"] += len(URL_RE.findall(original or ""))

            processed = process_line(original)

            # high-level tracking: crude before/after unicode presence
            before_has_unicode = contains_tibetan(original or "")
            after_has_unicode = contains_tibetan(processed or "")
            if before_has_unicode and after_has_unicode and processed == original:
                stats["kept_unicode"] += 1
            if (not before_has_unicode) and after_has_unicode:
                stats["converted_from_wylie"] += 1
                if max_examples and len(stats["examples"]) < max_examples:
                    stats["examples"].append({
                        "line": stats["lines"],
                        "before_snippet": _first_n_words(original or "", 300),
                        "after_snippet": _snippet_around_first_tibetan(processed or "", 300),
                    })

            if processed != original:
                stats["changed"] += 1

            obj["text"] = processed
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if total_lines:
                render_progress(stats["lines"], total_lines, label=f"{in_path.name}")

    return stats


def default_results_dir(script_file: Path) -> Path:
    parent = script_file.parent
    results_root = parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    out_dir = results_root / "make_all_unicode_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Tibetan Wylie/EWTS/ACIP -> Unicode; keep Unicode; drop EN/URLs/numbers.")
    parser.add_argument("--input_dir", default=os.path.join("cleaned_data", "step3_wylie_converted"))
    parser.add_argument("--output_dir", default=None, help="Defaults to results/make_all_unicode_results under this script's directory")
    parser.add_argument("--debug_examples", type=int, default=0, help="Collect up to N examples of conversions with 300-word snippets")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    default_out = default_results_dir(script_path)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_out

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    files = [p for p in sorted(input_dir.iterdir()) if p.suffix == ".jsonl"]
    if not files:
        print(f"No .jsonl files found in: {input_dir}")
        sys.exit(0)

    print(f"Processing {len(files)} files from {input_dir} -> {output_dir}")
    total = {
        "files": 0,
        "lines": 0,
        "json_errors": 0,
        "changed": 0,
        "kept_unicode": 0,
        "converted_from_wylie": 0,
        "link_segments_removed": 0,
        "examples": [],
    }

    total_files = len(files)
    render_progress(0, total_files, label="Files")
    for idx, src in enumerate(files, start=1):
        dst = output_dir / src.name
        st = process_file(src, dst, max_examples=args.debug_examples)
        total["files"] += 1
        for k in ("lines", "json_errors", "changed", "kept_unicode", "converted_from_wylie", "link_segments_removed"):
            total[k] += st.get(k, 0)
        if args.debug_examples and st.get("examples"):
            for ex in st["examples"]:
                if len(total["examples"]) < args.debug_examples:
                    total["examples"].append({
                        "file": src.name,
                        **ex,
                    })
        print()  # newline after per-file progress
        print(f"  - {src.name}: lines={st['lines']} changed={st['changed']} json_errors={st['json_errors']}")
        render_progress(idx, total_files, label="Files")

    # Write a brief run report next to outputs
    report_path = output_dir / "run_report.txt"
    with report_path.open("w", encoding="utf-8") as rep:
        rep.write("================ All-Unicode Conversion Report ================\n")
        rep.write(f"Input dir: {input_dir}\n")
        rep.write(f"Output dir: {output_dir}\n")
        rep.write(f"Files: {total['files']}\n")
        rep.write(f"Lines: {total['lines']}\n")
        rep.write(f"Changed lines: {total['changed']}\n")
        rep.write(f"JSON errors: {total['json_errors']}\n")
        rep.write(f"Kept Tibetan Unicode lines: {total['kept_unicode']}\n")
        rep.write(f"Converted from Wylie/EWTS/ACIP: {total['converted_from_wylie']}\n")
        rep.write(f"Link segments removed (occurrences): {total['link_segments_removed']}\n")

    if args.debug_examples and total["examples"]:
        dbg_path = output_dir / "debug_converted_snippets.txt"
        with dbg_path.open("w", encoding="utf-8") as dbg:
            for i, ex in enumerate(total["examples"], start=1):
                dbg.write(f"--- Example {i} | file={ex['file']} line={ex['line']} ---\n")
                dbg.write("BEFORE (first 300 words)\n")
                dbg.write(ex["before_snippet"] + "\n\n")
                dbg.write("AFTER (300 words around first Tibetan)\n")
                dbg.write(ex["after_snippet"] + "\n\n")
        print(f"Debug examples written to: {dbg_path}")

    print()  # newline after files progress
    print(f"Done. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()


