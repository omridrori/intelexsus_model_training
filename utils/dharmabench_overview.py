"""
Generate an overview report for DharmaBench tasks.

For each task directory under data/downstream_tasks/DharmaBench, this script writes:
- Task name
- Absolute path
- Files contained in the task directory
- Field names found in test.jsonl (if present)
- First 3 full JSON lines from test.jsonl (as-is)

Results are saved to utils/results/dharmabench_overview_results/<timestamped>.txt

Notes:
- Progress bar is shown if tqdm is available; otherwise falls back to plain output.
- All comments are in English per project preference.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple


def try_import_tqdm():
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        # Fallback: define a no-op wrapper that mimics tqdm's interface minimally
        def passthrough(iterable: Iterable, total: Optional[int] = None, desc: str = ""):
            if desc:
                print(f"[INFO] {desc} ({total if total is not None else '?'} items)")
            return iterable

        return passthrough


def find_task_directories(dharmabench_root: Path) -> List[Path]:
    """Find task directories under DharmaBench. A task directory is any directory
    directly under a language directory (e.g., Sanskrit/RCMS) that contains at least one .jsonl file.
    """
    task_dirs: List[Path] = []
    if not dharmabench_root.exists():
        return task_dirs

    for language_dir in dharmabench_root.iterdir():
        if not language_dir.is_dir():
            continue
        for task_dir in language_dir.iterdir():
            if not task_dir.is_dir():
                continue
            has_jsonl = any(child.suffix.lower() == ".jsonl" for child in task_dir.iterdir() if child.is_file())
            if has_jsonl:
                task_dirs.append(task_dir)
    return sorted(task_dirs)


def collect_test_file_info(test_path: Path) -> Tuple[Set[str], List[str]]:
    """Collect field names and the first 3 raw JSON lines from test.jsonl.

    Returns a tuple of (field_names, sample_lines_raw).
    If file is missing or unreadable, returns (empty set, empty list).
    """
    field_names: Set[str] = set()
    sample_lines_raw: List[str] = []

    if not test_path.exists() or not test_path.is_file():
        return field_names, sample_lines_raw

    try:
        with test_path.open("r", encoding="utf-8") as f:
            for idx, raw_line in enumerate(f):
                line = raw_line.strip()
                if not line:
                    continue
                # Capture first 3 raw JSON lines as-is (without trailing newline)
                if len(sample_lines_raw) < 3:
                    sample_lines_raw.append(raw_line.rstrip("\n\r"))
                # Parse to accumulate field names
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        field_names.update(obj.keys())
                except json.JSONDecodeError:
                    # Skip malformed lines but continue processing
                    continue
    except Exception:
        # If any error occurs, return what we have (likely empty)
        return field_names, sample_lines_raw

    return field_names, sample_lines_raw


def format_section(
    task_dir: Path,
    field_names: Set[str],
    sample_lines_raw: List[str],
) -> str:
    """Format a single task section for the report."""
    lines: List[str] = []
    task_name = task_dir.name
    lines.append("=" * 80)
    lines.append(f"Task Name: {task_name}")
    lines.append(f"Path: {task_dir.resolve()}\n")

    # Files list
    try:
        files = sorted([p.name for p in task_dir.iterdir() if p.is_file()])
    except Exception:
        files = []
    lines.append("Files:")
    if files:
        for fname in files:
            lines.append(f"  - {fname}")
    else:
        lines.append("  (no files)")

    # Test file details
    lines.append("")
    test_path = task_dir / "test.jsonl"
    if test_path.exists():
        sorted_fields = sorted(field_names)
        lines.append(f"test.jsonl fields ({len(sorted_fields)}):")
        if sorted_fields:
            lines.append("  " + ", ".join(sorted_fields))
        else:
            lines.append("  (no fields detected)")

        lines.append("")
        lines.append("First 3 JSON lines from test.jsonl:")
        if sample_lines_raw:
            for raw in sample_lines_raw:
                # Print the raw JSON line as-is
                lines.append(raw)
        else:
            lines.append("  (no JSON lines found)")
    else:
        lines.append("test.jsonl: (not found)")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    script_path = Path(__file__).resolve()
    utils_dir = script_path.parent
    project_root = utils_dir.parent

    dharmabench_root = project_root / "data" / "downstream_tasks" / "DharmaBench"
    if not dharmabench_root.exists():
        print(f"[ERROR] DharmaBench directory not found at: {dharmabench_root}")
        return 1

    # Prepare results directory
    results_dir = utils_dir / "results" / "dharmabench_overview_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = results_dir / f"dharmabench_overview_{timestamp}.txt"

    task_dirs = find_task_directories(dharmabench_root)
    tqdm = try_import_tqdm()

    sections: List[str] = []
    for task_dir in tqdm(task_dirs, total=len(task_dirs), desc="Processing DharmaBench tasks"):
        test_path = task_dir / "test.jsonl"
        field_names, sample_lines_raw = collect_test_file_info(test_path)
        section_text = format_section(task_dir, field_names, sample_lines_raw)
        sections.append(section_text)

    header = [
        "DharmaBench Overview Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Root: {dharmabench_root.resolve()}",
        f"Total tasks: {len(task_dirs)}",
        "",
    ]

    try:
        report_path.write_text("\n".join(header + sections), encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to write report to {report_path}: {e}")
        return 1

    print(f"[OK] Report written to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())



