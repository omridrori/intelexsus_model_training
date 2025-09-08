"""Count label distribution for task_1_label and task_3_label (or any fields) in a Tibetan downstream dataset.
Usage:
python count_labels.py --task tib_04_task --fields task_1_label task_3_label
"""
import json
from collections import Counter
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[4]


def count_labels(base_dir: Path, files, fields):
    counters = {f: Counter() for f in fields}
    total = 0
    for fname in files:
        path = base_dir / fname
        if not path.exists():
            continue
        with path.open(encoding='utf-8') as f:
            for line in f:
                total += 1
                obj = json.loads(line)
                for fld in fields:
                    counters[fld].update([obj.get(fld, 'NULL')])
    return counters, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help='subfolder name e.g. tib_04_task or tib_04_task_wylie')
    parser.add_argument('--fields', nargs='+', default=['task_1_label','task_3_label'], help='label fields to count')
    args = parser.parse_args()

    base_dir = ROOT / 'data' / 'downstream_tasks' / 'Tibetan' / args.task
    if not base_dir.exists():
        print(f'Task directory not found: {base_dir}')
        return

    counters, total = count_labels(base_dir, ['train.jsonl','test.jsonl'], args.fields)
    print(f'Task dir: {base_dir} (total lines {total})')
    for fld in args.fields:
        print(f'\nField {fld}:')
        for lbl, cnt in counters[fld].most_common():
            print(f'  {lbl}: {cnt}')

if __name__ == '__main__':
    main()
