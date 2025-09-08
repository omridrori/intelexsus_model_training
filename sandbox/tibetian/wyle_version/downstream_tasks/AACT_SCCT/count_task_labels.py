import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / 'data' / 'downstream_tasks' / 'Tibetan' / 'tib_03_task_wylie'
# Fallback to non-wylie if _wylie doesn't exist
if not DATA_DIR.exists():
    DATA_DIR = ROOT / 'data' / 'downstream_tasks' / 'Tibetan' / 'tib_03_task'

files = [f for f in ['train.jsonl', 'test.jsonl'] if (DATA_DIR / f).exists()]

c1 = Counter()
c3 = Counter()

total = 0
for fname in files:
    path = DATA_DIR / fname
    with path.open(encoding='utf-8') as f:
        for line in f:
            total += 1
            obj = json.loads(line)
            c1.update([obj.get('task_1_label', 'NULL')])
            c3.update([obj.get('task_3_label', 'NULL')])

print(f"Dataset directory: {DATA_DIR}\nFiles read: {', '.join(files)} (total lines {total})\n")
print('Task 1 label distribution:')
for lbl, cnt in c1.most_common():
    print(f"  {lbl}: {cnt}")
print('\nTask 3 label distribution:')
for lbl, cnt in c3.most_common():
    print(f"  {lbl}: {cnt}")
