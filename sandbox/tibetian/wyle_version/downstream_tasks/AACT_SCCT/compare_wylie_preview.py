import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]

# Paths
REFERENCE_FILE = ROOT / 'cleaned_data' / 'step3_wylie_converted' / 'Tibetan_1.jsonl'
ORIG_TRAIN = ROOT / 'data' / 'downstream_tasks' / 'Tibetan' / 'tib_03_task' / 'train.jsonl'
CONV_TRAIN = ROOT / 'data' / 'downstream_tasks' / 'Tibetan' / 'tib_03_task_wylie' / 'train.jsonl'

MAX_REF_LINES = 5
MAX_COMPARE_LINES = 5

print("=== Sample lines from reference Wylie corpus ===")
if REFERENCE_FILE.exists():
    with REFERENCE_FILE.open(encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= MAX_REF_LINES:
                break
            obj = json.loads(line)
            print(f"[{i+1}] {obj.get('text', '')[:200]}\n")
else:
    print(f"Reference file not found: {REFERENCE_FILE}\n")

print("\n=== Before/After examples from tib_03_task train ===")
if ORIG_TRAIN.exists() and CONV_TRAIN.exists():
    with ORIG_TRAIN.open(encoding='utf-8') as forig, CONV_TRAIN.open(encoding='utf-8') as fconv:
        for i, (l_orig, l_conv) in enumerate(zip(forig, fconv)):
            if i >= MAX_COMPARE_LINES:
                break
            try:
                o_obj = json.loads(l_orig)
                c_obj = json.loads(l_conv)
            except Exception:
                continue
            o_text = o_obj.get('text', '')
            c_text = c_obj.get('text', '')
            print(f"Example {i+1}:")
            print("Original :", o_text[:200])
            print("Converted:", c_text[:200])
            print("-")
else:
    if not ORIG_TRAIN.exists():
        print(f"Original train file missing: {ORIG_TRAIN}")
    if not CONV_TRAIN.exists():
        print(f"Converted train file missing: {CONV_TRAIN}")
