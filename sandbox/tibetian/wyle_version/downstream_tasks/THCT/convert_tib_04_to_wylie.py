import json
import re
from pathlib import Path
import pyewts

# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / 'data' / 'downstream_tasks' / 'Tibetan' / 'tib_04_task'
DST_DIR = SRC_DIR.parent / f'{SRC_DIR.name}_wylie'
RESULTS_DIR = HERE / 'results' / 'convert_tib_04_to_wylie_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIB_RE = re.compile(r'[\u0F00-\u0FFF]')

def cleanup(text: str) -> str:
    if not text:
        return ''
    text = re.sub(r'[@#_]+', ' ', text)
    text = text.replace('[', '').replace(']', '')
    text = re.sub(r'\(\s*\d+\s*\)', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'(?<!\S)[\u0F20-\u0F29]+(?!\S)', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def convert_file(src: Path, dst: Path, cv: pyewts.pyewts, stats: dict):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open('r', encoding='utf-8') as fin, dst.open('w', encoding='utf-8') as fout:
        for line in fin:
            try:
                obj = json.loads(line)
                txt = obj.get('text', '')
                if TIB_RE.search(txt):
                    obj['text'] = cleanup(cv.toWylie(txt))
                    stats['converted'] += 1
                else:
                    stats['skipped'] += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
            except Exception:
                stats['errors'] +=1
                fout.write(line)


def main():
    if not SRC_DIR.exists():
        print(f'Source dir not found: {SRC_DIR}')
        return

    files = [f for f in ['train.jsonl', 'test.jsonl'] if (SRC_DIR/f).exists()]
    cv = pyewts.pyewts()
    overall = {'converted':0,'skipped':0,'errors':0}

    for f in files:
        s = {'converted':0,'skipped':0,'errors':0}
        convert_file(SRC_DIR/f, DST_DIR/f, cv, s)
        print(f'{f}: converted {s["converted"]}, skipped {s["skipped"]}, errors {s["errors"]}')
        for k in overall: overall[k]+=s[k]

    with (RESULTS_DIR/'report.txt').open('w',encoding='utf-8') as rep:
        rep.write(f'Source: {SRC_DIR}\nDest: {DST_DIR}\n')
        rep.write(json.dumps(overall, indent=2))
    print('Done. Report saved.')

if __name__ == '__main__':
    main()
