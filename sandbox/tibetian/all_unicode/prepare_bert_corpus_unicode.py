import os
import re
import json
import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from tokenizers import Tokenizer


SENT_PUNC = {"\u0F0D", "\u0F0E", "\u0F14"}  # །, ༎, ༔


def render_progress(current: int, total: int, width: int = 40, label: str | None = None) -> None:
    if total <= 0:
        return
    current = min(max(current, 0), total)
    fraction = current / total
    filled = int(fraction * width)
    bar = "#" * filled + "-" * (width - filled)
    prefix = f"{label} " if label else ""
    print(f"\r{prefix}[{bar}] {current}/{total} ({fraction:.0%})", end="", flush=True)


def default_results_dir(script_file: Path) -> Path:
    parent = script_file.parent
    results_root = parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    out_dir = results_root / "prepare_bert_corpus_unicode_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def iter_texts(input_dir: str) -> Iterable[str]:
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".jsonl"):
            continue
        fp = os.path.join(input_dir, fname)
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if text:
                    yield text


def split_sentences_keep_punc(text: str) -> List[str]:
    if not text:
        return []
    sentences: List[str] = []
    buf: List[str] = []
    for ch in text:
        buf.append(ch)
        if ch in ("\u0F0D", "\u0F0E", "\u0F14"):
            sentence = "".join(buf).strip()
            if sentence:
                sentences.append(sentence)
            buf = []
    # trailing fragment without terminal punctuation
    tail = "".join(buf).strip()
    if tail:
        sentences.append(tail)
    return sentences


def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text).ids)


def slice_sentence_by_tokens(tokenizer: Tokenizer, ids: List[int], max_tokens: int) -> List[str]:
    parts: List[str] = []
    for i in range(0, len(ids), max_tokens):
        sub_ids = ids[i : i + max_tokens]
        parts.append(tokenizer.decode(sub_ids))
    return [p for p in parts if p and p.strip()]


def chunk_sentences_batched(
    tokenizer: Tokenizer,
    sentences: List[str],
    max_tokens: int,
    batch_size: int = 4096,
    show_progress: bool = False,
) -> List[Tuple[str, int]]:
    chunks: List[Tuple[str, int]] = []
    cur_text_parts: List[str] = []
    cur_tokens = 0
    total = len(sentences)

    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch = [s.strip() for s in sentences[start:end]]
        encodings = tokenizer.encode_batch(batch)
        for local_idx, (sent, enc) in enumerate(zip(batch, encodings), start=1):
            if not sent:
                continue
            ids = enc.ids
            sent_tok = len(ids)
            if sent_tok > max_tokens:
                if cur_text_parts:
                    chunks.append((" ".join(cur_text_parts).strip(), cur_tokens))
                    cur_text_parts, cur_tokens = [], 0
                for piece in slice_sentence_by_tokens(tokenizer, ids, max_tokens):
                    piece_tok = len(tokenizer.encode(piece).ids)
                    chunks.append((piece, piece_tok))
                continue

            if cur_tokens + sent_tok <= max_tokens:
                cur_text_parts.append(sent)
                cur_tokens += sent_tok
            else:
                chunks.append((" ".join(cur_text_parts).strip(), cur_tokens))
                cur_text_parts, cur_tokens = [sent], sent_tok

        if show_progress:
            render_progress(end, total, label="Chunking")

    if cur_text_parts:
        chunks.append((" ".join(cur_text_parts).strip(), cur_tokens))

    return [(t, n) for (t, n) in chunks if t]


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare BERT-ready 512-token chunks for Tibetan Unicode corpus")
    ap.add_argument(
        "--input_dir",
        default=os.path.join("sandbox", "tibetian", "all_unicode", "results", "make_all_unicode_results"),
    )
    ap.add_argument(
        "--tokenizer",
        default=os.path.join("sandbox", "tibetian", "all_unicode", "results", "train_unicode_tokenizer_results", "tokenizer.json"),
    )
    ap.add_argument("--max_tokens", type=int, default=512)
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    out_dir = default_results_dir(script_path)
    out_txt = out_dir / "bert_ready_512.txt"
    out_jsonl = out_dir / "bert_ready_512.jsonl"
    report = out_dir / "report.txt"

    tokenizer = Tokenizer.from_file(args.tokenizer)

    total_sentences = 0
    total_chunks = 0
    total_tokens = 0

    # Gather sentences across all files with a files progress bar
    sentences: List[str] = []
    files = [fn for fn in sorted(os.listdir(args.input_dir)) if fn.endswith('.jsonl')]
    total_files = len(files)
    render_progress(0, total_files, label="Files")
    for idx, fname in enumerate(files, start=1):
        fp = os.path.join(args.input_dir, fname)
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if text:
                    sentences.extend(split_sentences_keep_punc(text))
        render_progress(idx, total_files, label="Files")
    print()

    total_sentences = len(sentences)

    chunks = chunk_sentences_batched(tokenizer, sentences, args.max_tokens, batch_size=4096, show_progress=True)

    with out_txt.open("w", encoding="utf-8") as ft, out_jsonl.open("w", encoding="utf-8") as fj:
        for text, n_tok in chunks:
            ft.write(text + "\n")
            fj.write(json.dumps({"text": text, "num_tokens": n_tok}, ensure_ascii=False) + "\n")
            total_tokens += n_tok
            total_chunks += 1

    # finish progress line
    print()

    with report.open("w", encoding="utf-8") as rep:
        rep.write("================ BERT Corpus Prep (Unicode) ================\n")
        rep.write(f"Input dir: {args.input_dir}\n")
        rep.write(f"Tokenizer: {args.tokenizer}\n")
        rep.write(f"Max tokens per chunk: {args.max_tokens}\n")
        rep.write(f"Sentences: {total_sentences}\n")
        rep.write(f"Chunks: {total_chunks}\n")
        if total_chunks:
            rep.write(f"Avg tokens per chunk: {total_tokens/total_chunks:.2f}\n")

    print(f"Done. Wrote: {out_txt} and {out_jsonl}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()


