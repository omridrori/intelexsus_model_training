import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Tuple

try:
    from tokenizers import Tokenizer  # optional, for verification
except Exception:
    Tokenizer = None  # type: ignore

def default_results_dir(script_file: Path) -> Path:
    # Ensure results directory exists under the script's parent directory
    parent = script_file.parent
    results_root = parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    out_dir = results_root / "inspect_bert_chunks_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def reservoir_sample_indices(n_items: int, k: int, rng: random.Random) -> List[int]:
    # Reservoir sampling for k unique indices from n_items
    k = max(0, min(k, n_items))
    if k == 0:
        return []
    res = list(range(k))
    for i in range(k, n_items):
        j = rng.randrange(0, i + 1)
        if j < k:
            res[j] = i
    return sorted(res)

def main() -> None:
    ap = argparse.ArgumentParser(description="Sample BERT-ready chunks and optionally verify with tokenizer")
    ap.add_argument("--jsonl", default=os.path.join("sandbox", "tibetian", "all_unicode", "results", "prepare_bert_corpus_unicode_results", "bert_ready_512.jsonl"))
    ap.add_argument("--txt", default=os.path.join("sandbox", "tibetian", "all_unicode", "results", "prepare_bert_corpus_unicode_results", "bert_ready_512.txt"))
    ap.add_argument("--samples", type=int, default=30, help="Number of random samples to preview")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verify_tokenizer", default="", help="Optional path to tokenizer.json to verify sampled num_tokens and print tokens")
    ap.add_argument("--max_tokens_print", type=int, default=200, help="Max tokens to print per sample line")
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    out_dir = default_results_dir(script_path)
    report_path = out_dir / "report.txt"
    samples_path = out_dir / "samples.txt"
    verify_path = out_dir / "verification.txt"

    rng = random.Random(args.seed)

    # 1) Count total lines in JSONL (to know how many samples to draw)
    total = 0
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for _ in f:
            total += 1

    # 2) Sample indices
    sample_indices = reservoir_sample_indices(total, args.samples, rng)
    previews: List[Tuple[int, str]] = []
    if sample_indices:
        wanted = set(sample_indices)
        with open(args.txt, "r", encoding="utf-8") as tf:
            for i, line in enumerate(tf):
                if i in wanted:
                    previews.append((i, line.rstrip("\n")))
                if len(previews) == len(sample_indices):
                    break

    # 3) Prepare tokenizer if available and requested
    tokenizer = None
    tokenizer_path = args.verify_tokenizer if args.verify_tokenizer else None
    if tokenizer_path and Tokenizer is not None:
        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
        except Exception:
            tokenizer = None

    # 4) Write sampled lines to file, including their tokens if tokenizer is available
    with open(samples_path, "w", encoding="utf-8") as sf:
        for idx, text in previews:
            sf.write(f"--- Sample line #{idx} ---\n")
            sf.write(text + "\n")
            if tokenizer is not None:
                try:
                    encoding = tokenizer.encode(text)
                    tokens = encoding.tokens
                    n_all = len(tokens)
                    n_print = max(0, min(args.max_tokens_print, n_all))
                    shown = tokens[:n_print]
                    tail = n_all - n_print
                    line = "Tokens (n=" + str(n_all) + "): " + " ".join(shown)
                    if tail > 0:
                        line += f" ... [+{tail} more]"
                    sf.write(line + "\n")
                except Exception as e:
                    sf.write(f"Tokenizer error: {e}\n")
            sf.write("\n")

    # 5) Optional verification with tokenizer (write encoded token count for each sample)
    if tokenizer_path and tokenizer is not None:
        mismatches: List[str] = []
        try:
            for idx, text in previews:
                n_enc = len(tokenizer.encode(text).ids)
                mismatches.append(f"line={idx} encoded_tokens={n_enc}")
        except Exception as e:
            mismatches.append(f"Tokenizer verification failed: {e}")
        with open(verify_path, "w", encoding="utf-8") as vf:
            vf.write("\n".join(mismatches) + "\n")

    # 6) Write summary report
    with open(report_path, "w", encoding="utf-8") as rep:
        rep.write("================ Inspect BERT Chunks (Sample Only) ================\n")
        rep.write(f"Input JSONL: {args.jsonl}\n")
        rep.write(f"Input TXT:   {args.txt}\n")
        rep.write(f"Total Chunks (lines): {total}\n")
        rep.write(f"Sampled {len(previews)} lines.\n")
        rep.write(f"Samples file:  {samples_path}\n")
        if tokenizer_path:
            rep.write(f"Verification (encoded counts) at: {verify_path}\n")

    print(f"Wrote report: {report_path}")
    print(f"Wrote samples: {samples_path}")
    if tokenizer_path:
        print(f"Wrote verification: {verify_path}")

if __name__ == "__main__":
    main()






