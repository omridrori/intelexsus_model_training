"""CLI to prepare Tibetan Wylie corpus for BERT continual pre-training.

Writes outputs under results/prepare_corpus_results by default.
"""
import argparse
from pathlib import Path

from ..config import PROCESSED_DIR
from ..preprocessing.prepare_corpus import prepare_corpus


def default_results_dir(script_file: Path) -> Path:
    parent = script_file.parent
    results_root = parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    out_dir = results_root / "prepare_corpus_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Tibetan BERT corpus")
    parser.add_argument("--input_dir", default="cleaned_data/step3_wylie_converted")
    parser.add_argument("--tokenizer", default="tibetan-tokenizer/tokenizer.json")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--max_len", type=int, default=512)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    results_dir = default_results_dir(script_path)
    output_path = Path(args.output_file) if args.output_file else results_dir / "tibetan_bert_ready_512.txt"

    output = prepare_corpus(
        input_dir=Path(args.input_dir),
        tokenizer_json=Path(args.tokenizer),
        output_file=output_path,
        max_len=args.max_len,
    )
    print(f"Corpus prepared at: {output}")


if __name__ == "__main__":
    main()





