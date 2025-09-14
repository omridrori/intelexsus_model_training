import os
import re
import json
import argparse
from pathlib import Path

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Split, Sequence


def default_results_dir(script_file: Path) -> Path:
    parent = script_file.parent
    results_root = parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    out_dir = results_root / "train_unicode_tokenizer_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_text_iterator(data_dir: str):
    for filename in os.listdir(data_dir):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        text = json.loads(line).get("text")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        continue


def count_lines(path: str) -> int:
    total = 0
    for filename in os.listdir(path):
        if filename.endswith(".jsonl"):
            fp = os.path.join(path, filename)
            with open(fp, "r", encoding="utf-8") as f:
                for _ in f:
                    total += 1
    return total


def build_tsheg_aware_pretokenizer() -> Sequence:
    """
    Split by (whitespace OR tsheg) and isolate shad/double‑shad.
    This treats tsheg as a separator like space, without touching intra‑syllabic letters.
    """
    split_ws_or_tsheg = Regex(r"\s+|[\u0F0B]+")
    shad_pattern = Regex(r"([\u0F0D\u0F0E\u0F11\u0F14])")  # ། ༎ ༑ ༔
    return Sequence([
        Split(split_ws_or_tsheg, behavior="removed"),
        Split(shad_pattern, behavior="isolated"),
    ])


def train_unicode_tokenizer(input_dir: str, output_dir: str, vocab_size: int, min_frequency: int) -> None:
    print(f"Reading Unicode data from: {input_dir}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = build_tsheg_aware_pretokenizer()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True,
    )

    text_iter = get_text_iterator(input_dir)
    total = count_lines(input_dir)

    print(f"Starting BPE training (vocab={vocab_size}, min_freq={min_frequency})...")
    tokenizer.train_from_iterator(text_iter, trainer=trainer, length=total)
    print("Training complete.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenizer.json (all-in-one) and raw vocab/merges for compatibility
    tok_path = out_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))
    print(f"Saved tokenizer to: {tok_path}")

    model_dir = out_dir / "bpe_files"
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.model.save(str(model_dir))
    print(f"Saved vocab/merges to: {model_dir}")

    # Quick smoke test
    example = "བོད་སྐད་ཡིག་གཉིས་པ་ ། གཞན་དག་ཀྱང་ བློ་སྣང་"
    enc = tokenizer.encode(example)
    print("Example:", example)
    print("Tokens:", enc.tokens)
    print("IDs:", enc.ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on Tibetan Unicode text (.jsonl with 'text' field)")
    parser.add_argument("--input_dir", default=os.path.join("sandbox", "tibetian", "all_unicode", "results", "make_all_unicode_results"))
    parser.add_argument("--output_dir", default=None, help="Defaults to results/train_unicode_tokenizer_results next to this script")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--min_frequency", type=int, default=2)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    default_out = default_results_dir(script_path)
    input_dir = args.input_dir
    output_dir = args.output_dir or str(default_out)

    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    print(f"Outputs will be saved under: {output_dir}")
    train_unicode_tokenizer(input_dir, output_dir, args.vocab_size, args.min_frequency)


if __name__ == "__main__":
    main()


