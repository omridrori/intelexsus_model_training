"""CLI for packing chunks into ≤512 token sequences."""
import argparse
from pathlib import Path

from sanscrit.preprocessing.pack_tokens import pack_chunks_to_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack cleaned chunks into <=512 token sequences")
    parser.add_argument("--chunks", type=str, help="Input chunks file (default from config)")
    parser.add_argument("--out", type=str, help="Output sequences file (default from config)")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer path (default from config)")
    args = parser.parse_args()

    pack_chunks_to_sequences(
        chunks_input=Path(args.chunks) if args.chunks else None,
        output_file=Path(args.out) if args.out else None,
        tokenizer_path=Path(args.tokenizer) if args.tokenizer else None,
    )


if __name__ == "__main__":
    main()