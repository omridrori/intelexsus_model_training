"""CLI wrapper for BERT pre-training."""
import os
# Ensure duplicate OpenMP runtime doesn\'t crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

from sanscrit.model.pretrain import pretrain_bert


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train a BERT model from scratch on Sanskrit data")
    parser.add_argument("--data", type=str, help="Path to training text file (default from config)")
    parser.add_argument(
        "--out", type=str, help="Directory to store checkpoints and final model (default from config)"
    )
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    args = parser.parse_args()

    pretrain_bert(
        data_file=Path(args.data) if args.data else None,
        model_output_dir=Path(args.out) if args.out else None,
        num_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()