"""CLI wrapper for continual pre-training of mBERT on Sanskrit."""
import os
# Ensure duplicate OpenMP runtime doesn't crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

from sanscrit.model.pretrain_continual import continual_pretrain_mbert


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual pre-train mBERT on Sanskrit data")
    parser.add_argument("--data", type=str, help="Path to training text file (default from config)")
    parser.add_argument("--out", type=str, help="Directory to store checkpoints and final model (default from config)")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--base", type=str, default="bert-base-multilingual-cased", help="Base pretrained model name or path")
    args = parser.parse_args()

    continual_pretrain_mbert(
        data_file=Path(args.data) if args.data else None,
        model_output_dir=Path(args.out) if args.out else None,
        num_epochs=args.epochs,
        base_model=args.base,
    )


if __name__ == "__main__":
    main()

