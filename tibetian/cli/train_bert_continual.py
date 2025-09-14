"""CLI wrapper for continual pre-training of mBERT on Tibetan."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

from ..model.pretrain_continual import continual_pretrain_mbert
from ..config import BERT_READY_FILE, MODEL_OUTPUT_DIR


def default_results_dir(script_file: Path) -> Path:
    parent = script_file.parent
    results_root = parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    out_dir = results_root / "train_bert_continual_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual pre-train mBERT on Tibetan data")
    parser.add_argument("--data", type=str, help="Path to training text file (default from config)")
    parser.add_argument("--out", type=str, help="Directory to store checkpoints and final model (default results dir)")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--base", type=str, default="bert-base-multilingual-cased", help="Base pretrained model name or path")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    results_dir = default_results_dir(script_path)
    default_out = results_dir

    continual_pretrain_mbert(
        data_file=Path(args.data) if args.data else BERT_READY_FILE,
        model_output_dir=Path(args.out) if args.out else default_out,
        num_epochs=args.epochs,
        base_model=args.base,
    )


if __name__ == "__main__":
    main()





