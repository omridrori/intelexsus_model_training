"""CLI wrapper around sanscrit.model.downstream.fine_tune_root_commentary"""
from __future__ import annotations

import argparse
from pathlib import Path

from sanscrit.model.downstream import fine_tune_root_commentary


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Fine-tune a sequence-classification model on the Sanskrit *root vs. commentary* task"
    )
    parser.add_argument("--model", required=True, help="HF identifier or local path to the model checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Path/identifier of the matching tokenizer")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Directory to store checkpoints / log_history.json / final model",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs (default: 3)")
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--batch", type=int, default=8, help="Batch size per device (default: 8)"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=50, help="Evaluation/save/log frequency (default: 50 steps)"
    )
    args = parser.parse_args()

    metrics = fine_tune_root_commentary(
        model_name_or_path=args.model,
        tokenizer_name_or_path=args.tokenizer,
        output_dir=Path(args.out),
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch,
        eval_steps=args.eval_steps,
    )

    print("\nFinal evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":  
    main()
