import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi
from tqdm import tqdm


def generate_model_card(repo_id: str, is_private: bool) -> str:
    """Return a Markdown model card for the Tibetan Unicode mBERT continual model."""
    username, model_name = (repo_id.split("/", 1) + [""])[:2]
    privacy_note = (
        "This repository is private." if is_private else "This repository is public."
    )

    return f"""
# {model_name}

{privacy_note}

## Overview
**This is a BERT model continually trained from `bert-base-multilingual-cased` on Tibetan data.**

It was trained as part of the Intelexsus project on a mixed Tibetan corpus that includes:
- Tibetan text written in the original Tibetan script (Unicode)
- Data originally in Wylie transliteration that was converted into Tibetan script

The aim is to improve Tibetan representations for downstream tasks while preserving compatibility with multilingual BERT.

## Model Details
- **Base model**: `bert-base-multilingual-cased`
- **Language**: Tibetan (bo)
- **Training objective**: Masked Language Modeling (MLM)
- **Architecture**: 12-layer, 768-hidden, 12-heads
- **Tokenizer**: WordPiece tokenizer compatible with mBERT (includes Tibetan Unicode support)

## How to Use
You can use this model directly with the `transformers` library for the fill-mask task.

```python
from transformers import pipeline

model_name = "{repo_id}"
unmasker = pipeline("fill-mask", model=model_name)

# Example sentence in Tibetan (demonstrative only)
result = unmasker("བོད་ཡིག་ [MASK] ཡིན་པ་རེད།")
print(result)
```

You can also load the model and tokenizer directly for more control:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
# You can now use the model for your own fine-tuning and inference tasks.
```

## Training Data
The continual training used a Tibetan corpus consisting of:
- Native Tibetan text in Unicode (U+0F00–U+0FFF block)
- Wylie transliterated data converted into Tibetan script prior to training

This combination aims to cover both native-script Tibetan and content originally prepared in transliteration that has been normalized to Unicode Tibetan.

## Intended Use and Limitations
This model is intended for research and downstream tasks involving Tibetan. It may contain biases present in the training data and may not perform well outside the Tibetan domain.

## Citation
If you use this model, please cite the Intelexsus project or link to the model page: `https://huggingface.co/{repo_id}`
"""


def copy_tree_with_progress(source_dir: Path, destination_dir: Path) -> int:
    """Copy all files from source_dir to destination_dir with a progress bar.

    Returns the number of files copied.
    """
    source_dir = source_dir.resolve()
    destination_dir = destination_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    all_files = [p for p in source_dir.rglob("*") if p.is_file()]
    destination_dir.mkdir(parents=True, exist_ok=True)

    for path in source_dir.rglob("*"):
        if path.is_dir():
            (destination_dir / path.relative_to(source_dir)).mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(all_files, desc="Copying files", unit="file"):
        rel = file_path.relative_to(source_dir)
        dest_file = destination_dir / rel
        shutil.copy2(file_path, dest_file)

    return len(all_files)


def validate_checkpoint_dir(checkpoint_dir: Path) -> None:
    """Validate that the checkpoint directory looks like a Transformers BERT checkpoint."""
    required_files = [
        "config.json",
        "model.safetensors",
    ]
    missing = [name for name in required_files if not (checkpoint_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint directory {checkpoint_dir} is missing required files: {missing}"
        )


def upload_to_hub(
    repo_id: str,
    source_dir: Path,
    is_private: bool,
    commit_message: str,
    hf_token: str | None,
) -> dict:
    """Upload the folder and a generated README.md to Hugging Face Hub.

    Returns a summary dict with repo_url, files_count, and timestamps.
    """
    api = HfApi(token=hf_token)

    # Ensure repository exists
    api.create_repo(repo_id=repo_id, repo_type="model", private=is_private, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tibetan_unicode_mbert_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Copy model files with progress bar
        files_copied = copy_tree_with_progress(source_dir, tmpdir_path)

        # Write README.md
        readme_text = generate_model_card(repo_id=repo_id, is_private=is_private)
        (tmpdir_path / "README.md").write_text(readme_text, encoding="utf-8")

        # Upload the whole folder (displays its own progress bars during upload)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(tmpdir_path),
            repo_type="model",
            path_in_repo=".",
            commit_message=commit_message,
        )

    return {
        "repo_id": repo_id,
        "repo_url": f"https://huggingface.co/{repo_id}",
        "files_copied": files_copied,
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "private": is_private,
    }


def write_results(results_dir: Path, summary: dict, readme_text: str) -> None:
    """Persist summary JSON and a copy of README.md into the results directory."""
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "upload_summary.json"
    readme_copy_path = results_dir / "README_uploaded.md"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    readme_copy_path.write_text(readme_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a Tibetan Unicode continual mBERT checkpoint (with tokenizer) to Hugging Face Hub. "
            "Includes a generated model card and saves logs under results/"
        )
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target repo id on the Hub, e.g. 'OMRIDRORI/mbert-tibetan-continual-unicode-240k'",
    )
    parser.add_argument(
        "--source-dir",
        default=str(Path("mbert-tibetan-continual-unicode") / "checkpoint-240000"),
        help="Path to the local checkpoint directory to upload",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update the repository as private (default: public)",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload Tibetan Unicode continual mBERT checkpoint and model card",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", None),
        help="Hugging Face token (or set HF_TOKEN env var). If omitted, uses local login.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    # Per workspace rule: put results under parent/results/<script_name>_results
    results_dir = script_dir / "results" / "upload_tibetan_unicode_mbert_results"

    source_dir = Path(args.source_dir).resolve()
    validate_checkpoint_dir(source_dir)

    # Prepare README text early so we can persist it to results regardless of upload outcome
    readme_text = generate_model_card(repo_id=args.repo_id, is_private=args.private)

    try:
        summary = upload_to_hub(
            repo_id=args.repo_id,
            source_dir=source_dir,
            is_private=bool(args.private),
            commit_message=args.commit_message,
            hf_token=args.hf_token,
        )
        print(f"Upload complete: {summary['repo_url']}")
    except Exception as exc:
        summary = {
            "repo_id": args.repo_id,
            "repo_url": f"https://huggingface.co/{args.repo_id}",
            "error": str(exc),
            "attempted_at": datetime.utcnow().isoformat() + "Z",
        }
        print(f"ERROR: {exc}", file=sys.stderr)
        # Still write results with the error for debugging
    finally:
        write_results(results_dir=results_dir, summary=summary, readme_text=readme_text)
        print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()



