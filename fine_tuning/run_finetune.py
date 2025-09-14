from __future__ import annotations

import os
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend for Windows/servers
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import evaluate

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    raise

try:
    import pyewts  # type: ignore
except Exception:  # pragma: no cover
    pyewts = None


# --------------------------------------------------------------------------------------
# Constants & Utilities
# --------------------------------------------------------------------------------------

SCRIPT_NAME = "run_finetune"
TIBETAN_UNICODE_RE = re.compile(r"[\u0F00-\u0FFF]")


def make_results_dir(script_file: Path, run_name: str | None = None) -> Path:
    parent_dir = script_file.parent
    results_dir = parent_dir / "results" / f"{SCRIPT_NAME}_results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    leaf = f"{run_name}_{timestamp}" if run_name else timestamp
    timestamp_dir = results_dir / leaf
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    return timestamp_dir


def is_wylie_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return "wylie" in lowered or "wyle" in lowered


def cleanup_wylie_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    # Remove decorative symbols and underscores; keep shad (/)
    text = re.sub(r"[@#_]+", " ", text)
    # Remove brackets but keep inside text
    text = text.replace("[", "").replace("]", "")
    # Remove parenthesised numbers e.g. (147)
    text = re.sub(r"\(\s*\d+\s*\)", " ", text)
    # Remove standalone Latin digits
    text = re.sub(r"\b\d+\b", " ", text)
    # Remove standalone Tibetan digits (0F20–0F29)
    text = re.sub(r"(?<!\S)[\u0F20-\u0F29]+(?!\S)", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class Config:
    dataset_dir: str
    text_fields: List[str]
    label_field: str
    model_name: str
    tokenizer_name: Optional[str] = None
    tokenizers: Optional[List[str]] = None  # optional: list parallel to models
    convert_to_wylie: str = "auto"  # one of: "auto", "true", "false"
    output_dir: Optional[str] = None
    files: Optional[List[str]] = None  # default [train.jsonl, test.jsonl]
    seed: int = 42
    num_epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 8
    eval_steps: int = 50
    save_total_limit: int = 2
    label_map: Optional[Dict[str, int]] = None  # explicit mapping if provided
    metrics: Optional[List[str]] = None  # e.g., ["f1"]
    main_metric: Optional[str] = None    # e.g., "f1"
    f1_average: str = "weighted"        # for f1/precision/recall if used
    models: Optional[List[str]] = None   # optional: list of model names to compare


def load_config(cfg_path: Path) -> Config:
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Minimal validation
    required_base = ["dataset_dir", "text_fields", "label_field"]
    for key in required_base:
        if key not in raw:
            raise ValueError(f"Missing required config key: {key}")
    if not raw.get("model_name") and not raw.get("models"):
        raise ValueError("Provide either 'model_name' or a non-empty 'models' list in the config.")

    # Defaults
    files = raw.get("files") or ["train.jsonl", "test.jsonl"]

    # If model_name missing but models list exists, default to first model as model_name
    default_model_name = raw.get("model_name")
    if (not default_model_name) and raw.get("models"):
        models_list_tmp = list(raw.get("models", []))
        default_model_name = models_list_tmp[0] if models_list_tmp else None

    cfg = Config(
        dataset_dir=raw["dataset_dir"],
        text_fields=list(raw["text_fields"]),
        label_field=raw["label_field"],
        model_name=default_model_name,
        tokenizer_name=raw.get("tokenizer_name"),
        tokenizers=list(raw.get("tokenizers", [])) if raw.get("tokenizers") is not None else None,
        convert_to_wylie=str(raw.get("convert_to_wylie", "auto")).lower(),
        output_dir=raw.get("output_dir"),
        files=files,
        seed=int(raw.get("seed", 42)),
        num_epochs=int(raw.get("num_epochs", 3)),
        learning_rate=float(raw.get("learning_rate", 2e-5)),
        batch_size=int(raw.get("batch_size", 8)),
        eval_steps=int(raw.get("eval_steps", 50)),
        save_total_limit=int(raw.get("save_total_limit", 2)),
        label_map=raw.get("label_map"),
        metrics=list(raw.get("metrics", ["f1"])),
        main_metric=str(raw.get("main_metric", "f1")),
        f1_average=str(raw.get("f1_average", "weighted")),
        models=list(raw.get("models", [])) if raw.get("models") is not None else None,
    )
    return cfg


def determine_conversion_needed(cfg: Config) -> bool:
    if cfg.convert_to_wylie in {"true", "yes", "1"}:
        return True
    if cfg.convert_to_wylie in {"false", "no", "0"}:
        return False
    return is_wylie_model(cfg.model_name)


def build_results_summary_path(results_dir: Path) -> Path:
    return results_dir / "summary.json"


def maybe_convert_dataset_in_memory(cfg: Config, results_dir: Path) -> Tuple[DatasetDict, Dict[str, Any]]:
    """Load the dataset and, if needed, convert Tibetan Unicode to Wylie in-memory.

    Returns the (possibly) converted dataset and a conversion report dict.
    """
    data_files: Dict[str, str] = {}
    dataset_dir = Path(cfg.dataset_dir)
    for fname in cfg.files or []:
        name = Path(fname).stem  # train/test
        data_files[name] = str(dataset_dir / fname)

    dset = load_dataset("json", data_files=data_files)

    report: Dict[str, Any] = {
        "conversion_applied": False,
        "converted_examples": 0,
        "skipped_examples": 0,
        "errors": 0,
        "text_fields": cfg.text_fields,
    }

    need_conversion = determine_conversion_needed(cfg)
    if not need_conversion:
        return dset, report

    if pyewts is None:
        raise RuntimeError("pyewts is required for Wylie conversion. Please install it.")

    converter = pyewts.pyewts()
    report["conversion_applied"] = True

    def convert_example(example: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal report
        try:
            touched = False
            for field in cfg.text_fields:
                val = example.get(field, "")
                if isinstance(val, str) and TIBETAN_UNICODE_RE.search(val):
                    converted = cleanup_wylie_text(converter.toWylie(val))
                    example[field] = converted
                    touched = True
            if touched:
                report["converted_examples"] += 1
            else:
                report["skipped_examples"] += 1
        except Exception:
            report["errors"] += 1
        return example

    # datasets.map shows a progress bar by default; keep it simple here
    for split in list(dset.keys()):
        dset[split] = dset[split].map(convert_example)

    return dset, report


def build_label_mapping(dset: DatasetDict, cfg: Config) -> Tuple[Dict[Any, int], List[str]]:
    if cfg.label_map:
        # Provided mapping wins; ensure labels str for consistency
        label2id = {k: int(v) for k, v in cfg.label_map.items()}
        # Build id2label sorted by id
        max_id = max(label2id.values()) if label2id else -1
        id2label = [""] * (max_id + 1)
        for k, v in label2id.items():
            id2label[v] = str(k)
        return label2id, id2label

    # Infer from train split
    unique = set()
    for ex in tqdm(dset["train"], desc="Collecting labels"):
        unique.add(ex[cfg.label_field])
    # Sort deterministically (string compare)
    labels_sorted = sorted(list(unique), key=lambda x: str(x))
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    return label2id, labels_sorted


def tokenise_builder(tokenizer: AutoTokenizer, cfg: Config, label2id: Dict[Any, int]):
    def _tokenise(example: Dict[str, Any]) -> Dict[str, Any]:
        fields = cfg.text_fields
        if len(fields) == 1:
            toks = tokenizer(
                str(example.get(fields[0], "")),
                padding="max_length",
                truncation=True,
                max_length=512,
            )
        elif len(fields) == 2:
            toks = tokenizer(
                str(example.get(fields[0], "")),
                str(example.get(fields[1], "")),
                padding="max_length",
                truncation=True,
                max_length=512,
            )
        else:
            # concatenate fields if more than two
            joined = " \n ".join([str(example.get(f, "")) for f in fields])
            toks = tokenizer(
                joined,
                padding="max_length",
                truncation=True,
                max_length=512,
            )

        label_value = example.get(cfg.label_field)
        if label_value in label2id:
            toks["labels"] = int(label2id[label_value])
        else:
            # If label not in mapping, try to coerce int
            try:
                toks["labels"] = int(label_value)
            except Exception:
                raise ValueError(f"Unknown label '{label_value}' with no mapping.")
        return toks

    return _tokenise


def plot_metrics_from_history(log_history: List[Dict[str, Any]], out_png: Path, label: Optional[str] = None) -> None:
    # Plot only eval_f1
    desired_keys = ["eval_f1"]

    steps: List[int] = []
    series: Dict[str, List[float]] = {k: [] for k in desired_keys}

    for entry in log_history:
        if "step" not in entry:
            continue
        # include only entries that have at least one of the desired metrics
        if not any(k in entry for k in desired_keys):
            continue
        steps.append(entry["step"])
        for k in desired_keys:
            series[k].append(entry.get(k, np.nan))

    if not steps:
        return

    plt.figure(figsize=(9, 5))
    if any(not np.isnan(v) for v in series["eval_f1"]):
        plot_label = label if label else "eval_f1"
        plt.plot(steps, series["eval_f1"], label=plot_label, marker="s")
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics Over Steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    """Entry point: run finetuning based on a YAML config file path.

    Usage (PowerShell):
      python fine_tuning\\run_finetune.py --config fine_tuning\\configs\\tib_02.yml
    """
    import argparse

    parser = argparse.ArgumentParser(description="Config-driven fine-tuning (with optional Tibetan→Wylie conversion)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()

    # Disable Weights & Biases explicitly
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid OpenMP clash on Windows

    cfg = load_config(Path(args.config))
    # Create results dir using task name for readability
    task_name = Path(cfg.dataset_dir).name
    results_dir = make_results_dir(script_path, run_name=task_name)

    # ---------------------------- Training Runner ----------------------------
    def run_one_model(model_name: str) -> Dict[str, Any]:
        # Prepare output directory for model artifacts
        if cfg.output_dir:
            output_dir_local = Path(cfg.output_dir)
        else:
            model_stub_local = model_name.split("/")[-1]
            output_dir_local = Path("experiments") / f"{task_name}_{model_stub_local}"
        output_dir_local.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(cfg.seed)

        # Load and optionally convert dataset (once outside if heavy) — already loaded before

        model_local = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(id2label_list),
            id2label=id2label,
            label2id={str(k): v for k, v in label2id.items()},
        )
        if len(id2label_list) >= 2:
            model_local.config.problem_type = "single_label_classification"

        metric_f1_local = evaluate.load("f1")

        def _compute_metrics_local(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            f1 = metric_f1_local.compute(predictions=preds, references=labels, average=cfg.f1_average)["f1"]
            return {"f1": float(f1)}

        training_args_local = TrainingArguments(
            output_dir=str(output_dir_local),
            overwrite_output_dir=True,
            num_train_epochs=cfg.num_epochs,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=cfg.eval_steps,
            save_steps=cfg.eval_steps,
            logging_steps=cfg.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=[],
            save_total_limit=cfg.save_total_limit,
            seed=cfg.seed,
            optim="adamw_torch",
        )

        trainer_local = Trainer(
            model=model_local,
            args=training_args_local,
            train_dataset=tokenised.get("train"),
            eval_dataset=tokenised.get("test"),
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=_compute_metrics_local,
        )

        trainer_local.train()

        log_hist_path_local = Path(output_dir_local) / "log_history.json"
        with log_hist_path_local.open("w", encoding="utf-8") as f_:
            json.dump(trainer_local.state.log_history, f_, indent=2)

        final_metrics_local = trainer_local.evaluate()
        return {
            "model": model_name,
            "output_dir": str(output_dir_local),
            "log_history": trainer_local.state.log_history,
            "final_metrics": final_metrics_local,
        }

    # Load and optionally convert dataset once
    dset, conversion_report = maybe_convert_dataset_in_memory(cfg, results_dir)
    # Build tokenizer once (based on first model) and labels once
    first_model_name = (cfg.models[0] if cfg.models else cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name or first_model_name)
    tokenise_fn = tokenise_builder(tokenizer, cfg, {})  # placeholder mapping, will rebuild after label map
    label2id, id2label_list = build_label_mapping(dset, cfg)
    id2label = {i: lab for i, lab in enumerate(id2label_list)}
    # Recreate tokeniser function with proper label mapping
    tokenise_fn = tokenise_builder(tokenizer, cfg, label2id)
    tokenised = DatasetDict()
    for split in dset.keys():
        tokenised[split] = dset[split].map(tokenise_fn, remove_columns=dset[split].column_names)

    # Run either single model or multiple models
    model_names = cfg.models if cfg.models else [cfg.model_name]
    all_runs: List[Dict[str, Any]] = []
    for i, m in enumerate(model_names):
        print(f"=== Fine-tuning model: {m} ===")
        result = run_one_model(m)
        all_runs.append(result)

    # Save combined summary and comparison plots
    # Per-model plots
    for r in all_runs:
        model_stub = r["model"].split("/")[-1]
        per_plot = results_dir / f"{model_stub}_metrics_over_steps.png"
        plot_metrics_from_history(r["log_history"], per_plot, label=model_stub)

    # Combined plot: F1 over steps for all models
    combined_plot = results_dir / "comparison_f1_over_steps.png"
    plt.figure(figsize=(9, 5))
    for r in all_runs:
        model_stub = r["model"].split("/")[-1]
        # Create a temporary figure per series onto the current axes
        # Reuse plot_metrics_from_history logic by extracting steps/series here
        steps = []
        f1_series = []
        for entry in r["log_history"]:
            if "step" in entry and "eval_f1" in entry:
                steps.append(entry["step"])
                f1_series.append(entry["eval_f1"])
        if steps:
            plt.plot(steps, f1_series, label=model_stub, marker="o")
    plt.xlabel("Step")
    plt.ylabel("F1")
    plt.title("Model Comparison: F1 over Steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(combined_plot, dpi=160)
    plt.close()

    # Final F1 bar chart
    bar_plot = results_dir / "comparison_final_f1.png"
    labels_list = []
    values = []
    for r in all_runs:
        labels_list.append(r["model"].split("/")[-1])
        values.append(float(r["final_metrics"].get("eval_f1", 0.0)))
    if labels_list:
        plt.figure(figsize=(8, 4))
        plt.bar(labels_list, values)
        plt.ylabel("Final F1")
        plt.title("Model Comparison: Final F1")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(bar_plot, dpi=160)
        plt.close()

    # Write combined JSON summary
    with (results_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"runs": all_runs, "conversion_report": conversion_report}, f, indent=2)

    print(f"\nDone. Results saved under: {results_dir}")
    print(f"Comparison plots: {combined_plot}, {bar_plot}")


if __name__ == "__main__":
    main()


