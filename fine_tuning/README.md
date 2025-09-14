## Fine-tuning utility (config-driven)

This tool fine-tunes BERT-like models from Hugging Face on your downstream tasks with minimal setup. It supports Tibetan datasets stored as JSONL and can automatically convert Tibetan Unicode text to Wylie transliteration when using a Wylie-trained model.

### Features
- Config-driven YAML with clear fields
- Auto model/tokenizer download from Hugging Face
- Optional Tibetan→Wylie conversion (auto/true/false)
- Multi-field inputs (e.g., `root` + `commentary`)
- Final metrics saved as JSON and a metrics-over-steps PNG
- All results under `fine_tuning/results/run_finetune_results/<timestamp>/`

### Install (Windows PowerShell)
Run from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you want a minimal install for just this tool:
```powershell
pip install transformers datasets evaluate pyyaml matplotlib tqdm pyewts
```

### Configuration
Create a single YAML file with the following keys:

```yaml
dataset_dir: data/downstream_tasks/Tibetan/tib_02_task
text_fields: [text]          # one or more fields concatenated in order
label_field: label           # target field name in JSONL
models:
  - OMRIDRORI/mbert-tibetan-continual-unicode-240k
convert_to_wylie: auto       # auto | true | false
num_epochs: 3
learning_rate: 2.0e-5
batch_size: 8
eval_steps: 50
save_total_limit: 2
files: [train.jsonl, test.jsonl]
f1_average: weighted         # F1 averaging method (only F1 is tracked)
# To compare multiple models, list them all under 'models'.
# Each model always uses its own default tokenizer.
# models:
#   - OMRIDRORI/mbert-tibetan-continual-wylie-final
#   - OMRIDRORI/mbert-tibetan-continual-unicode-240k
```

Notes:
- `convert_to_wylie: auto` will enable conversion if the model name contains "wylie"/"wyle".
- Set `convert_to_wylie: true` to force conversion, or `false` to disable.
- The tool reads `train.jsonl` and `test.jsonl` by default; override via `files`.
- For multi-input tasks, list both fields in order, e.g. `text_fields: [root, commentary]`.

### Run
From repo root in PowerShell:

```powershell
python fine_tuning\run_finetune.py --config fine_tuning\configs\AACT.yaml
```

### Outputs
- `fine_tuning/results/run_finetune_results/<timestamp>/final_metrics.json` — final evaluation metrics
- `fine_tuning/results/run_finetune_results/<timestamp>/metrics_over_steps.png` — plot of metrics over training steps
- `fine_tuning/results/run_finetune_results/<timestamp>/summary.json` — run summary including config and conversion report
- Trained checkpoints are stored under `experiments/<task>_<model>/` by default (or `output_dir` if provided in config)

### Tips
- If your Wylie model name does not include "wylie"/"wyle", set `convert_to_wylie: true` explicitly.
- If labels are strings and you need a custom mapping, supply `label_map` in the YAML, e.g.:

```yaml
label_map: {"Verse": 0, "Prose": 1}
```

### Troubleshooting
- If you see an error about `pyewts` when converting: install it with `pip install pyewts`.
- For long training, reduce `eval_steps` or `num_epochs`.


