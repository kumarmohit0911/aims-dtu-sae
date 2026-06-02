# Feature Drift in Sparse Autoencoders After Narrow-Domain Fine-Tuning

This repository is a reproducible scaffold for the AIMS DTU Research Intern 2026
mechanistic interpretability preview task. It trains a sparse autoencoder (SAE)
on a middle layer of `EleutherAI/pythia-160m`, fine-tunes the base model on a
narrow Python-code corpus, trains a second SAE on the same layer, and compares
the two learned feature spaces.

The code is intentionally small and readable. It is suitable for a single GPU
run and can be scaled by increasing the activation count, SAE width, and number
of training steps.

## Project Layout

```text
configs/
  base_sae.yaml          # Base model activation + SAE config
  finetune.yaml          # Domain fine-tuning config
  finetuned_sae.yaml     # Fine-tuned model activation + SAE config
scripts/
  collect_activations.py # Saves layer activations as tensor shards
  train_sae.py           # Trains a ReLU SAE on cached activations
  finetune_model.py      # Fine-tunes Pythia-160M on Python code
  compare_saes.py        # Matches SAE features and reports drift metrics
  inspect_features.py    # Collects top activating token contexts
src/
  data.py
  model_utils.py
  sae.py
  metrics.py
  visualization.py
report/
  report.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For CUDA, install the PyTorch wheel that matches your driver from the official
PyTorch selector before installing the remaining requirements. If you previously
installed unpinned future versions of Hugging Face packages, run:

```powershell
python -m pip install --upgrade --force-reinstall -r requirements.txt
```

## End-to-End Run

Collect activations from layer 6 MLP of the pretrained model:

```powershell
python scripts/collect_activations.py --config configs/base_sae.yaml
```

Train the base SAE:

```powershell
python scripts/train_sae.py --config configs/base_sae.yaml
```

Fine-tune Pythia-160M on Python code:

```powershell
python scripts/finetune_model.py --config configs/finetune.yaml
```

Collect activations from the same layer of the fine-tuned model:

```powershell
python scripts/collect_activations.py --config configs/finetuned_sae.yaml
```

Train the fine-tuned SAE:

```powershell
python scripts/train_sae.py --config configs/finetuned_sae.yaml
```

Compare SAE features:

```powershell
python scripts/compare_saes.py `
  --base-sae artifacts/saes/base_layer6_mlp/sae_final.pt `
  --tuned-sae artifacts/saes/finetuned_layer6_mlp/sae_final.pt `
  --base-stats artifacts/saes/base_layer6_mlp/activation_stats.npz `
  --tuned-stats artifacts/saes/finetuned_layer6_mlp/activation_stats.npz `
  --output-dir artifacts/comparisons/base_vs_python
```

Inspect top activating contexts:

```powershell
python scripts/inspect_features.py `
  --config configs/base_sae.yaml `
  --sae-path artifacts/saes/base_layer6_mlp/sae_final.pt `
  --output-dir artifacts/inspections/base

python scripts/inspect_features.py `
  --config configs/finetuned_sae.yaml `
  --sae-path artifacts/saes/finetuned_layer6_mlp/sae_final.pt `
  --output-dir artifacts/inspections/finetuned
```

## What To Submit

The `report/report.md` file is a 4-6 page technical report draft. After running
the experiment, fill in the result table from:

- `artifacts/comparisons/base_vs_python/summary.json`
- `artifacts/comparisons/base_vs_python/feature_matches.csv`
- `artifacts/inspections/base/top_feature_examples.json`
- `artifacts/inspections/finetuned/top_feature_examples.json`

Large files such as model checkpoints, SAE weights, and activation caches should
be uploaded to Google Drive or Hugging Face and linked in the report.

## Notes

The default narrow domain is Python code from `flytech/python-codes-25k` because
code has sharply recognizable features: indentation, punctuation, imports,
comments, string literals, function definitions, class definitions, and
error-handling idioms. These make qualitative feature inspection easier than
with many prose domains.
