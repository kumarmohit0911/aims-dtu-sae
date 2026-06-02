````md
# Feature Drift in Sparse Autoencoders After Narrow-Domain Fine-Tuning

This repository contains the code, experiments, and analysis for the **AIMS DTU Research Internship 2026 – Mechanistic Interpretability Task**.

The project investigates whether fine-tuning a language model on a narrow domain changes its internal Sparse Autoencoder (SAE) features or primarily alters how existing features are utilized.

---

## Research Question

Large language models learn dense and distributed internal representations that are difficult to interpret directly.

Sparse Autoencoders (SAEs) provide a way to decompose these representations into sparse, interpretable features.

This project investigates the following question:

> When a language model is fine-tuned on a narrow domain, do its internal features fundamentally change, or does the model primarily reuse existing representations?

---

## Approach

The experimental pipeline consists of:

1. Collecting activations from Layer 6 MLP of `EleutherAI/Pythia-160M`
2. Training a Sparse Autoencoder on the collected activations
3. Fine-tuning Pythia-160M on a Python code corpus
4. Training a second SAE on activations from the fine-tuned model
5. Comparing the two feature dictionaries using decoder similarity and activation statistics

---

## Repository Structure

```text
configs/
  base_sae.yaml
  finetune.yaml
  finetuned_sae.yaml

scripts/
  collect_activations.py
  train_sae.py
  finetune_model.py
  compare_saes.py
  inspect_features.py

src/
  data.py
  model_utils.py
  sae.py
  metrics.py
  visualization.py

artifacts/
  activations/
  saes/
  comparisons/
  inspections/

report/
  report.md
````

### Important Scripts

| Script                 | Purpose                         |
| ---------------------- | ------------------------------- |
| collect_activations.py | Extract layer activations       |
| train_sae.py           | Train Sparse Autoencoder        |
| finetune_model.py      | Fine-tune Pythia-160M           |
| compare_saes.py        | Compare SAE feature spaces      |
| inspect_features.py    | Collect top activating contexts |

---

## Setup

### Create Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
pip install -r requirements.txt
```

For CUDA support, install the appropriate PyTorch wheel from the official PyTorch website before installing the remaining dependencies.

If dependency conflicts occur:

```powershell
python -m pip install --upgrade --force-reinstall -r requirements.txt
```

---

## End-to-End Run

### 1. Collect Activations from Pretrained Model

```powershell
python scripts/collect_activations.py --config configs/base_sae.yaml
```

### 2. Train Base SAE

```powershell
python scripts/train_sae.py --config configs/base_sae.yaml
```

### 3. Fine-Tune Pythia-160M

```powershell
python scripts/finetune_model.py --config configs/finetune.yaml
```

### 4. Collect Activations from Fine-Tuned Model

```powershell
python scripts/collect_activations.py --config configs/finetuned_sae.yaml
```

### 5. Train Fine-Tuned SAE

```powershell
python scripts/train_sae.py --config configs/finetuned_sae.yaml
```

### 6. Compare Feature Dictionaries

```powershell
python scripts/compare_saes.py `
  --base-sae artifacts/saes/base_layer6_mlp/sae_final.pt `
  --tuned-sae artifacts/saes/finetuned_layer6_mlp/sae_final.pt `
  --base-stats artifacts/saes/base_layer6_mlp/activation_stats.npz `
  --tuned-stats artifacts/saes/finetuned_layer6_mlp/activation_stats.npz `
  --output-dir artifacts/comparisons/base_vs_python
```

### 7. Inspect Features

```powershell
python scripts/inspect_features.py `
  --config configs/base_sae.yaml `
  --sae-path artifacts/saes/base_layer6_mlp/sae_final.pt `
  --output-dir artifacts/inspections/base
```

```powershell
python scripts/inspect_features.py `
  --config configs/finetuned_sae.yaml `
  --sae-path artifacts/saes/finetuned_layer6_mlp/sae_final.pt `
  --output-dir artifacts/inspections/finetuned
```

---

## Research Outputs

The primary outputs of the experiment are stored in:

```text
artifacts/comparisons/base_vs_python/
```

Key files:

```text
summary.json
feature_matches.csv
decoder_similarity_hist.png
frequency_shift_scatter.png
```

Feature inspection outputs:

```text
artifacts/inspections/base/
artifacts/inspections/finetuned/
```

---

## Results Summary

A complete analysis is available in:

```text
report/report.md
```

The report includes:

* Experimental setup
* SAE training details
* Feature matching methodology
* Quantitative evaluation
* Qualitative feature analysis
* Visualizations
* Discussion and limitations

---

## Notes

The default fine-tuning domain is Python code from:

```text
flytech/python-codes-25k
```

Python was selected because it contains highly recognizable structural patterns such as:

* indentation
* imports
* comments
* string literals
* function definitions
* class definitions
* exception handling

These characteristics make feature specialization easier to identify and interpret.

---

## Author

**Kumar Mohit**

AIMS DTU Research Internship 2026
Mechanistic Interpretability Track

```
```
