param(
  [switch]$SkipFineTune
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
  $PSNativeCommandUseErrorActionPreference = $true
}
$env:USE_TF = "0"
$env:TRANSFORMERS_NO_TF = "1"

function Invoke-Step {
  param(
    [Parameter(Mandatory=$true)]
    [string]$Name,
    [Parameter(Mandatory=$true)]
    [scriptblock]$Command
  )

  Write-Host ""
  Write-Host "==> $Name"
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Name"
  }
}

python -c "import datasets, transformers, torch, scipy, sklearn, pandas, yaml, tqdm, rich" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Error "Missing Python dependencies. Run: pip install -r requirements.txt"
}

if (-not (Test-Path "artifacts/activations/base_layer6_mlp/metadata.json")) {
  Invoke-Step "Collect base activations" {
    python scripts/collect_activations.py --config configs/base_sae.yaml
  }
} else {
  Write-Host "Skipping base activation collection; artifacts already exist."
}

if (-not (Test-Path "artifacts/saes/base_layer6_mlp/sae_final.pt")) {
  Invoke-Step "Train base SAE" {
    python scripts/train_sae.py --config configs/base_sae.yaml
  }
} else {
  Write-Host "Skipping base SAE training; artifacts already exist."
}

if (-not $SkipFineTune) {
  if (-not (Test-Path "artifacts/models/pythia160m_python_code/config.json")) {
    Invoke-Step "Fine-tune model" {
      python scripts/finetune_model.py --config configs/finetune.yaml
    }
  } else {
    Write-Host "Skipping fine-tuning; model artifacts already exist."
  }
}

if (-not (Test-Path "artifacts/models/pythia160m_python_code/config.json")) {
  throw "Fine-tuned model is missing. Run without -SkipFineTune first."
}

if (-not (Test-Path "artifacts/activations/finetuned_layer6_mlp/metadata.json")) {
  Invoke-Step "Collect fine-tuned activations" {
    python scripts/collect_activations.py --config configs/finetuned_sae.yaml
  }
} else {
  Write-Host "Skipping fine-tuned activation collection; artifacts already exist."
}

if (-not (Test-Path "artifacts/saes/finetuned_layer6_mlp/sae_final.pt")) {
  Invoke-Step "Train fine-tuned SAE" {
    python scripts/train_sae.py --config configs/finetuned_sae.yaml
  }
} else {
  Write-Host "Skipping fine-tuned SAE training; artifacts already exist."
}

Invoke-Step "Compare SAEs" {
  python scripts/compare_saes.py `
    --base-sae artifacts/saes/base_layer6_mlp/sae_final.pt `
    --tuned-sae artifacts/saes/finetuned_layer6_mlp/sae_final.pt `
    --base-stats artifacts/saes/base_layer6_mlp/activation_stats.npz `
    --tuned-stats artifacts/saes/finetuned_layer6_mlp/activation_stats.npz `
    --output-dir artifacts/comparisons/base_vs_python
}

Invoke-Step "Inspect base features" {
  python scripts/inspect_features.py `
    --config configs/base_sae.yaml `
    --sae-path artifacts/saes/base_layer6_mlp/sae_final.pt `
    --output-dir artifacts/inspections/base
}

Invoke-Step "Inspect fine-tuned features" {
  python scripts/inspect_features.py `
    --config configs/finetuned_sae.yaml `
    --sae-path artifacts/saes/finetuned_layer6_mlp/sae_final.pt `
    --output-dir artifacts/inspections/finetuned
}
