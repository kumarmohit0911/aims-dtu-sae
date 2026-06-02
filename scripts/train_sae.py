from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from rich.table import Table
from rich.console import Console
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import ensure_dir, load_config
from src.metrics import activation_summary
from src.sae import SparseAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-activation-shards", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_activations(directory: str, max_shards: int | None) -> torch.Tensor:
    paths = sorted(Path(directory).glob("acts_*.pt"))
    if max_shards is not None:
        paths = paths[:max_shards]
    if not paths:
        raise FileNotFoundError(f"No activation shards found in {directory}")
    return torch.cat([torch.load(path, map_location="cpu") for path in paths], dim=0)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    sae_cfg = cfg["sae"]
    out_dir = ensure_dir(cfg["output_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    activations = load_activations(cfg["activation_dir"], args.max_activation_shards)
    hidden_dim = sae_cfg["input_dim"] * sae_cfg["expansion_factor"]
    sae = SparseAutoencoder(sae_cfg["input_dim"], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=sae_cfg["learning_rate"])

    table = Table(title="SAE training")
    for col in ["step", "loss", "mse", "l1", "l0"]:
        table.add_column(col)
    console = Console()

    n = activations.shape[0]
    batch_size = sae_cfg["batch_size"]
    last_metrics = {}

    for step in trange(1, sae_cfg["steps"] + 1, desc="training"):
        indices = torch.randint(0, n, (batch_size,))
        batch = activations[indices].to(device)
        loss = sae.loss(batch, sae_cfg["l1_coefficient"])
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        optimizer.step()
        sae.normalize_decoder()

        if step % sae_cfg["log_every"] == 0:
            last_metrics = {
                "loss": float(loss.total.detach().cpu()),
                "reconstruction": float(loss.reconstruction.detach().cpu()),
                "sparsity": float(loss.sparsity.detach().cpu()),
                "l0": float(loss.l0.detach().cpu()),
            }
            table.add_row(
                str(step),
                f"{last_metrics['loss']:.5f}",
                f"{last_metrics['reconstruction']:.5f}",
                f"{last_metrics['sparsity']:.5f}",
                f"{last_metrics['l0']:.2f}",
            )

        if step % sae_cfg["save_every"] == 0:
            checkpoint = {
                "state_dict": sae.state_dict(),
                "config": {"input_dim": sae_cfg["input_dim"], "hidden_dim": hidden_dim},
                "training_config": sae_cfg,
                "step": step,
            }
            torch.save(checkpoint, out_dir / f"sae_step_{step}.pt")

    console.print(table)
    with torch.no_grad():
        sample = activations[torch.randperm(n)[: min(n, 100000)]].to(device)
        features = sae.encode(sample)
        summary = activation_summary(features, threshold=0.0)

    np.savez(out_dir / "activation_stats.npz", **summary)
    final = {
        "state_dict": sae.state_dict(),
        "config": {"input_dim": sae_cfg["input_dim"], "hidden_dim": hidden_dim},
        "training_config": sae_cfg,
        "last_metrics": last_metrics,
    }
    torch.save(final, out_dir / "sae_final.pt")
    (out_dir / "metrics.json").write_text(json.dumps(last_metrics, indent=2), encoding="utf-8")
    print(f"Saved SAE to {out_dir / 'sae_final.pt'}")


if __name__ == "__main__":
    main()
