from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import ensure_dir, load_config
from src.data import batch_texts, iter_texts
from src.model_utils import capture_module_output, load_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=250000)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg["activation_dir"])
    model, tokenizer = load_lm(cfg["model_name"])
    data_cfg = cfg["data"]
    texts = iter_texts(
        data_cfg["dataset_name"],
        data_cfg.get("dataset_config"),
        data_cfg["split"],
        data_cfg["text_column"],
        data_cfg.get("max_examples"),
    )

    current: list[torch.Tensor] = []
    shard_idx = 0
    total_tokens = 0
    device = next(model.parameters()).device

    for text_batch in tqdm(batch_texts(texts, args.text_batch_size), desc="collecting"):
        encoded = tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=data_cfg["max_length"],
        ).to(device)
        captured: list[torch.Tensor] = []
        with capture_module_output(model, cfg["layer_name"], captured):
            model(**encoded)
        acts = captured[-1].float().cpu()
        mask = encoded["attention_mask"].bool().cpu()
        flat = acts[mask]
        current.append(flat)
        total_tokens += flat.shape[0]

        if sum(t.shape[0] for t in current) >= args.shard_size:
            shard = torch.cat(current, dim=0)
            torch.save(shard, out_dir / f"acts_{shard_idx:04d}.pt")
            current = []
            shard_idx += 1

    if current:
        shard = torch.cat(current, dim=0)
        torch.save(shard, out_dir / f"acts_{shard_idx:04d}.pt")

    metadata = {
        "model_name": cfg["model_name"],
        "layer_name": cfg["layer_name"],
        "total_tokens": total_tokens,
        "input_dim": cfg["sae"]["input_dim"],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
