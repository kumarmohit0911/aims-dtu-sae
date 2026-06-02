from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
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
from src.sae import load_sae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-features", type=int, default=64)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(args.output_dir)
    model, tokenizer = load_lm(cfg["model_name"])
    device = next(model.parameters()).device
    sae = load_sae(args.sae_path).to(device)
    data_cfg = cfg["data"]
    texts = iter_texts(
        data_cfg["dataset_name"],
        data_cfg.get("dataset_config"),
        data_cfg["split"],
        data_cfg["text_column"],
        min(data_cfg.get("max_examples", 5000), 5000),
    )

    top_examples: dict[int, list[tuple[float, str, str]]] = defaultdict(list)
    feature_limit = min(args.num_features, sae.hidden_dim)

    for text_batch in tqdm(batch_texts(texts, 4), desc="inspecting"):
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
        acts = captured[-1].float()
        features = sae.encode(acts)
        tokens = encoded["input_ids"]
        mask = encoded["attention_mask"].bool()

        for feature_idx in range(feature_limit):
            values = features[:, :, feature_idx].masked_fill(~mask, 0.0)
            flat_vals, flat_idx = values.reshape(-1).topk(min(args.top_k, values.numel()))
            for value, flat_position in zip(flat_vals.tolist(), flat_idx.tolist()):
                if value <= 0:
                    continue
                batch_idx = flat_position // values.shape[1]
                pos_idx = flat_position % values.shape[1]
                token = tokenizer.decode(tokens[batch_idx, pos_idx])
                context = tokenizer.decode(tokens[batch_idx, max(0, pos_idx - 16) : pos_idx + 17])
                top_examples[feature_idx].append((float(value), token, context))
            top_examples[feature_idx] = sorted(
                top_examples[feature_idx],
                key=lambda item: item[0],
                reverse=True,
            )[: args.top_k]

    serializable = {
        str(feature): [
            {"activation": act, "token": token, "context": context}
            for act, token, context in examples
        ]
        for feature, examples in top_examples.items()
    }
    (out_dir / "top_feature_examples.json").write_text(
        json.dumps(serializable, indent=2),
        encoding="utf-8",
    )
    print(f"Saved top feature examples to {out_dir / 'top_feature_examples.json'}")


if __name__ == "__main__":
    main()
