from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.metrics import decoder_directions, match_features
from src.sae import load_sae
from src.visualization import save_shift_scatter, save_similarity_histogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sae", required=True)
    parser.add_argument("--tuned-sae", required=True)
    parser.add_argument("--base-stats", required=True)
    parser.add_argument("--tuned-stats", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_sae(args.base_sae)
    tuned = load_sae(args.tuned_sae)
    matched = match_features(decoder_directions(base), decoder_directions(tuned))
    base_stats = np.load(args.base_stats)
    tuned_stats = np.load(args.tuned_stats)

    df = pd.DataFrame(
        {
            "base_feature": matched["base_feature"],
            "tuned_feature": matched["tuned_feature"],
            "decoder_cosine": matched["decoder_cosine"],
        }
    )
    df["base_frequency"] = base_stats["frequency"][df["base_feature"].to_numpy()]
    df["tuned_frequency"] = tuned_stats["frequency"][df["tuned_feature"].to_numpy()]
    df["frequency_delta"] = df["tuned_frequency"] - df["base_frequency"]
    df["frequency_delta_abs"] = df["frequency_delta"].abs()
    df["base_mean_activation"] = base_stats["mean_activation"][df["base_feature"].to_numpy()]
    df["tuned_mean_activation"] = tuned_stats["mean_activation"][df["tuned_feature"].to_numpy()]
    df["mean_activation_delta"] = df["tuned_mean_activation"] - df["base_mean_activation"]

    df["category"] = "stable"
    df.loc[df["decoder_cosine"] < 0.35, "category"] = "reoriented"
    df.loc[(df["decoder_cosine"] >= 0.35) & (df["frequency_delta_abs"] > 0.02), "category"] = "frequency_shifted"

    df.sort_values(["category", "decoder_cosine"], inplace=True)
    df.to_csv(out_dir / "feature_matches.csv", index=False)
    save_similarity_histogram(df, out_dir / "decoder_similarity_hist.png")
    save_shift_scatter(df, out_dir / "frequency_shift_scatter.png")

    summary = {
        "num_features": int(len(df)),
        "median_decoder_cosine": float(df["decoder_cosine"].median()),
        "mean_decoder_cosine": float(df["decoder_cosine"].mean()),
        "reoriented_features": int((df["category"] == "reoriented").sum()),
        "frequency_shifted_features": int((df["category"] == "frequency_shifted").sum()),
        "stable_features": int((df["category"] == "stable").sum()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
