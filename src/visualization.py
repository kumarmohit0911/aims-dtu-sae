from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_similarity_histogram(matches: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    sns.histplot(matches["decoder_cosine"], bins=40)
    plt.xlabel("Matched decoder cosine similarity")
    plt.ylabel("Feature count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_shift_scatter(matches: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=matches,
        x="decoder_cosine",
        y="frequency_delta_abs",
        s=12,
        linewidth=0,
    )
    plt.xlabel("Matched decoder cosine similarity")
    plt.ylabel("Absolute activation frequency shift")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
