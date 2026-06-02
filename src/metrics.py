from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def decoder_directions(sae) -> np.ndarray:
    weights = sae.decoder.weight.detach().cpu().numpy()
    return weights.T


def match_features(base_dirs: np.ndarray, tuned_dirs: np.ndarray) -> dict[str, np.ndarray]:
    sims = cosine_similarity(base_dirs, tuned_dirs)
    row_ind, col_ind = linear_sum_assignment(-sims)
    return {
        "base_feature": row_ind,
        "tuned_feature": col_ind,
        "decoder_cosine": sims[row_ind, col_ind],
        "similarity_matrix": sims,
    }


def activation_summary(features: torch.Tensor, threshold: float = 0.0) -> dict[str, np.ndarray]:
    active = features > threshold
    return {
        "frequency": active.float().mean(dim=0).cpu().numpy(),
        "mean_activation": features.mean(dim=0).cpu().numpy(),
        "mean_active_activation": (
            features.sum(dim=0) / active.float().sum(dim=0).clamp_min(1)
        ).cpu().numpy(),
    }


def jaccard_top_tokens(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)
