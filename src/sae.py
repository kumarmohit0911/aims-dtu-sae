from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SAELoss:
    total: torch.Tensor
    reconstruction: torch.Tensor
    sparsity: torch.Tensor
    l0: torch.Tensor


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
        nn.init.zeros_(self.decoder.bias)
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    def loss(self, x: torch.Tensor, l1_coefficient: float) -> SAELoss:
        reconstruction, features = self.forward(x)
        reconstruction_loss = F.mse_loss(reconstruction, x)
        sparsity_loss = features.abs().mean()
        l0 = (features > 0).float().sum(dim=-1).mean()
        total = reconstruction_loss + l1_coefficient * sparsity_loss
        return SAELoss(total, reconstruction_loss, sparsity_loss, l0)


def load_sae(path: str, map_location: str | torch.device = "cpu") -> SparseAutoencoder:
    checkpoint = torch.load(path, map_location=map_location)
    config = checkpoint["config"]
    model = SparseAutoencoder(config["input_dim"], config["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
