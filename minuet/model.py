"""Compact Minuet model and encoder-only deployment artifact."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PRODUCTION_HIDDEN_DIM = 224
PRODUCTION_SHARED_DIM = 64
PRODUCTION_PRIVATE_DIM = 16
PRODUCTION_DECODER_RANK = 64
PRODUCTION_RESIDUAL_BLOCKS = 3
PRODUCTION_DROPOUT = 0.18597052969975905
PRODUCTION_UPDATES = 16_000
PRODUCTION_LEARNING_RATE = 0.0003090937377859925
PRODUCTION_WEIGHT_DECAY = 3.569446559492511e-7
PRODUCTION_LATE_LR_MULTIPLIER = 0.2410370721169952


@dataclass(frozen=True)
class MinuetConfig:
    rna_dim: int
    atac_dim: int
    hidden_dim: int = PRODUCTION_HIDDEN_DIM
    shared_dim: int = PRODUCTION_SHARED_DIM
    private_dim: int = PRODUCTION_PRIVATE_DIM
    decoder_rank: int = PRODUCTION_DECODER_RANK
    residual_blocks: int = PRODUCTION_RESIDUAL_BLOCKS
    dropout: float = PRODUCTION_DROPOUT
    min_logvar: float = -8.0
    max_logvar: float = 8.0

    def __post_init__(self) -> None:
        for name in (
            "rna_dim",
            "atac_dim",
            "hidden_dim",
            "shared_dim",
            "private_dim",
            "decoder_rank",
            "residual_blocks",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.min_logvar >= self.max_logvar:
            raise ValueError("min_logvar must be smaller than max_logvar")


class _ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, blocks: int) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(blocks)
        )
        self.output_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.input(x)
        for block in self.residual:
            hidden = hidden + block(hidden)
        return self.output_norm(hidden)


class _ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, config: MinuetConfig) -> None:
        super().__init__()
        self.trunk = _ResidualMLP(
            input_dim, config.hidden_dim, config.dropout, config.residual_blocks
        )
        self.shared = nn.Linear(config.hidden_dim, config.shared_dim)
        self.private_mu = nn.Linear(config.hidden_dim, config.private_dim)
        self.private_logvar = nn.Linear(config.hidden_dim, config.private_dim)
        self.min_logvar = config.min_logvar
        self.max_logvar = config.max_logvar

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        hidden = self.trunk(x)
        return {
            "shared": self.shared(hidden),
            "private_mu": self.private_mu(hidden),
            "private_logvar": self.private_logvar(hidden).clamp(
                self.min_logvar, self.max_logvar
            ),
        }


class _PrivateDecoder(nn.Module):
    def __init__(self, output_dim: int, config: MinuetConfig) -> None:
        super().__init__()
        self.projection = nn.Linear(config.private_dim, config.decoder_rank)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.decoder_rank, output_dim)

    def forward(self, private: Tensor) -> Tensor:
        return self.output(self.dropout(self.activation(self.projection(private))))


def _sample_gaussian(mu: Tensor, logvar: Tensor, sample: bool) -> Tensor:
    if not sample:
        return mu
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


def _validate_inputs(config: MinuetConfig, rna: Tensor, atac: Tensor) -> None:
    if rna.ndim != 2 or atac.ndim != 2:
        raise ValueError("RNA and ATAC inputs must be two-dimensional")
    if rna.shape[0] != atac.shape[0]:
        raise ValueError("RNA and ATAC must contain the same paired cells")
    if rna.shape[1] != config.rna_dim or atac.shape[1] != config.atac_dim:
        raise ValueError("input feature dimensions do not match the model")
    if rna.device != atac.device:
        raise ValueError("RNA and ATAC inputs must share a device")
    if not rna.is_floating_point() or not atac.is_floating_point():
        raise TypeError("RNA and ATAC inputs must have floating-point dtypes")


def _encode_pair(
    config: MinuetConfig,
    rna_encoder: nn.Module,
    atac_encoder: nn.Module,
    rna: Tensor,
    atac: Tensor,
) -> dict[str, Tensor]:
    _validate_inputs(config, rna, atac)
    rna_q = rna_encoder(rna)
    atac_q = atac_encoder(atac)
    return {
        "rna_shared": rna_q["shared"],
        "atac_shared": atac_q["shared"],
        "joint_shared": 0.5 * (rna_q["shared"] + atac_q["shared"]),
        "rna_private": rna_q["private_mu"],
        "atac_private": atac_q["private_mu"],
        "rna_private_logvar": rna_q["private_logvar"],
        "atac_private_logvar": atac_q["private_logvar"],
    }


class MinuetEncoder(nn.Module):
    """Frozen per-cell encoder; its forward pass accepts measurements only."""

    checkpoint_schema = "minuet-encoder-only-v1"
    _checkpoint_keys = frozenset({"schema", "model_config", "state_dict"})

    def __init__(self, config: MinuetConfig) -> None:
        super().__init__()
        self.config = config
        self.rna_encoder = _ModalityEncoder(config.rna_dim, config)
        self.atac_encoder = _ModalityEncoder(config.atac_dim, config)

    def encode(self, rna: Tensor, atac: Tensor) -> dict[str, Tensor]:
        return _encode_pair(self.config, self.rna_encoder, self.atac_encoder, rna, atac)

    def forward(self, rna: Tensor, atac: Tensor) -> dict[str, Tensor]:
        return self.encode(rna, atac)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def checkpoint(self) -> dict[str, Any]:
        return {
            "schema": self.checkpoint_schema,
            "model_config": asdict(self.config),
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.state_dict().items()
            },
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "MinuetEncoder":
        if set(checkpoint) != cls._checkpoint_keys:
            raise ValueError("encoder checkpoint keys do not match the strict schema")
        if checkpoint.get("schema") != cls.checkpoint_schema:
            raise ValueError("unexpected encoder checkpoint schema")
        config_values = checkpoint.get("model_config")
        state = checkpoint.get("state_dict")
        if not isinstance(config_values, Mapping) or not isinstance(state, Mapping):
            raise TypeError("encoder checkpoint config and state must be mappings")
        encoder = cls(MinuetConfig(**dict(config_values)))
        encoder.load_state_dict(dict(state), strict=True)
        encoder.eval()
        encoder.requires_grad_(False)
        return encoder


class MinuetModule(nn.Module):
    """Training model with deterministic shared and Gaussian-private variables."""

    def __init__(self, config: MinuetConfig) -> None:
        super().__init__()
        self.config = config
        self.rna_encoder = _ModalityEncoder(config.rna_dim, config)
        self.atac_encoder = _ModalityEncoder(config.atac_dim, config)
        self.rna_decoder = _PrivateDecoder(config.rna_dim, config)
        self.atac_decoder = _PrivateDecoder(config.atac_dim, config)
        self.rna_log_inverse_dispersion = nn.Parameter(torch.zeros(config.rna_dim))

    def encode(self, rna: Tensor, atac: Tensor) -> dict[str, Tensor]:
        return _encode_pair(self.config, self.rna_encoder, self.atac_encoder, rna, atac)

    @staticmethod
    def _rna_rate(logits: Tensor, library_size: Tensor) -> Tensor:
        if library_size.ndim != 1 or library_size.shape[0] != logits.shape[0]:
            raise ValueError("RNA library_size must have one value per cell")
        return F.softmax(logits, dim=-1) * library_size.clamp_min(1.0).unsqueeze(-1)

    def forward(
        self,
        rna: Tensor,
        atac: Tensor,
        *,
        rna_library_size: Tensor | None = None,
        sample: bool | None = None,
    ) -> dict[str, Tensor]:
        encoded = self.encode(rna, atac)
        should_sample = self.training if sample is None else bool(sample)
        rna_private = _sample_gaussian(
            encoded["rna_private"], encoded["rna_private_logvar"], should_sample
        )
        atac_private = _sample_gaussian(
            encoded["atac_private"], encoded["atac_private_logvar"], should_sample
        )
        rna_logits = self.rna_decoder(rna_private)
        atac_logits = self.atac_decoder(atac_private)
        if rna_library_size is None:
            rna_library_size = rna.sum(dim=-1)
        return {
            **encoded,
            "rna_rate": self._rna_rate(rna_logits, rna_library_size),
            "atac_logits": atac_logits,
            "rna_inverse_dispersion": F.softplus(
                self.rna_log_inverse_dispersion
            ).clamp_min(1.0e-4),
        }

    def encoder(self) -> MinuetEncoder:
        return MinuetEncoder.from_checkpoint(self.inference_checkpoint())

    def inference_checkpoint(self) -> dict[str, Any]:
        encoder = MinuetEncoder(self.config)
        encoder.rna_encoder.load_state_dict(self.rna_encoder.state_dict(), strict=True)
        encoder.atac_encoder.load_state_dict(self.atac_encoder.state_dict(), strict=True)
        return encoder.checkpoint()

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
