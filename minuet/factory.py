"""Factory: build a Minuet model and compute its total loss from a config dict.

The Minuet recipe is the canonical (former Apollo-V3) two-loss decomposition:
  - NT-Xent contrastive  -> RNA<->ATAC alignment
  - CMI-BI               -> within-modality batch-conditional cluster geometry
"""
from __future__ import annotations

from typing import Any

import torch

from .losses import MinuetLosses
from .model import Minuet, MinuetConfig


def build_model_from_section(
    model_section: dict[str, Any],
    *,
    rna_dim: int,
    atac_dim: int,
    device: str,
) -> tuple[str, dict[str, Any], torch.nn.Module]:
    n_factor_tokens = int(model_section.get("n_factor_tokens", 2))
    legacy_latent_tokens = model_section.get("n_latent_tokens")
    n_cell_tokens = int(model_section.get("n_cell_tokens", n_factor_tokens))
    n_context_tokens = int(model_section.get("n_context_tokens", n_factor_tokens))
    n_private_tokens = int(model_section.get("n_private_tokens", n_factor_tokens))
    num_encoder_blocks = int(model_section.get("num_encoder_blocks", model_section.get("num_feature_blocks", 2)))
    num_joint_blocks = int(model_section.get("num_joint_blocks", model_section.get("num_blocks", 4)))
    if legacy_latent_tokens is not None and "n_memory_tokens" not in model_section:
        n_memory_tokens = max(int(legacy_latent_tokens) - (n_cell_tokens + n_context_tokens + n_private_tokens), 0)
    else:
        n_memory_tokens = int(model_section.get("n_memory_tokens", 8))
    cfg = MinuetConfig(
        rna_dim=rna_dim,
        atac_dim=atac_dim,
        group_size=int(model_section.get("group_size", 64)),
        token_dim=int(model_section.get("token_dim", 192)),
        n_cell_tokens=n_cell_tokens,
        n_context_tokens=n_context_tokens,
        n_private_tokens=n_private_tokens,
        n_memory_tokens=n_memory_tokens,
        num_heads=int(model_section.get("num_heads", 4)),
        num_encoder_blocks=num_encoder_blocks,
        num_joint_blocks=num_joint_blocks,
        ff_multiplier=int(model_section.get("ff_multiplier", 4)),
        token_dropout=float(model_section.get("token_dropout", 0.0)),
        cell_dim=int(model_section.get("cell_dim", 32)),
        context_dim=int(model_section.get("context_dim", 32)),
        private_dim=int(model_section["private_dim"]),
        decoder_hidden_dims=tuple(model_section.get("decoder_hidden_dims", [512, 1024])),
        dropout=float(model_section["dropout"]),
        min_logvar=float(model_section.get("min_logvar", -8.0)),
        max_logvar=float(model_section.get("max_logvar", 8.0)),
        n_batches=int(model_section.get("n_batches", 0)),
        batch_embed_dim=int(model_section.get("batch_embed_dim", 0)),
        deep_tokenizer=bool(model_section.get("deep_tokenizer", False)),
        pooler_mode=bool(model_section.get("pooler_mode", False)),
        cell_pool_feature=bool(model_section.get("cell_pool_feature", False)),
        use_encoder_batch_cov=bool(model_section.get("use_encoder_batch_cov", False)),
        adversarial_batch=bool(model_section.get("adversarial_batch", False)),
        adversary_hidden=int(model_section.get("adversary_hidden", 64)),
        structured_latent=bool(model_section.get("structured_latent", False)),
        nuisance_pred_hidden=int(model_section.get("nuisance_pred_hidden", 64)),
    )
    return "v3", {"type": "v3", **cfg.__dict__}, Minuet(cfg).to(device)


def load_model_from_checkpoint(checkpoint: dict[str, Any], device: str) -> tuple[str, torch.nn.Module]:
    model_cfg = dict(checkpoint["model_cfg"])
    model_cfg["type"] = "v3"
    _, _, model = build_model_from_section(
        model_cfg,
        rna_dim=int(model_cfg["rna_dim"]),
        atac_dim=int(model_cfg["atac_dim"]),
        device=device,
    )
    load_compatible_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    return "v3", model


def load_compatible_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
) -> dict[str, list[str]]:
    """Load weights, skipping keys that don't shape-match.

    Lets us load Minuet checkpoints that include training-only auxiliary
    parameters (e.g., the CMI-BI CLUB estimator) into a plain inference
    model rebuilt from config.
    """
    model_state = model.state_dict()
    filtered_state: dict[str, Any] = {}
    skipped_mismatch: list[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if getattr(model_state[key], "shape", None) != getattr(value, "shape", None):
            skipped_mismatch.append(key)
            continue
        filtered_state[key] = value

    incompatible = model.load_state_dict(filtered_state, strict=False)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "skipped_mismatch": skipped_mismatch,
    }


def compute_total_losses(
    model_type: str,
    *,
    loss_cfg: dict[str, Any],
    rna_x,
    atac_x,
    out,
):
    return MinuetLosses.total(
        rna_x,
        atac_x,
        out,
        recon_weight=loss_cfg["recon_weight"],
        alignment_weight=loss_cfg["alignment_weight"],
        contrastive_weight=loss_cfg.get("contrastive_weight", 1.0),
        shared_recon_weight=loss_cfg.get("shared_recon_weight", 0.0),
        self_shared_recon_weight=loss_cfg.get("self_shared_recon_weight", 0.0),
        kl_shared_weight=loss_cfg.get("kl_shared_weight", 1.0e-3),
        kl_private_weight=loss_cfg.get("kl_private_weight", loss_cfg.get("private_weight", 1.0e-3)),
        fusion_weight=loss_cfg.get("fusion_weight", 1.0e-3),
        decouple_weight=loss_cfg.get("decouple_weight", 1.0e-3),
        adv_weight=loss_cfg.get("adv_weight", 0.0),
        nuisance_pred_weight=loss_cfg.get("nuisance_pred_weight", 0.0),
        temperature=loss_cfg.get("temperature", 0.07),
    )
