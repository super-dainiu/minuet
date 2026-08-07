"""Production Minuet v3 objectives.

Donor labels define the paired galleries and paired-target centering. Recorded
biological context is used only to measure residual donor association.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def donor_centered_exact_pair_infonce(
    rna: Tensor, atac: Tensor, donor: Tensor, *, temperature: float = 0.05
) -> Tensor:
    """Symmetric exact-pair InfoNCE, centered and averaged by donor."""
    terms: list[Tensor] = []
    for value in torch.unique(donor[donor >= 0]):
        idx = torch.nonzero(donor == value, as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        x = F.normalize(rna[idx] - rna[idx].mean(0, keepdim=True), dim=-1)
        y = F.normalize(atac[idx] - atac[idx].mean(0, keepdim=True), dim=-1)
        logits = x @ y.T / temperature
        labels = torch.arange(idx.numel(), device=rna.device)
        terms.append(0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)))
    return torch.stack(terms).mean() if terms else (rna.sum() + atac.sum()) * 0.0


def donor_centered_smooth_l1(prediction: Tensor, target: Tensor, donor: Tensor) -> Tensor:
    """Smooth-L1 after separate within-donor centering; donors are equally weighted."""
    terms: list[Tensor] = []
    for value in torch.unique(donor[donor >= 0]):
        block = donor == value
        if int(block.sum()) < 2:
            continue
        pred = prediction[block] - prediction[block].mean(0, keepdim=True)
        truth = target[block] - target[block].mean(0, keepdim=True)
        terms.append(F.smooth_l1_loss(pred, truth))
    return torch.stack(terms).mean() if terms else prediction.sum() * 0.0


def context_centered_donor_association(
    z: Tensor, donor: Tensor, biology_context: Tensor, *, eps: float = 1e-12
) -> Tensor:
    """Normalized linear donor association, computed separately within contexts."""
    terms: list[Tensor] = []
    labelled = (donor >= 0) & (biology_context >= 0)
    if not labelled.any():
        return z.sum() * 0.0
    z, donor, biology_context = z[labelled], donor[labelled], biology_context[labelled]
    n_donors = int(donor.max()) + 1
    for value in torch.unique(biology_context):
        idx = torch.nonzero(biology_context == value, as_tuple=False).flatten()
        if idx.numel() < 4 or torch.unique(donor[idx]).numel() < 2:
            continue
        x = z[idx] - z[idx].mean(0, keepdim=True)
        d = F.one_hot(donor[idx], n_donors).to(z.dtype)
        d = d - d.mean(0, keepdim=True)
        numerator = (x.T @ d).square().sum()
        denominator = ((x.T @ x).square().sum().sqrt() * (d.T @ d).square().sum().sqrt()).detach()
        terms.append(numerator / denominator.clamp_min(eps))
    return torch.stack(terms).mean() if terms else z.sum() * 0.0


def gaussian_kl(mu: Tensor, logvar: Tensor) -> Tensor:
    return 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(-1).mean()


class MinuetLosses:
    """Small public collection matching the production v3 roles."""

    donor_centered_exact_pair_infonce = staticmethod(donor_centered_exact_pair_infonce)
    donor_centered_smooth_l1 = staticmethod(donor_centered_smooth_l1)
    context_centered_donor_association = staticmethod(context_centered_donor_association)

    @classmethod
    def total(
        cls, rna_x: Tensor, atac_x: Tensor, out: dict[str, Tensor], *, donor: Tensor,
        biology_context: Tensor, paired_target: Tensor, target_head: torch.nn.Module,
        pair_weight: float = 2.466, prediction_weight: float = 0.1,
        donor_association_weight: float = 0.1, private_kl_weight: float = 7.23e-5,
        temperature: float = 0.0355,
    ) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}
        losses["reconstruction"] = F.mse_loss(out["rna_recon"], rna_x) + F.mse_loss(out["atac_recon"], atac_x)
        losses["private_kl"] = gaussian_kl(out["rna_private_mu"], out["rna_private_logvar"]) + gaussian_kl(out["atac_private_mu"], out["atac_private_logvar"])
        losses["pair"] = donor_centered_exact_pair_infonce(out["rna_shared"], out["atac_shared"], donor, temperature=temperature)
        losses["prediction"] = 0.5 * (
            donor_centered_smooth_l1(target_head(out["rna_shared"]), paired_target, donor)
            + donor_centered_smooth_l1(target_head(out["atac_shared"]), paired_target, donor)
        )
        losses["donor_association"] = context_centered_donor_association(out["joint_shared"], donor, biology_context)
        losses["total"] = (losses["reconstruction"] + private_kl_weight * losses["private_kl"]
                           + pair_weight * losses["pair"] + prediction_weight * losses["prediction"]
                           + donor_association_weight * losses["donor_association"])
        return losses
