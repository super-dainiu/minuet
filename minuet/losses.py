"""Production Minuet objectives used only while fitting the encoder."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PAIR_WEIGHT = 1.9435173526733809
PAIR_TEMPERATURE = 0.08808674812619335
PRIVATE_KL_WEIGHT = 7.231904795762552e-5
PAIRED_TARGET_RANK = 32
PAIRED_TARGET_WEIGHT = 1.5841466269183442
DONOR_ASSOCIATION_WEIGHT = 0.11343234492276047
COVARIANCE_AGREEMENT_WEIGHT = 0.3423206519502428
COVARIANCE_START_FRACTION = 0.789253759592594


def donor_centered_exact_pair_infonce(
    rna: Tensor,
    atac: Tensor,
    donor: Tensor,
    *,
    temperature: float = PAIR_TEMPERATURE,
    eps: float = 1.0e-6,
) -> Tensor:
    """Symmetric exact-pair InfoNCE within separately centered donor blocks."""
    if rna.ndim != 2 or atac.shape != rna.shape or donor.shape != (rna.shape[0],):
        raise ValueError("paired embeddings and donor labels have incompatible shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    terms: list[Tensor] = []
    for value in torch.unique(donor[donor >= 0]):
        index = torch.nonzero(donor == value, as_tuple=False).flatten()
        if index.numel() < 2:
            continue
        x = F.normalize(
            rna[index] - rna[index].mean(dim=0, keepdim=True), dim=-1, eps=eps
        )
        y = F.normalize(
            atac[index] - atac[index].mean(dim=0, keepdim=True), dim=-1, eps=eps
        )
        logits = x @ y.T / temperature
        labels = torch.arange(index.numel(), device=rna.device)
        terms.append(
            0.5
            * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.T, labels)
            )
        )
    return torch.stack(terms).mean() if terms else (rna.sum() + atac.sum()) * 0.0


def donor_centered_smooth_l1(
    prediction: Tensor, target: Tensor, donor: Tensor
) -> Tensor:
    """Within-donor centered Smooth-L1 with equal donor weighting."""
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same 2D shape")
    if donor.shape != (prediction.shape[0],):
        raise ValueError("donor must have one value per prediction")
    terms: list[Tensor] = []
    for value in torch.unique(donor[donor >= 0]):
        block = donor == value
        if int(block.sum()) < 2:
            continue
        pred = prediction[block] - prediction[block].mean(dim=0, keepdim=True)
        truth = target[block] - target[block].mean(dim=0, keepdim=True)
        terms.append(F.smooth_l1_loss(pred, truth))
    return torch.stack(terms).mean() if terms else prediction.sum() * 0.0


def context_centered_donor_association(
    z: Tensor, donor: Tensor, biology_context: Tensor, *, eps: float = 1.0e-12
) -> Tensor:
    """Normalized linear donor association computed separately within contexts."""
    labelled = (donor >= 0) & (biology_context >= 0)
    if not labelled.any():
        return z.sum() * 0.0
    z, donor, biology_context = z[labelled], donor[labelled], biology_context[labelled]
    n_donors = int(donor.max().detach()) + 1
    terms: list[Tensor] = []
    for value in torch.unique(biology_context):
        index = torch.nonzero(biology_context == value, as_tuple=False).flatten()
        if index.numel() < 4 or torch.unique(donor[index]).numel() < 2:
            continue
        x = z[index] - z[index].mean(dim=0, keepdim=True)
        d = F.one_hot(donor[index], n_donors).to(z.dtype)
        d = d - d.mean(dim=0, keepdim=True)
        numerator = (x.T @ d).square().sum()
        denominator = (
            (x.T @ x).square().sum().sqrt()
            * (d.T @ d).square().sum().sqrt()
        ).detach()
        terms.append(numerator / denominator.clamp_min(eps))
    return torch.stack(terms).mean() if terms else z.sum() * 0.0


def within_context_covariance_agreement(
    z: Tensor,
    donor: Tensor,
    biology_context: Tensor,
    *,
    eps: float = 1.0e-8,
) -> Tensor:
    """Match trace-normalized donor covariances within each context."""
    labelled = (donor >= 0) & (biology_context >= 0)
    z, donor, biology_context = (
        z[labelled], donor[labelled], biology_context[labelled]
    )
    context_terms: list[Tensor] = []
    for context_value in torch.unique(biology_context):
        in_context = biology_context == context_value
        covariances: list[Tensor] = []
        for donor_value in torch.unique(donor[in_context]):
            block = in_context & (donor == donor_value)
            n_cells = int(block.sum())
            if n_cells < 2:
                continue
            centered = z[block] - z[block].mean(dim=0, keepdim=True)
            covariance = centered.T @ centered / float(n_cells - 1)
            covariance = covariance / covariance.diagonal().sum().clamp_min(eps)
            covariances.append(covariance)
        if len(covariances) < 2:
            continue
        stacked = torch.stack(covariances)
        reference = stacked.mean(dim=0)
        context_terms.append(
            (stacked - reference).square().sum(dim=(-2, -1)).mean()
        )
    return torch.stack(context_terms).mean() if context_terms else z.sum() * 0.0


def gaussian_kl(mu: Tensor, logvar: Tensor) -> Tensor:
    return 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(dim=-1).mean()


def negative_binomial_nll(
    counts: Tensor, mean: Tensor, inverse_dispersion: Tensor, *, eps: float = 1.0e-8
) -> Tensor:
    if counts.shape != mean.shape:
        raise ValueError("counts and NB mean must have identical shapes")
    theta = inverse_dispersion.clamp_min(eps).unsqueeze(0)
    mean = mean.clamp_min(eps)
    log_prob = (
        torch.lgamma(counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(counts + 1.0)
        + theta * (torch.log(theta + eps) - torch.log(theta + mean))
        + counts * (torch.log(mean) - torch.log(theta + mean))
    )
    return -log_prob


class MinuetLosses(nn.Module):
    """Trial45 objective, including its bias-free fitting-only target head."""

    def __init__(self, shared_dim: int, target_rank: int) -> None:
        super().__init__()
        if shared_dim <= 0 or target_rank <= 0:
            raise ValueError("shared_dim and target_rank must be positive")
        self.shared_paired_target_head = nn.Linear(
            shared_dim, 2 * target_rank, bias=False
        )

    def forward(
        self,
        rna_counts: Tensor,
        atac_counts: Tensor,
        output: dict[str, Tensor],
        *,
        donor: Tensor,
        biology_context: Tensor,
        rna_target_score: Tensor,
        atac_target_score: Tensor,
        covariance_active: bool,
    ) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}
        losses["rna_reconstruction"] = negative_binomial_nll(
            rna_counts, output["rna_rate"], output["rna_inverse_dispersion"]
        ).mean()
        losses["atac_reconstruction"] = F.binary_cross_entropy_with_logits(
            output["atac_logits"], (atac_counts > 0).to(output["atac_logits"].dtype)
        )
        losses["private_kl"] = gaussian_kl(
            output["rna_private"], output["rna_private_logvar"]
        ) + gaussian_kl(output["atac_private"], output["atac_private_logvar"])
        losses["pair"] = donor_centered_exact_pair_infonce(
            output["rna_shared"], output["atac_shared"], donor
        )
        paired_target = torch.cat((rna_target_score, atac_target_score), dim=-1)
        losses["prediction"] = 0.5 * (
            donor_centered_smooth_l1(
                self.shared_paired_target_head(output["rna_shared"]),
                paired_target,
                donor,
            )
            + donor_centered_smooth_l1(
                self.shared_paired_target_head(output["atac_shared"]),
                paired_target,
                donor,
            )
        )
        joint = 0.5 * (output["rna_shared"] + output["atac_shared"])
        losses["donor_association"] = context_centered_donor_association(
            joint, donor, biology_context
        )
        losses["covariance_agreement"] = within_context_covariance_agreement(
            joint, donor, biology_context
        )
        losses["total"] = (
            losses["rna_reconstruction"]
            + losses["atac_reconstruction"]
            + PRIVATE_KL_WEIGHT * losses["private_kl"]
            + PAIR_WEIGHT * losses["pair"]
            + PAIRED_TARGET_WEIGHT * losses["prediction"]
            + DONOR_ASSOCIATION_WEIGHT * losses["donor_association"]
        )
        if covariance_active:
            losses["total"] = (
                losses["total"]
                + COVARIANCE_AGREEMENT_WEIGHT * losses["covariance_agreement"]
            )
        return losses
