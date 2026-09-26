"""PyTorch module of Minuet: encoders, decoders, and the training objective."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Encoder(nn.Module):
    """Residual MLP encoder of one assay.

    Maps counts to a deterministic shared code and to the mean and log variance
    of a Gaussian private variable.

    Parameters
    ----------
    n_input
        Number of input features.
    n_hidden
        Width of the residual trunk.
    n_latent
        Dimensionality of the shared code.
    n_private
        Dimensionality of the private variable.
    n_layers
        Number of residual blocks.
    dropout_rate
        Dropout rate of every block.
    norm
        ``"batch"`` applies BatchNorm after the input layer and to the trunk
        output. ``"layer"`` applies LayerNorm before every residual block and
        to the trunk output, so no reference batch statistics are used at
        inference.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_latent: int,
        n_private: int,
        n_layers: int,
        dropout_rate: float,
        norm: Literal["batch", "layer"],
    ):
        super().__init__()
        if norm not in ("batch", "layer"):
            raise ValueError("norm must be 'batch' or 'layer'.")
        batch_norm = norm == "batch"
        self.input = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            *([nn.BatchNorm1d(n_hidden)] if batch_norm else []),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                *([] if batch_norm else [nn.LayerNorm(n_hidden)]),
                nn.Linear(n_hidden, n_hidden),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
            for _ in range(n_layers)
        )
        self.output_norm = nn.BatchNorm1d(n_hidden) if batch_norm else nn.LayerNorm(n_hidden)
        self.shared = nn.Linear(n_hidden, n_latent)
        self.private_mean = nn.Linear(n_hidden, n_private)
        self.private_logvar = nn.Linear(n_hidden, n_private)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Shared code, private mean, and private log variance."""
        h = self.input(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.output_norm(h)
        return self.shared(h), self.private_mean(h), self.private_logvar(h).clamp(-8.0, 8.0)


class MinuetModule(nn.Module):
    """Minuet encoders, private decoders, and training objective.

    Parameters
    ----------
    n_genes
        Number of RNA features.
    n_peaks
        Number of ATAC features.
    n_hidden
        Width of the encoder trunks.
    n_latent
        Dimensionality of the shared representation.
    n_private
        Dimensionality of the private variable of each assay.
    n_layers
        Number of residual blocks per encoder.
    n_decoder_hidden
        Width of the private decoders.
    dropout_rate
        Dropout rate of encoders and decoders.
    norm
        Normalization of the encoder trunks, ``"batch"`` or ``"layer"``.
    prediction_rank
        Rank of each assay's block of the prediction target.
    """

    def __init__(
        self,
        n_genes: int,
        n_peaks: int,
        n_hidden: int = 224,
        n_latent: int = 64,
        n_private: int = 16,
        n_layers: int = 3,
        n_decoder_hidden: int = 64,
        dropout_rate: float = 0.2,
        norm: Literal["batch", "layer"] = "batch",
        prediction_rank: int = 32,
    ):
        super().__init__()
        self.rna_encoder = Encoder(
            n_genes, n_hidden, n_latent, n_private, n_layers, dropout_rate, norm
        )
        self.atac_encoder = Encoder(
            n_peaks, n_hidden, n_latent, n_private, n_layers, dropout_rate, norm
        )
        self.rna_decoder = _decoder(n_private, n_decoder_hidden, n_genes, dropout_rate)
        self.atac_decoder = _decoder(n_private, n_decoder_hidden, n_peaks, dropout_rate)
        self.rna_log_theta = nn.Parameter(torch.zeros(n_genes))
        self.prediction_head = nn.Linear(n_latent, 2 * prediction_rank, bias=False)

    def forward(self, rna: Tensor, atac: Tensor) -> tuple[Tensor, Tensor]:
        """Shared representations of the RNA and ATAC profiles of paired cells."""
        return self.rna_encoder(rna)[0], self.atac_encoder(atac)[0]

    def loss(
        self,
        rna: Tensor,
        atac: Tensor,
        target: Tensor,
        donor: Tensor,
        group: Tensor,
        *,
        temperature: float,
        kl_weight: float,
        pair_weight: float,
        prediction_weight: float,
        donor_weight: float,
        covariance_weight: float,
    ) -> Tensor:
        """Minuet objective on one batch.

        Parameters
        ----------
        rna, atac
            Counts of the paired cells.
        target
            Prediction target of the cells.
        donor, group
            Integer donor and covariate-group codes of the cells.
        temperature
            Temperature of the within-donor contrastive loss.
        kl_weight, pair_weight, prediction_weight, donor_weight, covariance_weight
            Weights of the private KL, pair, prediction, donor-association, and
            covariance-alignment terms.
        """
        z_rna, mean_rna, logvar_rna = self.rna_encoder(rna)
        z_atac, mean_atac, logvar_atac = self.atac_encoder(atac)
        v_rna = mean_rna + torch.randn_like(mean_rna) * torch.exp(0.5 * logvar_rna)
        v_atac = mean_atac + torch.randn_like(mean_atac) * torch.exp(0.5 * logvar_atac)
        rate = F.softmax(self.rna_decoder(v_rna), dim=-1) * rna.sum(-1, keepdim=True).clamp_min(1.0)
        theta = F.softplus(self.rna_log_theta).clamp_min(1e-4)
        reconstruction = _negative_binomial_nll(rna, rate, theta) + F.binary_cross_entropy_with_logits(
            self.atac_decoder(v_atac), (atac > 0).float()
        )
        kl = _gaussian_kl(mean_rna, logvar_rna) + _gaussian_kl(mean_atac, logvar_atac)
        pair = _within_donor_infonce(z_rna, z_atac, donor, temperature)
        prediction = 0.5 * (
            _donor_centered_smooth_l1(self.prediction_head(z_rna), target, donor)
            + _donor_centered_smooth_l1(self.prediction_head(z_atac), target, donor)
        )
        joint = 0.5 * (z_rna + z_atac)
        loss = (
            reconstruction
            + kl_weight * kl
            + pair_weight * pair
            + prediction_weight * prediction
            + donor_weight * _donor_association(joint, donor, group)
        )
        if covariance_weight:
            loss = loss + covariance_weight * _covariance_alignment(joint, donor, group)
        return loss


def _decoder(n_input: int, n_hidden: int, n_output: int, dropout_rate: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_input, n_hidden), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(n_hidden, n_output)
    )


def _negative_binomial_nll(x: Tensor, mu: Tensor, theta: Tensor, eps: float = 1e-8) -> Tensor:
    mu = mu.clamp_min(eps)
    log_prob = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - torch.log(theta + mu))
        + x * (torch.log(mu) - torch.log(theta + mu))
    )
    return -log_prob.mean()


def _gaussian_kl(mean: Tensor, logvar: Tensor) -> Tensor:
    return 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).sum(-1).mean()


def _within_donor_infonce(z_rna: Tensor, z_atac: Tensor, donor: Tensor, temperature: float) -> Tensor:
    """Symmetric InfoNCE with negatives from the anchor's donor, after donor centering."""
    terms = []
    for d in torch.unique(donor):
        cells = donor == d
        r = F.normalize(z_rna[cells] - z_rna[cells].mean(0, keepdim=True), dim=-1, eps=1e-6)
        a = F.normalize(z_atac[cells] - z_atac[cells].mean(0, keepdim=True), dim=-1, eps=1e-6)
        logits = r @ a.T / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        terms.append(0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)))
    return torch.stack(terms).mean()


def _donor_centered_smooth_l1(prediction: Tensor, target: Tensor, donor: Tensor) -> Tensor:
    """SmoothL1 between predictions and targets, both centered within each donor."""
    terms = []
    for d in torch.unique(donor):
        cells = donor == d
        p = prediction[cells] - prediction[cells].mean(0, keepdim=True)
        t = target[cells] - target[cells].mean(0, keepdim=True)
        terms.append(F.smooth_l1_loss(p, t))
    return torch.stack(terms).mean()


def _donor_association(z: Tensor, donor: Tensor, group: Tensor) -> Tensor:
    """Linear CKA between the representation and donor identity within each group.

    The normalizer is detached, so the penalty moves donor means rather than
    the scale of the representation.
    """
    n_donors = int(donor.max()) + 1
    terms = []
    for g in torch.unique(group):
        cells = group == g
        z_c = z[cells] - z[cells].mean(0, keepdim=True)
        d_c = F.one_hot(donor[cells], num_classes=n_donors).to(z.dtype)
        d_c = d_c - d_c.mean(0, keepdim=True)
        numerator = (z_c.T @ d_c).square().sum()
        z_energy = (z_c.T @ z_c).square().sum().sqrt()
        d_energy = (d_c.T @ d_c).square().sum().sqrt()
        terms.append(numerator / (z_energy * d_energy).detach().clamp_min(1e-12))
    return torch.stack(terms).mean()


def _covariance_alignment(z: Tensor, donor: Tensor, group: Tensor) -> Tensor:
    """Distance of trace-normalized donor covariances to their mean within each group."""
    terms = []
    for g in torch.unique(group):
        in_group = group == g
        covariances = []
        for d in torch.unique(donor[in_group]):
            cells = in_group & (donor == d)
            x = z[cells] - z[cells].mean(0, keepdim=True)
            covariance = x.T @ x / (x.shape[0] - 1)
            covariances.append(covariance / covariance.diagonal().sum().clamp_min(1e-8))
        stacked = torch.stack(covariances)
        terms.append((stacked - stacked.mean(0)).square().sum((-2, -1)).mean())
    return torch.stack(terms).mean()
