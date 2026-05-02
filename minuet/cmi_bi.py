"""Conditional Mutual Information Batch-Invariance (CMI-BI).

Implements the training-time alternative to post-hoc Harmony described
in paper Appendix C. The objective is to minimize I(z; b | y) — the
mutual information between the shared latent z and the batch label b
*within each cell type y* — while preserving I(z; y) via a supervised
contrastive loss. Targets the "batch-invariance where it does not
fight biology" quantity that marginal DANN fails to optimize on brain
atlases (where cell types are donor-enriched).

Three pieces:
  - ConditionalCLUBAux: auxiliary q_psi(b | z, y) classifier used to
    compute a CLUB upper bound on I(z; b | y).
  - conditional_club_upper_bound(): the upper bound itself, differentiable
    w.r.t. z when the aux network parameters are held fixed.
  - supervised_contrastive_loss(): sup-con (Khosla 2020) on the shared
    latent with label y as the positive-selection key.

Training proceeds bilevel-style: at each step, first update the aux
classifier on detached z (inner loop, standard cross-entropy), then
compute the CLUB upper bound with the aux params held fixed and
backprop into the main model.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConditionalCLUBAux(nn.Module):
    """Auxiliary variational network q_psi(b | z, y).

    Takes the shared latent z concatenated with a one-hot encoding of y
    and predicts logits over the n_batches batch vocabulary. Trained
    by standard cross-entropy on real (z, b, y) triples to approximate
    the true p(b | z, y); when plugged into the CLUB upper-bound formula
    it gives a differentiable upper bound on I(z; b | y).
    """

    def __init__(
        self,
        latent_dim: int,
        n_classes: int,
        n_batches: int,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.n_batches = int(n_batches)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + n_classes, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_batches),
        )

    def forward(self, z: Tensor, y: Tensor) -> Tensor:
        y_onehot = F.one_hot(y.long(), num_classes=self.n_classes).float()
        return self.classifier(torch.cat([z, y_onehot], dim=-1))


def aux_cross_entropy(
    aux: ConditionalCLUBAux,
    z: Tensor,
    b: Tensor,
    y: Tensor,
) -> Tensor:
    """Inner-loop loss to train the aux net.

    This expects z to be detached from the main model's graph so only
    the aux params receive gradient.
    """
    logits = aux(z, y)
    return F.cross_entropy(logits, b.long())


def conditional_club_upper_bound(
    aux: ConditionalCLUBAux,
    z: Tensor,
    b: Tensor,
    y: Tensor,
) -> Tensor:
    """CLUB-style upper bound on I(z; b | y).

    CLUB (Cheng et al. 2020) bound: the MI is upper-bounded by
      E_p(z,b|y)[log q(b|z,y)] − E_p(z|y) p(b|y)[log q(b|z,y)],
    where the second expectation is computed by shuffling (z, b) pairs
    within each class y. We realize that by a class-wise permutation
    on the minibatch.
    """
    # First term: positive log-prob on true pairs.
    logits_pos = aux(z, y)
    log_q_pos = F.log_softmax(logits_pos, dim=-1).gather(
        -1, b.long().unsqueeze(-1)
    ).squeeze(-1)

    # Second term: class-wise shuffle. For each cell i, pair z_i with a
    # randomly chosen b_j from another cell that shares y_i. If a class
    # has only one member in the minibatch, fall back to the true b_i
    # (bound degenerates to zero contribution for that cell).
    shuffled_b = _class_wise_shuffle(b, y)
    log_q_neg = F.log_softmax(logits_pos, dim=-1).gather(
        -1, shuffled_b.long().unsqueeze(-1)
    ).squeeze(-1)

    # CLUB bound, averaged over the minibatch.
    return (log_q_pos - log_q_neg).mean()


def _class_wise_shuffle(b: Tensor, y: Tensor) -> Tensor:
    """For each cell i, return b_j for a random j with y_j = y_i (j != i
    when possible). Implements the p(b | y) marginal in the CLUB bound.
    """
    out = b.clone()
    unique_y = torch.unique(y)
    for cls in unique_y.tolist():
        mask = y == cls
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idx.numel() <= 1:
            continue
        perm = idx[torch.randperm(idx.numel(), device=b.device)]
        # If any element happens to permute to itself, roll once so the
        # shuffle excludes the self-pair for at least one position.
        if (perm == idx).any():
            perm = torch.roll(perm, shifts=1, dims=0)
        out[idx] = b[perm]
    return out


def supervised_contrastive_loss(
    z: Tensor,
    y: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """Supervised contrastive loss (Khosla et al. 2020).

    For each anchor i, positives P(i) = { j != i : y_j = y_i } and the
    loss per anchor is the -log mean-exp similarity over positives
    normalized by all-pairs similarity. Anchors whose positive set is
    empty contribute zero. Lower bounds I(z; y) in the limit of large
    batch; preserves class-separability in the shared latent.
    """
    z_norm = F.normalize(z, dim=-1)
    sim = (z_norm @ z_norm.t()) / temperature              # (N, N)

    n = z.size(0)
    # Self-mask for the denominator log-sum-exp.
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim_masked = sim.masked_fill(self_mask, float("-inf"))
    log_denom = torch.logsumexp(sim_masked, dim=-1)        # (N,)

    # Positive mask (same y, excluding self).
    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & ~self_mask
    # For numerical stability, gather log p for each positive pair.
    log_p = sim - log_denom.unsqueeze(-1)                  # (N, N)

    # Average log-probability over positives per anchor.
    num_pos = pos_mask.float().sum(dim=-1).clamp(min=1.0)
    log_prob_pos = (log_p * pos_mask.float()).sum(dim=-1) / num_pos

    # Loss: -mean(log prob) over anchors that have >= 1 positive.
    has_positive = pos_mask.any(dim=-1)
    if not has_positive.any():
        return z.new_zeros(())
    return -log_prob_pos[has_positive].mean()


def cmi_bi_total(
    aux: ConditionalCLUBAux,
    z: Tensor,
    b: Tensor,
    y: Tensor,
    *,
    club_weight: float = 1.0,
    supcon_weight: float = 0.1,
    temperature: float = 0.07,
) -> dict[str, Tensor]:
    """Combined CMI-BI loss.

    Returns a dict with 'cmi_bi_club', 'cmi_bi_supcon', and
    'cmi_bi_total' scalar tensors.
    """
    club = conditional_club_upper_bound(aux, z, b, y)
    supcon = supervised_contrastive_loss(z, y, temperature=temperature)
    total = club_weight * club + supcon_weight * supcon
    return {
        "cmi_bi_club": club,
        "cmi_bi_supcon": supcon,
        "cmi_bi_total": total,
    }
