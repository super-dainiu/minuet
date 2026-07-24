from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _context_rms_standardize(z: Tensor, group: Tensor, eps: float = 1.0e-6) -> Tensor:
    standardized = torch.zeros_like(z)
    for value in torch.unique(group):
        idx = torch.nonzero(group == value, as_tuple=False).squeeze(-1)
        centered = z[idx] - z[idx].mean(dim=0, keepdim=True)
        rms = centered.square().mean(dim=0, keepdim=True).clamp_min(eps * eps).sqrt()
        standardized = standardized.index_copy(0, idx, centered / rms)
    return standardized


def _within_group_exact_pair(
    rna: Tensor,
    atac: Tensor,
    donor: Tensor,
    context: Tensor,
    *,
    temperature: float,
    standardize: bool,
) -> Tensor:
    labelled = (donor >= 0) & (context >= 0)
    rna = rna[labelled]
    atac = atac[labelled]
    donor = donor[labelled]
    context = context[labelled]
    if rna.shape[0] == 0:
        return rna.sum() + atac.sum()
    if standardize:
        pairs = torch.stack((donor, context), dim=1)
        _, group = torch.unique(pairs, dim=0, return_inverse=True)
        rna = _context_rms_standardize(rna, group)
        atac = _context_rms_standardize(atac, group)
    context_terms = []
    for context_value in torch.unique(context):
        donor_terms = []
        for donor_value in torch.unique(donor[context == context_value]):
            idx = torch.nonzero(
                (context == context_value) & (donor == donor_value),
                as_tuple=False,
            ).squeeze(-1)
            if idx.numel() < 2:
                continue
            rna_group = F.normalize(rna[idx], dim=-1)
            atac_group = F.normalize(atac[idx], dim=-1)
            logits = (rna_group @ atac_group.T) / temperature
            labels = torch.arange(idx.numel(), device=logits.device)
            donor_terms.append(
                0.5
                * (
                    F.cross_entropy(logits, labels)
                    + F.cross_entropy(logits.T, labels)
                )
            )
        if donor_terms:
            context_terms.append(torch.stack(donor_terms).mean())
    if not context_terms:
        return rna.sum() * 0.0 + atac.sum() * 0.0
    return torch.stack(context_terms).mean()


class MinuetLosses:
    @staticmethod
    def reconstruction(x: Tensor, x_hat: Tensor) -> Tensor:
        return F.mse_loss(x_hat, x)

    @staticmethod
    def contrastive(x_shared: Tensor, y_shared: Tensor, temperature: float = 0.07,
                    fn_sim: float = 0.0, alignment_mode: str = "legacy_fnc",
                    donor: Tensor | None = None, context: Tensor | None = None) -> Tensor:
        """Symmetric cross-modal InfoNCE with label-free false-negative cancellation.

        A plain contrastive treats every other cell in the batch as a negative,
        including cells of the same biological state as the anchor. Those false
        negatives dominate the gradient when cell types are few and collapse
        cross-modal retrieval. When ``fn_sim`` > 0 we drop, from each anchor's
        negatives, the cells that are similar to it in BOTH modalities' shared codes
        (cosine > ``fn_sim``): likely same-state pairs. This uses only the embeddings,
        so it needs no cell-type labels, and the absolute-similarity gate self-adapts
        -- diverse cohorts trip it rarely, low-diversity cohorts often.
        """
        if alignment_mode in {"within_group_exact_pair", "within_group_raw_exact_pair"}:
            if donor is None or context is None:
                raise ValueError(f"{alignment_mode} requires donor and context labels")
            return _within_group_exact_pair(
                x_shared,
                y_shared,
                donor,
                context,
                temperature=temperature,
                standardize=alignment_mode == "within_group_exact_pair",
            )
        if alignment_mode not in {"legacy_fnc", "global_exact_pair"}:
            raise ValueError(f"unknown alignment_mode: {alignment_mode!r}")
        x = F.normalize(x_shared, dim=-1)
        y = F.normalize(y_shared, dim=-1)
        logits = (x @ y.T) / temperature
        n = x.shape[0]
        labels = torch.arange(n, device=x.device)
        if alignment_mode == "legacy_fnc" and fn_sim > 0.0:
            with torch.no_grad():
                eye = torch.eye(n, dtype=torch.bool, device=x.device)
                false_neg = (x @ x.T > fn_sim) & (y @ y.T > fn_sim) & ~eye
            logits = logits.masked_fill(false_neg, -1e9)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    @staticmethod
    def alignment(x_shared: Tensor, y_shared: Tensor) -> Tensor:
        x_norm = F.normalize(x_shared, dim=-1)
        y_norm = F.normalize(y_shared, dim=-1)
        return 1.0 - (x_norm * y_norm).sum(dim=-1).mean()

    @staticmethod
    def gaussian_kl(mu: Tensor, logvar: Tensor) -> Tensor:
        return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1).mean()

    @staticmethod
    def symmetric_gaussian_kl(mu_p: Tensor, logvar_p: Tensor, mu_q: Tensor, logvar_q: Tensor) -> Tensor:
        var_p = logvar_p.exp()
        var_q = logvar_q.exp()
        kl_pq = 0.5 * ((logvar_q - logvar_p) + (var_p + (mu_p - mu_q).pow(2)) / (var_q + 1.0e-8) - 1.0)
        kl_qp = 0.5 * ((logvar_p - logvar_q) + (var_q + (mu_q - mu_p).pow(2)) / (var_p + 1.0e-8) - 1.0)
        return 0.5 * (kl_pq.sum(dim=-1).mean() + kl_qp.sum(dim=-1).mean())

    @staticmethod
    def cross_covariance(x: Tensor, y: Tensor) -> Tensor:
        # Guard the empty-latent case (e.g. context_dim=0 / no context tokens):
        # the cross-covariance of a zero-width latent is vacuously 0, but the
        # naive code would compute mean() of an empty tensor (NaN) which then
        # poisons the weighted sum even at weight 0 (0 * NaN = NaN).
        if x.numel() == 0 or y.numel() == 0 or x.shape[-1] == 0 or y.shape[-1] == 0:
            return x.new_zeros(())
        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)
        x_scaled = x_centered / (x_centered.std(dim=0, keepdim=True, unbiased=False) + 1.0e-4)
        y_scaled = y_centered / (y_centered.std(dim=0, keepdim=True, unbiased=False) + 1.0e-4)
        cov = x_scaled.T @ y_scaled / max(x.shape[0], 1)
        return cov.pow(2).mean()

    @classmethod
    def total(
        cls,
        rna_x: Tensor | None,
        atac_x: Tensor | None,
        out: dict[str, Tensor],
        recon_weight: float = 1.0,
        alignment_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        shared_recon_weight: float = 0.0,
        self_shared_recon_weight: float = 0.0,
        kl_shared_weight: float = 1.0e-3,
        kl_private_weight: float = 1.0e-3,
        fusion_weight: float = 1.0e-3,
        decouple_weight: float = 1.0e-3,
        adv_weight: float = 0.0,
        nuisance_pred_weight: float = 0.0,
        temperature: float = 0.07,
        fn_sim: float = 0.0,
        alignment_mode: str = "legacy_fnc",
        alignment_donor: Tensor | None = None,
        alignment_context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}

        if rna_x is not None:
            losses["rna_recon"] = cls.reconstruction(rna_x, out["rna_recon"])
            losses["rna_private_kl"] = cls.gaussian_kl(out["rna_private_mu"], out["rna_private_logvar"])
            losses["rna_cell_kl"] = cls.gaussian_kl(out["rna_cell_mu"], out["rna_cell_logvar"])
            losses["rna_context_kl"] = cls.gaussian_kl(out["rna_context_mu"], out["rna_context_logvar"])
            losses["rna_cell_context_decouple"] = cls.cross_covariance(out["rna_cell_mu"], out["rna_context_mu"])
            # IndiSeek-style cross-modal disentanglement: the RNA-private factor should
            # be independent of the OTHER modality's shared factor (C_atac), not its own
            # shared factor -- the within-modality version is lossy when shared is redundant.
            losses["rna_shared_private_decouple"] = cls.cross_covariance(out["atac_shared"], out["rna_private_mu"])
            if "rna_shared_self_recon" in out:
                losses["rna_shared_self_recon"] = cls.reconstruction(rna_x, out["rna_shared_self_recon"])

        if atac_x is not None:
            losses["atac_recon"] = cls.reconstruction(atac_x, out["atac_recon"])
            losses["atac_private_kl"] = cls.gaussian_kl(out["atac_private_mu"], out["atac_private_logvar"])
            losses["atac_cell_kl"] = cls.gaussian_kl(out["atac_cell_mu"], out["atac_cell_logvar"])
            losses["atac_context_kl"] = cls.gaussian_kl(out["atac_context_mu"], out["atac_context_logvar"])
            losses["atac_cell_context_decouple"] = cls.cross_covariance(out["atac_cell_mu"], out["atac_context_mu"])
            # IndiSeek cross-modal: ATAC-private independent of the RNA shared factor.
            losses["atac_shared_private_decouple"] = cls.cross_covariance(out["rna_shared"], out["atac_private_mu"])
            if "atac_shared_self_recon" in out:
                losses["atac_shared_self_recon"] = cls.reconstruction(atac_x, out["atac_shared_self_recon"])

        if rna_x is not None and atac_x is not None:
            losses["alignment"] = cls.alignment(out["rna_shared"], out["atac_shared"])
            losses["contrastive"] = cls.contrastive(
                out["rna_shared"],
                out["atac_shared"],
                temperature=temperature,
                fn_sim=fn_sim,
                alignment_mode=alignment_mode,
                donor=alignment_donor,
                context=alignment_context,
            )
            losses["joint_cell_kl"] = cls.gaussian_kl(out["joint_cell_mu"], out["joint_cell_logvar"])
            losses["joint_context_kl"] = cls.gaussian_kl(out["joint_context_mu"], out["joint_context_logvar"])
            losses["cell_fusion_consistency"] = 0.5 * (
                cls.symmetric_gaussian_kl(
                    out["rna_cell_mu"],
                    out["rna_cell_logvar"],
                    out["joint_cell_mu"],
                    out["joint_cell_logvar"],
                )
                + cls.symmetric_gaussian_kl(
                    out["atac_cell_mu"],
                    out["atac_cell_logvar"],
                    out["joint_cell_mu"],
                    out["joint_cell_logvar"],
                )
            )
            losses["context_fusion_consistency"] = 0.5 * (
                cls.symmetric_gaussian_kl(
                    out["rna_context_mu"],
                    out["rna_context_logvar"],
                    out["joint_context_mu"],
                    out["joint_context_logvar"],
                )
                + cls.symmetric_gaussian_kl(
                    out["atac_context_mu"],
                    out["atac_context_logvar"],
                    out["joint_context_mu"],
                    out["joint_context_logvar"],
                )
            )
            losses["joint_cell_context_decouple"] = cls.cross_covariance(out["joint_cell_mu"], out["joint_context_mu"])
            if "rna_shared_recon" in out:
                losses["rna_shared_recon"] = cls.reconstruction(rna_x, out["rna_shared_recon"])
            if "atac_shared_recon" in out:
                losses["atac_shared_recon"] = cls.reconstruction(atac_x, out["atac_shared_recon"])

        # Adversarial batch classifier loss (cross-entropy on batch_adv_logits).
        # Paired with gradient reversal in the model forward; training this loss
        # pushes the shared latent to be batch-invariant (DANN, Ganin et al. 2015).
        if adv_weight > 0 and "batch_adv_logits" in out and "batch_adv_target" in out:
            losses["batch_adv"] = F.cross_entropy(out["batch_adv_logits"], out["batch_adv_target"])

        # Structured-latent nuisance head loss (cross-entropy on context_mu,
        # NO gradient reversal). Pushes context to be batch-predictive so it
        # absorbs nuisance; combined with cell_context_decouple this routes
        # batch information out of cell (= z_bio). Option A in the paper.
        if nuisance_pred_weight > 0 and "nuisance_pred_logits" in out and "nuisance_pred_target" in out:
            losses["nuisance_pred"] = F.cross_entropy(out["nuisance_pred_logits"], out["nuisance_pred_target"])

        total = torch.zeros((), device=next(iter(out.values())).device)
        for name, value in losses.items():
            if name in {"rna_recon", "atac_recon"}:
                total = total + recon_weight * value
            elif name in {"rna_shared_recon", "atac_shared_recon"}:
                total = total + shared_recon_weight * value
            elif name in {"rna_shared_self_recon", "atac_shared_self_recon"}:
                total = total + self_shared_recon_weight * value
            elif name == "alignment":
                total = total + alignment_weight * value
            elif name == "contrastive":
                total = total + contrastive_weight * value
            elif name == "batch_adv":
                total = total + adv_weight * value
            elif name == "nuisance_pred":
                total = total + nuisance_pred_weight * value
            elif "fusion_consistency" in name:
                total = total + fusion_weight * value
            elif "decouple" in name:
                total = total + decouple_weight * value
            elif "private_kl" in name:
                total = total + kl_private_weight * value
            else:
                total = total + kl_shared_weight * value

        losses["total"] = total
        return losses
