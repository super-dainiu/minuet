from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


class MinuetLosses:
    @staticmethod
    def reconstruction(x: Tensor, x_hat: Tensor) -> Tensor:
        return F.mse_loss(x_hat, x)

    @staticmethod
    def contrastive(x_shared: Tensor, y_shared: Tensor, temperature: float = 0.07) -> Tensor:
        x_norm = F.normalize(x_shared, dim=-1)
        y_norm = F.normalize(y_shared, dim=-1)
        logits = (x_norm @ y_norm.T) / temperature
        labels = torch.arange(x_shared.shape[0], device=x_shared.device)
        loss_xy = F.cross_entropy(logits, labels)
        loss_yx = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_xy + loss_yx)

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
        temperature: float = 0.07,
    ) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}

        if rna_x is not None:
            losses["rna_recon"] = cls.reconstruction(rna_x, out["rna_recon"])
            losses["rna_private_kl"] = cls.gaussian_kl(out["rna_private_mu"], out["rna_private_logvar"])
            losses["rna_cell_kl"] = cls.gaussian_kl(out["rna_cell_mu"], out["rna_cell_logvar"])
            losses["rna_context_kl"] = cls.gaussian_kl(out["rna_context_mu"], out["rna_context_logvar"])
            losses["rna_cell_context_decouple"] = cls.cross_covariance(out["rna_cell_mu"], out["rna_context_mu"])
            losses["rna_shared_private_decouple"] = cls.cross_covariance(out["rna_shared"], out["rna_private_mu"])
            if "rna_shared_self_recon" in out:
                losses["rna_shared_self_recon"] = cls.reconstruction(rna_x, out["rna_shared_self_recon"])

        if atac_x is not None:
            losses["atac_recon"] = cls.reconstruction(atac_x, out["atac_recon"])
            losses["atac_private_kl"] = cls.gaussian_kl(out["atac_private_mu"], out["atac_private_logvar"])
            losses["atac_cell_kl"] = cls.gaussian_kl(out["atac_cell_mu"], out["atac_cell_logvar"])
            losses["atac_context_kl"] = cls.gaussian_kl(out["atac_context_mu"], out["atac_context_logvar"])
            losses["atac_cell_context_decouple"] = cls.cross_covariance(out["atac_cell_mu"], out["atac_context_mu"])
            losses["atac_shared_private_decouple"] = cls.cross_covariance(out["atac_shared"], out["atac_private_mu"])
            if "atac_shared_self_recon" in out:
                losses["atac_shared_self_recon"] = cls.reconstruction(atac_x, out["atac_shared_self_recon"])

        if rna_x is not None and atac_x is not None:
            losses["alignment"] = cls.alignment(out["rna_shared"], out["atac_shared"])
            losses["contrastive"] = cls.contrastive(out["rna_shared"], out["atac_shared"], temperature=temperature)
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
