from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MinuetConfig:
    rna_dim: int
    atac_dim: int
    group_size: int = 64
    token_dim: int = 192
    n_cell_tokens: int = 2
    n_context_tokens: int = 2
    n_private_tokens: int = 2
    n_memory_tokens: int = 8
    num_heads: int = 4
    num_encoder_blocks: int = 2
    num_joint_blocks: int = 4
    ff_multiplier: int = 4
    token_dropout: float = 0.0
    cell_dim: int = 32
    context_dim: int = 32
    private_dim: int = 16
    decoder_hidden_dims: tuple[int, ...] = (512, 1024)
    dropout: float = 0.1
    min_logvar: float = -8.0
    max_logvar: float = 8.0
    n_batches: int = 0
    batch_embed_dim: int = 0
    deep_tokenizer: bool = False
    # LeCun/statistician-style simplification: replace joint self-attention
    # with a single kernel-smoothing cross-attention pooler
    pooler_mode: bool = False
    cell_pool_feature: bool = False  # DiT-style: pool shared code over feature tokens
    # Encoder-side batch conditioning for better batch correction (helps scIB batch metrics)
    use_encoder_batch_cov: bool = False
    # Adversarial batch classifier for batch-invariance (DANN-style, Ganin et al. 2015)
    # Trained with gradient reversal on the shared latent
    adversarial_batch: bool = False
    adversary_hidden: int = 64
    # Optional structured-latent research configuration; disabled by the API.
    structured_latent: bool = False
    nuisance_pred_hidden: int = 64


class _GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky, 2015). Forward = identity,
    backward = -alpha * grad. Used to train a batch-invariant shared latent by
    adversarially fooling a batch classifier."""
    @staticmethod
    def forward(ctx, x, alpha: float = 1.0):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: Tensor, alpha: float = 1.0) -> Tensor:
    return _GradientReversal.apply(x, alpha)


class BatchAdversary(nn.Module):
    """Small MLP classifier that predicts batch from the shared latent.
    Paired with gradient reversal to enforce batch-invariance (DANN-style)."""
    def __init__(self, latent_dim: int, hidden: int, n_batches: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_batches),
        )

    def forward(self, shared_z: Tensor, alpha: float = 1.0) -> Tensor:
        reversed_z = gradient_reversal(shared_z, alpha)
        return self.classifier(reversed_z)


class NuisanceBatchPredictor(nn.Module):
    """Forward batch classifier with NO gradient reversal — same MLP shape
    as `BatchAdversary` but trained to PREDICT batch from a latent slot.
    Used in the structured-latent recipe (Option A) to make context_mu
    explicitly absorb nuisance: the gradient flows back into the encoder
    in the natural direction, pushing context toward batch-predictiveness.
    Combined with the cell-context cross-covariance penalty, this routes
    batch information into context and away from cell (= z_bio)."""
    def __init__(self, latent_dim: int, hidden: int, n_batches: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_batches),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.classifier(z)


class GaussianHead(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, min_logvar: float, max_logvar: float) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.mu = nn.Linear(input_dim, latent_dim) if latent_dim > 0 else None
        self.logvar = nn.Linear(input_dim, latent_dim) if latent_dim > 0 else None
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.latent_dim == 0:
            empty = x.new_empty((x.shape[0], 0))
            return empty, empty
        assert self.mu is not None and self.logvar is not None
        mu = self.mu(x)
        logvar = self.logvar(x).clamp(min=self.min_logvar, max=self.max_logvar)
        return mu, logvar


def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(prev, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        prev = dim
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def _reparameterize(mu: Tensor, logvar: Tensor, training: bool) -> Tensor:
    if not training:
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def _product_of_experts(mus: list[Tensor], logvars: list[Tensor]) -> tuple[Tensor, Tensor]:
    if not mus:
        raise ValueError("product_of_experts requires at least one posterior")

    precision_terms = [torch.ones_like(logvars[0])]
    precision_mu_terms = [torch.zeros_like(mus[0])]
    for mu, logvar in zip(mus, logvars, strict=True):
        var = torch.exp(logvar)
        precision = torch.reciprocal(var + 1.0e-8)
        precision_terms.append(precision)
        precision_mu_terms.append(mu * precision)

    total_precision = torch.stack(precision_terms, dim=0).sum(dim=0)
    joint_var = torch.reciprocal(total_precision + 1.0e-8)
    joint_mu = joint_var * torch.stack(precision_mu_terms, dim=0).sum(dim=0)
    joint_logvar = torch.log(joint_var + 1.0e-8)
    return joint_mu, joint_logvar


class GroupedFeatureTokenizer(nn.Module):
    def __init__(self, input_dim: int, group_size: int, token_dim: int, dropout: float, deep_tokenizer: bool = False) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.group_size = int(group_size)
        self.n_groups = int(math.ceil(self.input_dim / self.group_size))
        self.padded_dim = self.n_groups * self.group_size
        if deep_tokenizer:
            mid = max(self.group_size * 2, token_dim)
            self.token_proj = nn.Sequential(
                nn.Linear(self.group_size, mid),
                nn.GELU(),
                nn.Linear(mid, token_dim),
            )
        else:
            self.token_proj = nn.Linear(self.group_size, token_dim)
        self.group_embed = nn.Parameter(torch.randn(1, self.n_groups, token_dim) * 0.02)
        self.norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] < self.padded_dim:
            x = nn.functional.pad(x, (0, self.padded_dim - x.shape[-1]))
        tokens = x.view(x.shape[0], self.n_groups, self.group_size)
        tokens = self.token_proj(tokens)
        tokens = self.norm(tokens + self.group_embed)
        return self.dropout(tokens)


class SelfAttentionBlock(nn.Module):
    def __init__(self, token_dim: int, num_heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        inner_dim = token_dim * ff_multiplier
        self.ff_in = nn.Linear(token_dim, inner_dim * 2)
        self.ff_out = nn.Linear(inner_dim, token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, latents: Tensor) -> Tensor:
        normed = self.norm1(latents)
        attn_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        latents = latents + self.dropout(attn_out)
        normed = self.norm2(latents)
        gate, value = self.ff_in(normed).chunk(2, dim=-1)
        ff_out = self.ff_out(torch.nn.functional.silu(gate) * value)
        return latents + self.dropout(ff_out)


class KernelPoolerBlock(nn.Module):
    """Kernel-smoothing cross-attention pooler: latent queries attend to feature
    keys/values. This is a Nadaraya-Watson estimator — each latent token is a
    softmax-weighted sum of feature tokens. Simple, interpretable, no feature-
    feature self-attention.
    """
    def __init__(self, token_dim: int, num_heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(token_dim)
        self.k_norm = nn.LayerNorm(token_dim)
        self.cross_attn = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        # Post-pool FFN to let latent tokens mix channels
        self.ff_norm = nn.LayerNorm(token_dim)
        inner = token_dim * ff_multiplier
        self.ff_in = nn.Linear(token_dim, inner * 2)
        self.ff_out = nn.Linear(inner, token_dim)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, latents: Tensor, features: Tensor) -> Tensor:
        q = self.q_norm(latents)
        k = self.k_norm(features)
        pooled, _ = self.cross_attn(q, k, k, need_weights=False)
        latents = latents + self.attn_dropout(pooled)
        # Channel-mixing on pooled latents only
        normed = self.ff_norm(latents)
        gate, value = self.ff_in(normed).chunk(2, dim=-1)
        ff_out = self.ff_out(torch.nn.functional.silu(gate) * value)
        return latents + self.ff_dropout(ff_out)


class FactorizedModalityEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        group_size: int,
        token_dim: int,
        n_cell_tokens: int,
        n_context_tokens: int,
        n_private_tokens: int,
        n_memory_tokens: int,
        num_heads: int,
        num_encoder_blocks: int,
        num_joint_blocks: int,
        ff_multiplier: int,
        token_dropout: float,
        cell_dim: int,
        context_dim: int,
        private_dim: int,
        dropout: float,
        min_logvar: float,
        max_logvar: float,
        deep_tokenizer: bool = False,
        pooler_mode: bool = False,
        encoder_cov_dim: int = 0,
        cell_pool_feature: bool = False,
    ) -> None:
        super().__init__()
        self.pooler_mode = bool(pooler_mode)
        self.cell_pool_feature = bool(cell_pool_feature)
        self.encoder_cov_dim = int(encoder_cov_dim)
        # If encoder-side batch conditioning: append a projection of batch embedding
        # to the input before tokenization
        tokenizer_input_dim = input_dim + self.encoder_cov_dim
        self.tokenizer = GroupedFeatureTokenizer(input_dim=tokenizer_input_dim, group_size=group_size, token_dim=token_dim, dropout=dropout, deep_tokenizer=deep_tokenizer)
        self.n_cell_tokens = int(n_cell_tokens)
        self.n_context_tokens = int(n_context_tokens)
        self.n_private_tokens = int(n_private_tokens)
        self.n_memory_tokens = int(n_memory_tokens)
        self.n_shared_tokens = self.n_cell_tokens + self.n_context_tokens
        self.token_dropout = float(token_dropout)
        self.cell_tokens = nn.Parameter(torch.randn(1, self.n_cell_tokens, token_dim) * 0.02)
        self.context_tokens = nn.Parameter(torch.randn(1, self.n_context_tokens, token_dim) * 0.02)
        self.private_tokens = nn.Parameter(torch.randn(1, self.n_private_tokens, token_dim) * 0.02)
        self.memory_tokens = (
            nn.Parameter(torch.randn(1, self.n_memory_tokens, token_dim) * 0.02)
            if self.n_memory_tokens > 0
            else None
        )
        self.token_type_embed = nn.Parameter(torch.randn(1, 5, token_dim) * 0.02)
        # Modality-specific feature blocks precede the joint latent blocks.
        self.feature_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    token_dim=token_dim,
                    num_heads=num_heads,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(num_encoder_blocks)
            ]
        )
        # Pooler mode: single kernel-smoothing cross-attention (N blocks of it)
        if self.pooler_mode:
            self.pooler_blocks = nn.ModuleList(
                [
                    KernelPoolerBlock(token_dim=token_dim, num_heads=num_heads, ff_multiplier=ff_multiplier, dropout=dropout)
                    for _ in range(num_joint_blocks)
                ]
            )
        else:
            self.pooler_blocks = nn.ModuleList()
        self.joint_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    token_dim=token_dim,
                    num_heads=num_heads,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(num_joint_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(token_dim)
        self.cell_head = GaussianHead(token_dim, cell_dim, min_logvar=min_logvar, max_logvar=max_logvar)
        self.context_head = GaussianHead(token_dim, context_dim, min_logvar=min_logvar, max_logvar=max_logvar)
        self.private_head = GaussianHead(token_dim, private_dim, min_logvar=min_logvar, max_logvar=max_logvar)

    def _drop_feature_tokens(self, feature_tokens: Tensor) -> Tensor:
        if not self.training or self.token_dropout <= 0.0 or feature_tokens.shape[1] <= 1:
            return feature_tokens
        keep = max(1, int(math.ceil(feature_tokens.shape[1] * (1.0 - self.token_dropout))))
        if keep >= feature_tokens.shape[1]:
            return feature_tokens
        noise = torch.rand(feature_tokens.shape[0], feature_tokens.shape[1], device=feature_tokens.device)
        keep_idx = torch.topk(noise, k=keep, dim=1, largest=False).indices
        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, feature_tokens.shape[-1])
        return feature_tokens.gather(dim=1, index=gather_idx)

    def forward(self, x: Tensor, batch_cov: Tensor | None = None) -> dict[str, Tensor]:
        # Encoder-side batch conditioning: concat batch covariate to input features
        if self.encoder_cov_dim > 0:
            if batch_cov is None:
                batch_cov = torch.zeros((x.shape[0], self.encoder_cov_dim), device=x.device, dtype=x.dtype)
            x = torch.cat([x, batch_cov], dim=-1)
        feature_tokens = self.tokenizer(x)
        feature_tokens = self._drop_feature_tokens(feature_tokens)
        feature_tokens = feature_tokens + self.token_type_embed[:, 4:5]
        for block in self.feature_blocks:
            feature_tokens = block(feature_tokens)

        batch_size = x.shape[0]
        cell_tokens = self.cell_tokens.expand(batch_size, -1, -1) + self.token_type_embed[:, 0:1]
        context_tokens = self.context_tokens.expand(batch_size, -1, -1) + self.token_type_embed[:, 1:2]
        private_tokens = self.private_tokens.expand(batch_size, -1, -1) + self.token_type_embed[:, 2:3]
        token_parts = [cell_tokens, context_tokens, private_tokens]
        if self.memory_tokens is not None:
            memory_tokens = self.memory_tokens.expand(batch_size, -1, -1) + self.token_type_embed[:, 3:4]
            token_parts.append(memory_tokens)
        n_latent = sum(part.shape[1] for part in token_parts)

        feat_out = None
        if self.pooler_mode:
            # LeCun/statistician-style: latents are weighted sums of features (kernel smoothing).
            # No feature-feature self-attention in the joint stage.
            latents = torch.cat(token_parts, dim=1)
            for block in self.pooler_blocks:
                latents = block(latents, feature_tokens)
            latent_tokens = self.output_norm(latents)
        else:
            tokens = torch.cat(token_parts + [feature_tokens], dim=1)
            for block in self.joint_blocks:
                tokens = block(tokens)
            latent_tokens = self.output_norm(tokens[:, :n_latent])
            feat_out = self.output_norm(tokens[:, n_latent:])
        if self.cell_pool_feature and feat_out is not None:
            # DiT-style mean-pool over self-attended feature tokens (no latent bottleneck)
            cell_repr = feat_out.mean(dim=1)
        else:
            cell_repr = latent_tokens[:, : self.n_cell_tokens].mean(dim=1)
        context_start = self.n_cell_tokens
        context_end = context_start + self.n_context_tokens
        context_repr = latent_tokens[:, context_start:context_end].mean(dim=1)
        private_start = context_end
        private_end = private_start + self.n_private_tokens
        private_repr = latent_tokens[:, private_start:private_end].mean(dim=1)

        cell_mu, cell_logvar = self.cell_head(cell_repr)
        context_mu, context_logvar = self.context_head(context_repr)
        private_mu, private_logvar = self.private_head(private_repr)
        return {
            "cell_mu": cell_mu,
            "cell_logvar": cell_logvar,
            "context_mu": context_mu,
            "context_logvar": context_logvar,
            "private_mu": private_mu,
            "private_logvar": private_logvar,
        }


class FactorizedModalityDecoder(nn.Module):
    def __init__(
        self,
        *,
        output_dim: int,
        cell_dim: int,
        context_dim: int,
        private_dim: int,
        batch_embed_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.batch_embed_dim = int(batch_embed_dim)
        shared_in = cell_dim + context_dim + self.batch_embed_dim
        full_in = shared_in + private_dim
        self.shared_decoder = _build_mlp(shared_in, hidden_dims, output_dim, dropout)
        self.full_decoder = _build_mlp(full_in, hidden_dims, output_dim, dropout)

    def _merge_covariate(self, x: Tensor, batch_cov: Tensor | None) -> Tensor:
        if self.batch_embed_dim <= 0:
            return x
        if batch_cov is None:
            batch_cov = torch.zeros((x.shape[0], self.batch_embed_dim), device=x.device, dtype=x.dtype)
        return torch.cat([x, batch_cov], dim=-1)

    def reconstruct(self, cell_z: Tensor, context_z: Tensor, private_z: Tensor, batch_cov: Tensor | None = None) -> Tensor:
        x = torch.cat([cell_z, context_z, private_z], dim=-1)
        return self.full_decoder(self._merge_covariate(x, batch_cov))

    def reconstruct_from_shared(self, cell_z: Tensor, context_z: Tensor, batch_cov: Tensor | None = None) -> Tensor:
        x = torch.cat([cell_z, context_z], dim=-1)
        return self.shared_decoder(self._merge_covariate(x, batch_cov))


class MinuetModule(nn.Module):
    def __init__(self, config: MinuetConfig) -> None:
        super().__init__()
        self.config = config
        self.batch_embed_dim = int(config.batch_embed_dim) if int(config.n_batches) > 0 else 0
        self.batch_embedding = (
            nn.Embedding(int(config.n_batches), self.batch_embed_dim)
            if self.batch_embed_dim > 0
            else None
        )
        self.rna = FactorizedModalityEncoder(
            input_dim=config.rna_dim,
            group_size=config.group_size,
            token_dim=config.token_dim,
            n_cell_tokens=config.n_cell_tokens,
            n_context_tokens=config.n_context_tokens,
            n_private_tokens=config.n_private_tokens,
            n_memory_tokens=config.n_memory_tokens,
            num_heads=config.num_heads,
            num_encoder_blocks=config.num_encoder_blocks,
            num_joint_blocks=config.num_joint_blocks,
            ff_multiplier=config.ff_multiplier,
            token_dropout=config.token_dropout,
            cell_dim=config.cell_dim,
            context_dim=config.context_dim,
            private_dim=config.private_dim,
            dropout=config.dropout,
            min_logvar=config.min_logvar,
            max_logvar=config.max_logvar,
            deep_tokenizer=config.deep_tokenizer,
            pooler_mode=config.pooler_mode,
            cell_pool_feature=config.cell_pool_feature,
            encoder_cov_dim=(self.batch_embed_dim if config.use_encoder_batch_cov else 0),
        )
        self.atac = FactorizedModalityEncoder(
            input_dim=config.atac_dim,
            group_size=config.group_size,
            token_dim=config.token_dim,
            n_cell_tokens=config.n_cell_tokens,
            n_context_tokens=config.n_context_tokens,
            n_private_tokens=config.n_private_tokens,
            n_memory_tokens=config.n_memory_tokens,
            num_heads=config.num_heads,
            num_encoder_blocks=config.num_encoder_blocks,
            num_joint_blocks=config.num_joint_blocks,
            ff_multiplier=config.ff_multiplier,
            token_dropout=config.token_dropout,
            cell_dim=config.cell_dim,
            context_dim=config.context_dim,
            private_dim=config.private_dim,
            dropout=config.dropout,
            min_logvar=config.min_logvar,
            max_logvar=config.max_logvar,
            deep_tokenizer=config.deep_tokenizer,
            pooler_mode=config.pooler_mode,
            cell_pool_feature=config.cell_pool_feature,
            encoder_cov_dim=(self.batch_embed_dim if config.use_encoder_batch_cov else 0),
        )
        self.rna_decoder = FactorizedModalityDecoder(
            output_dim=config.rna_dim,
            cell_dim=config.cell_dim,
            context_dim=config.context_dim,
            private_dim=config.private_dim,
            batch_embed_dim=self.batch_embed_dim,
            hidden_dims=config.decoder_hidden_dims,
            dropout=config.dropout,
        )
        self.atac_decoder = FactorizedModalityDecoder(
            output_dim=config.atac_dim,
            cell_dim=config.cell_dim,
            context_dim=config.context_dim,
            private_dim=config.private_dim,
            batch_embed_dim=self.batch_embed_dim,
            hidden_dims=config.decoder_hidden_dims,
            dropout=config.dropout,
        )

        # Structured-latent (Option A): nuisance head on context_mu. Active
        # only if `structured_latent=True`, `n_batches > 1`, and `context_dim > 0`.
        self.structured_latent = (
            bool(getattr(config, "structured_latent", False))
            and int(config.n_batches) > 1
            and int(config.context_dim) > 0
        )

        # Adversarial batch classifier (DANN): predicts batch from shared latent
        # with gradient reversal. Encourages the shared latent to be batch-invariant.
        # Under structured-latent, "shared" is cell-only, so the adversary dim
        # is cell_dim alone (else cell_dim + context_dim).
        self.adversarial_batch = bool(getattr(config, "adversarial_batch", False)) and int(config.n_batches) > 1
        if self.adversarial_batch:
            shared_dim = (
                int(config.cell_dim)
                if self.structured_latent
                else int(config.cell_dim) + int(config.context_dim)
            )
            self.batch_adversary = BatchAdversary(
                latent_dim=shared_dim,
                hidden=int(getattr(config, "adversary_hidden", 64)),
                n_batches=int(config.n_batches),
            )
        else:
            self.batch_adversary = None
        if self.structured_latent:
            self.nuisance_head = NuisanceBatchPredictor(
                latent_dim=int(config.context_dim),
                hidden=int(getattr(config, "nuisance_pred_hidden", 64)),
                n_batches=int(config.n_batches),
            )
        else:
            self.nuisance_head = None

    def _batch_covariate(self, batch_idx: Tensor | None) -> Tensor | None:
        if self.batch_embedding is None or batch_idx is None:
            return None
        return self.batch_embedding(batch_idx.long())

    def _encode_modality(
        self,
        tower: FactorizedModalityEncoder,
        x: Tensor,
        prefix: str,
        batch_cov: Tensor | None,
        out: dict[str, Tensor],
    ) -> None:
        # Pass batch_cov to encoder if it uses encoder-side batch conditioning
        enc_cov = batch_cov if tower.encoder_cov_dim > 0 else None
        encoded = tower(x, batch_cov=enc_cov)
        cell_mu = encoded["cell_mu"]
        cell_logvar = encoded["cell_logvar"]
        context_mu = encoded["context_mu"]
        context_logvar = encoded["context_logvar"]
        private_mu = encoded["private_mu"]
        private_logvar = encoded["private_logvar"]
        cell_z = _reparameterize(cell_mu, cell_logvar, training=self.training)
        context_z = _reparameterize(context_mu, context_logvar, training=self.training)
        private_z = _reparameterize(private_mu, private_logvar, training=self.training)
        out[f"{prefix}_cell"] = cell_mu
        out[f"{prefix}_cell_mu"] = cell_mu
        out[f"{prefix}_cell_logvar"] = cell_logvar
        out[f"{prefix}_context"] = context_mu
        out[f"{prefix}_context_mu"] = context_mu
        out[f"{prefix}_context_logvar"] = context_logvar
        out[f"{prefix}_private"] = private_mu
        out[f"{prefix}_private_mu"] = private_mu
        out[f"{prefix}_private_logvar"] = private_logvar
        out[f"{prefix}_cell_z"] = cell_z
        out[f"{prefix}_context_z"] = context_z
        out[f"{prefix}_private_z"] = private_z
        # Under structured-latent (Option A), context is reserved for nuisance
        # absorption — downstream embeddings use cell only. Under classical
        # mode, shared = cat(cell, context) as before.
        if self.structured_latent:
            out[f"{prefix}_shared"] = cell_mu
            out[f"{prefix}_shared_z"] = cell_z
        else:
            out[f"{prefix}_shared"] = torch.cat([cell_mu, context_mu], dim=-1)
            out[f"{prefix}_shared_z"] = torch.cat([cell_z, context_z], dim=-1)
        decoder = self.rna_decoder if prefix == "rna" else self.atac_decoder
        out[f"{prefix}_shared_self_recon"] = decoder.reconstruct_from_shared(cell_z, context_z, batch_cov)

    def forward(
        self,
        rna_x: Tensor | None = None,
        atac_x: Tensor | None = None,
        rna_batch_idx: Tensor | None = None,
        atac_batch_idx: Tensor | None = None,
    ) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        rna_cov = self._batch_covariate(rna_batch_idx)
        atac_cov = self._batch_covariate(atac_batch_idx)

        if rna_x is not None:
            self._encode_modality(self.rna, rna_x, "rna", rna_cov, out)

        if atac_x is not None:
            self._encode_modality(self.atac, atac_x, "atac", atac_cov, out)

        if rna_x is not None and atac_x is not None:
            joint_cell_mu, joint_cell_logvar = _product_of_experts(
                [out["rna_cell_mu"], out["atac_cell_mu"]],
                [out["rna_cell_logvar"], out["atac_cell_logvar"]],
            )
            joint_context_mu, joint_context_logvar = _product_of_experts(
                [out["rna_context_mu"], out["atac_context_mu"]],
                [out["rna_context_logvar"], out["atac_context_logvar"]],
            )
            joint_cell_z = _reparameterize(joint_cell_mu, joint_cell_logvar, training=self.training)
            joint_context_z = _reparameterize(joint_context_mu, joint_context_logvar, training=self.training)
            if self.structured_latent:
                # Downstream uses cell only; context_mu is held back for nuisance absorption.
                joint_shared = joint_cell_mu
                joint_shared_z = joint_cell_z
                joint_shared_logvar = joint_cell_logvar
            else:
                joint_shared = torch.cat([joint_cell_mu, joint_context_mu], dim=-1)
                joint_shared_z = torch.cat([joint_cell_z, joint_context_z], dim=-1)
                joint_shared_logvar = torch.cat([joint_cell_logvar, joint_context_logvar], dim=-1)
            out["joint_cell"] = joint_cell_mu
            out["joint_cell_mu"] = joint_cell_mu
            out["joint_cell_logvar"] = joint_cell_logvar
            out["joint_context"] = joint_context_mu
            out["joint_context_mu"] = joint_context_mu
            out["joint_context_logvar"] = joint_context_logvar
            out["joint_cell_z"] = joint_cell_z
            out["joint_context_z"] = joint_context_z
            out["joint_shared"] = joint_shared
            out["joint_shared_mu"] = joint_shared
            out["joint_shared_logvar"] = joint_shared_logvar
            out["joint_shared_z"] = joint_shared_z
            out["rna_recon"] = self.rna_decoder.reconstruct(joint_cell_z, joint_context_z, out["rna_private_z"], rna_cov)
            out["atac_recon"] = self.atac_decoder.reconstruct(joint_cell_z, joint_context_z, out["atac_private_z"], atac_cov)
            out["rna_shared_recon"] = self.rna_decoder.reconstruct_from_shared(joint_cell_z, joint_context_z, rna_cov)
            out["atac_shared_recon"] = self.atac_decoder.reconstruct_from_shared(joint_cell_z, joint_context_z, atac_cov)
        else:
            if rna_x is not None:
                out["rna_recon"] = self.rna_decoder.reconstruct(out["rna_cell_z"], out["rna_context_z"], out["rna_private_z"], rna_cov)
            if atac_x is not None:
                out["atac_recon"] = self.atac_decoder.reconstruct(out["atac_cell_z"], out["atac_context_z"], out["atac_private_z"], atac_cov)

        # Adversarial batch classifier: predict batch from the shared latent,
        # with gradient reversal so encoder learns batch-invariant representations.
        if self.batch_adversary is not None:
            # Prefer joint_shared if paired, else modality-specific shared
            if "joint_shared" in out:
                shared_for_adv = out["joint_shared"]
                batch_idx_for_adv = rna_batch_idx if rna_batch_idx is not None else atac_batch_idx
            elif rna_x is not None:
                shared_for_adv = out["rna_shared"]
                batch_idx_for_adv = rna_batch_idx
            elif atac_x is not None:
                shared_for_adv = out["atac_shared"]
                batch_idx_for_adv = atac_batch_idx
            else:
                shared_for_adv = None
                batch_idx_for_adv = None
            if shared_for_adv is not None and batch_idx_for_adv is not None:
                out["batch_adv_logits"] = self.batch_adversary(shared_for_adv, alpha=1.0)
                out["batch_adv_target"] = batch_idx_for_adv.long()

        # Structured-latent (Option A) nuisance head: predict batch from
        # context_mu with NO gradient reversal — pushes context to be
        # batch-predictive, absorbing nuisance away from cell (= z_bio).
        if self.nuisance_head is not None:
            if "joint_context" in out:
                context_for_nuisance = out["joint_context"]
                batch_idx_for_nuisance = rna_batch_idx if rna_batch_idx is not None else atac_batch_idx
            elif rna_x is not None:
                context_for_nuisance = out["rna_context_mu"]
                batch_idx_for_nuisance = rna_batch_idx
            elif atac_x is not None:
                context_for_nuisance = out["atac_context_mu"]
                batch_idx_for_nuisance = atac_batch_idx
            else:
                context_for_nuisance = None
                batch_idx_for_nuisance = None
            if context_for_nuisance is not None and batch_idx_for_nuisance is not None:
                out["nuisance_pred_logits"] = self.nuisance_head(context_for_nuisance)
                out["nuisance_pred_target"] = batch_idx_for_nuisance.long()

        return out


# Backward-compatible low-level import for pre-0.2 research code.
Minuet = MinuetModule
