"""scvi-style public interface for paired RNA--ATAC integration."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Sequence
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

from .data import PairedSparseDataset, PairedTransforms, build_paired_transforms
from .losses import MinuetLosses
from .model import MinuetConfig, MinuetModule


_SETUP_KEY = "_minuet_setup"
_MODEL_FILE = "model.pt"
_REGISTRY_FILE = "registry.json"


def _as_csr(matrix: Any) -> sp.csr_matrix:
    if sp.issparse(matrix):
        return sp.csr_matrix(matrix, dtype=np.float32)
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("modality matrices must be two-dimensional")
    return sp.csr_matrix(array)


def _layer(adata: Any, key: str | None) -> Any:
    if key is None:
        return adata.X
    if key not in adata.layers:
        raise KeyError(f"layer {key!r} is not present")
    return adata.layers[key]


def _strings(values: Any) -> np.ndarray:
    return np.asarray(values).astype(str)


def _device(accelerator: str = "auto", devices: Any = "auto") -> torch.device:
    if accelerator not in {"auto", "cpu", "gpu", "cuda", "mps"}:
        raise ValueError("accelerator must be one of: auto, cpu, gpu, cuda, mps")
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    wants_cuda = accelerator in {"gpu", "cuda"}
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError("GPU was requested but CUDA is unavailable")
    if torch.cuda.is_available() and accelerator != "mps":
        index = 0
        if isinstance(devices, int):
            index = devices
        elif isinstance(devices, (list, tuple)) and devices:
            index = int(devices[0])
        elif isinstance(devices, str) and devices not in {"auto", ""}:
            index = int(devices.split(",")[0])
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def _metadata(container: Any, key: str | None, modality: Any | None = None) -> np.ndarray | None:
    if key is None:
        return None
    if key in container.obs:
        return _strings(container.obs[key])
    if modality is not None and key in modality.obs:
        return _strings(modality.obs[key])
    raise KeyError(f"obs column {key!r} is not present")


class Minuet:
    """Paired RNA--ATAC integration with a scvi-tools-like workflow.

    Call :meth:`setup_mudata` or :meth:`setup_anndata`, construct the model,
    then call :meth:`train`. The public API deliberately mirrors the small
    subset of ``scvi.model.MULTIVI`` used in standard integration workflows.
    """

    @classmethod
    def setup_mudata(
        cls,
        mdata: Any,
        rna_layer: str | None = None,
        atac_layer: str | None = None,
        modalities: dict[str, str] | None = None,
        donor_key: str | None = None,
        context_key: str | None = None,
        **_: Any,
    ) -> None:
        modalities = dict(modalities or {})
        rna_mod = modalities.get("rna_layer", modalities.get("rna", "rna"))
        atac_mod = modalities.get("atac_layer", modalities.get("atac", "atac"))
        if rna_mod not in mdata.mod or atac_mod not in mdata.mod:
            raise KeyError(
                f"MuData must contain modalities {rna_mod!r} and {atac_mod!r}"
            )
        rna = mdata.mod[rna_mod]
        atac = mdata.mod[atac_mod]
        if rna.n_obs != atac.n_obs or not np.array_equal(rna.obs_names, atac.obs_names):
            raise ValueError("Minuet currently requires paired modalities with identical obs_names")
        _layer(rna, rna_layer)
        _layer(atac, atac_layer)
        _metadata(mdata, donor_key, rna)
        _metadata(mdata, context_key, rna)
        mdata.uns[_SETUP_KEY] = {
            "kind": "mudata",
            "rna_mod": rna_mod,
            "atac_mod": atac_mod,
            "rna_layer": rna_layer,
            "atac_layer": atac_layer,
            "donor_key": donor_key,
            "context_key": context_key,
        }

    @classmethod
    def setup_anndata(
        cls,
        adata: Any,
        layer: str | None = None,
        donor_key: str | None = None,
        context_key: str | None = None,
        **_: Any,
    ) -> None:
        """Register a concatenated AnnData (RNA columns followed by ATAC columns).

        ``n_genes`` must be supplied to the constructor so the two feature
        blocks can be separated.
        """
        _layer(adata, layer)
        _metadata(adata, donor_key)
        _metadata(adata, context_key)
        adata.uns[_SETUP_KEY] = {
            "kind": "anndata",
            "layer": layer,
            "donor_key": donor_key,
            "context_key": context_key,
        }

    def __init__(
        self,
        adata: Any,
        n_genes: int | None = None,
        n_regions: int | None = None,
        n_hidden: int | None = 128,
        n_latent: int | None = 32,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        dropout_rate: float = 0.1,
        fully_paired: bool = True,
        modality_weights: str = "equal",
        modality_penalty: str = "Jeffreys",
        region_factors: bool = True,
        gene_likelihood: str = "zinb",
        dispersion: str = "gene",
        use_batch_norm: str = "none",
        use_layer_norm: str = "both",
        latent_distribution: str = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = False,
        protein_dispersion: str = "protein",
        **model_kwargs: Any,
    ) -> None:
        if _SETUP_KEY not in adata.uns:
            raise ValueError("Run Minuet.setup_mudata() or Minuet.setup_anndata() first")
        if not fully_paired:
            raise NotImplementedError("Minuet currently supports fully paired RNA+ATAC only")
        # Accepted for drop-in constructor compatibility with MULTIVI. Minuet's
        # architecture fixes these choices rather than branching on them.
        del (
            modality_weights,
            modality_penalty,
            region_factors,
            gene_likelihood,
            dispersion,
            use_batch_norm,
            use_layer_norm,
            latent_distribution,
            deeply_inject_covariates,
            encode_covariates,
            protein_dispersion,
        )
        self.adata = adata
        self.registry_ = dict(adata.uns[_SETUP_KEY])
        rna, atac, obs_names, rna_names, atac_names = self._extract(adata, n_genes, n_regions)
        self._rna = rna
        self._atac = atac
        self.obs_names_ = obs_names
        self.rna_var_names_ = rna_names
        self.atac_var_names_ = atac_names
        self.n_genes_ = int(rna.shape[1])
        self.n_regions_ = int(atac.shape[1])
        if not self.obs_names_.size or len(set(self.obs_names_)) != len(self.obs_names_):
            raise ValueError("obs_names must be non-empty and unique")
        if len(set(self.rna_var_names_)) != len(self.rna_var_names_):
            raise ValueError("RNA var_names must be unique")
        if len(set(self.atac_var_names_)) != len(self.atac_var_names_):
            raise ValueError("ATAC var_names must be unique")
        for name, matrix in (("RNA", rna), ("ATAC", atac)):
            if matrix.data.size and (
                not np.isfinite(matrix.data).all() or (matrix.data < 0).any()
            ):
                raise ValueError(f"{name} input must contain finite, non-negative values")

        self.batch_categories_: list[str] = []
        self._batch_codes = None
        self._donor_codes = self._categorical_codes(
            self._obs_values(adata, self.registry_.get("donor_key"))
        )
        self._context_codes = self._categorical_codes(
            self._obs_values(adata, self.registry_.get("context_key"))
        )

        hidden = int(n_hidden or 128)
        latent = int(n_latent or 32)
        if hidden < 4 or latent < 1:
            raise ValueError("n_hidden must be >= 4 and n_latent must be positive")
        heads = int(model_kwargs.pop("num_heads", 4))
        while heads > 1 and hidden % heads:
            heads -= 1
        private_dim = int(model_kwargs.pop("private_dim", max(8, latent // 2)))
        config = MinuetConfig(
            rna_dim=self.n_genes_,
            atac_dim=self.n_regions_,
            group_size=int(model_kwargs.pop("group_size", 64)),
            token_dim=hidden,
            n_cell_tokens=int(model_kwargs.pop("n_cell_tokens", 2)),
            n_context_tokens=0,
            n_private_tokens=int(model_kwargs.pop("n_private_tokens", 2)),
            n_memory_tokens=int(model_kwargs.pop("n_memory_tokens", 0)),
            num_heads=heads,
            num_encoder_blocks=int(n_layers_encoder),
            num_joint_blocks=int(model_kwargs.pop("n_joint_layers", n_layers_encoder)),
            ff_multiplier=int(model_kwargs.pop("ff_multiplier", 4)),
            cell_dim=latent,
            context_dim=0,
            private_dim=private_dim,
            decoder_hidden_dims=tuple(hidden for _ in range(int(n_layers_decoder))),
            dropout=float(dropout_rate),
            n_batches=0,
            batch_embed_dim=int(model_kwargs.pop("batch_embed_dim", 8)),
            **model_kwargs,
        )
        self.module_ = MinuetModule(config)
        self.target_rank_ = min(32, max(1, len(self.obs_names_) - 1), self.n_genes_, self.n_regions_)
        self.target_head_ = torch.nn.Linear(latent, 2 * self.target_rank_, bias=False)
        self.config_ = asdict(config)
        self.init_params_ = {
            "n_genes": self.n_genes_,
            "n_regions": self.n_regions_,
            "n_hidden": hidden,
            "n_latent": latent,
            "n_layers_encoder": int(n_layers_encoder),
            "n_layers_decoder": int(n_layers_decoder),
            "dropout_rate": float(dropout_rate),
            "fully_paired": True,
        }
        self.transforms_: PairedTransforms | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "validation_loss": []}
        self.is_trained_ = False
        self.objective_: str | None = None
        self.device_ = torch.device("cpu")

    @staticmethod
    def _categorical_codes(values: np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        categories = sorted(set(values.tolist()))
        lookup = {value: index for index, value in enumerate(categories)}
        return np.asarray([lookup[value] for value in values], dtype=np.int64)

    def _extract(
        self,
        adata: Any,
        n_genes: int | None,
        n_regions: int | None,
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
        registry = dict(adata.uns[_SETUP_KEY])
        if registry["kind"] == "mudata":
            rna = adata.mod[registry["rna_mod"]]
            atac = adata.mod[registry["atac_mod"]]
            return (
                _as_csr(_layer(rna, registry.get("rna_layer"))),
                _as_csr(_layer(atac, registry.get("atac_layer"))),
                _strings(rna.obs_names),
                _strings(rna.var_names),
                _strings(atac.var_names),
            )
        matrix = _as_csr(_layer(adata, registry.get("layer")))
        if n_genes is None:
            raise ValueError("n_genes is required for concatenated AnnData")
        n_genes = int(n_genes)
        inferred_regions = matrix.shape[1] - n_genes
        if n_genes < 1 or inferred_regions < 1:
            raise ValueError("n_genes must split AnnData into non-empty RNA and ATAC blocks")
        if n_regions is not None and int(n_regions) != inferred_regions:
            raise ValueError("n_genes + n_regions must equal adata.n_vars")
        return (
            matrix[:, :n_genes].tocsr(),
            matrix[:, n_genes:].tocsr(),
            _strings(adata.obs_names),
            _strings(adata.var_names[:n_genes]),
            _strings(adata.var_names[n_genes:]),
        )

    def _obs_values(self, adata: Any, key: str | None) -> np.ndarray | None:
        if key is None:
            return None
        if key in adata.obs:
            return _strings(adata.obs[key])
        if self.registry_["kind"] == "mudata":
            rna = adata.mod[self.registry_["rna_mod"]]
            if key in rna.obs:
                return _strings(rna.obs[key])
        raise KeyError(f"obs column {key!r} is not present")

    def _dataset(
        self,
        indices: np.ndarray,
        *,
        adata: Any | None = None,
    ) -> PairedSparseDataset:
        if self.transforms_ is None:
            raise RuntimeError("Train or load the model before inference")
        if adata is None or adata is self.adata:
            rna, atac = self._rna, self._atac
            batch_codes = self._batch_codes
            donor_codes = self._donor_codes
            context_codes = self._context_codes
        else:
            rna, atac, _, rna_names, atac_names = self._extract(
                adata, self.n_genes_, self.n_regions_
            )
            if not np.array_equal(rna_names, self.rna_var_names_) or not np.array_equal(
                atac_names, self.atac_var_names_
            ):
                raise ValueError("query feature names and order must match the training data")
            batch_codes = None
            donor_codes = None
            context_codes = None
        selected_codes = None if batch_codes is None else batch_codes[indices]
        selected_donor = None if donor_codes is None else donor_codes[indices]
        selected_context = None if context_codes is None else context_codes[indices]
        return PairedSparseDataset(
            rna,
            atac,
            indices,
            self.transforms_,
            selected_codes,
            donor_idx=selected_donor,
            context_idx=selected_context,
        )

    def train(
        self,
        max_epochs: int = 100,
        lr: float = 5.0e-4,
        accelerator: str = "auto",
        devices: Any = "auto",
        train_size: float | None = 0.9,
        validation_size: float | None = None,
        batch_size: int = 128,
        weight_decay: float = 1.0e-4,
        early_stopping: bool = True,
        check_val_every_n_epoch: int | None = 1,
        n_epochs_kl_warmup: int | None = 10,
        **kwargs: Any,
    ) -> None:
        """Train the model; common argument names match ``MULTIVI.train``."""
        if max_epochs < 1 or batch_size < 2:
            raise ValueError("max_epochs must be positive and batch_size must be >= 2")
        if self._rna.shape[0] < 2:
            raise ValueError("training requires at least two paired cells")
        seed = int(kwargs.pop("seed", 0))
        patience = int(kwargs.pop("early_stopping_patience", 20))
        grad_clip = float(kwargs.pop("gradient_clip_val", 1.0))
        if self._donor_codes is None:
            raise ValueError("training requires donor_key in setup")
        if self._context_codes is None:
            raise ValueError("training requires context_key in setup")
        self.objective_ = "production-v3"
        if kwargs:
            raise TypeError(f"unsupported train arguments: {sorted(kwargs)}")
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        order = rng.permutation(self._rna.shape[0])
        fraction = 0.9 if train_size is None else float(train_size)
        if validation_size is not None:
            fraction = 1.0 - float(validation_size)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("train_size and validation_size must define a train fraction in (0, 1]")
        cut = min(max(int(round(len(order) * fraction)), 2), len(order))
        train_indices = order[:cut]
        validation_indices = order[cut:]
        self.transforms_ = build_paired_transforms(self._atac, train_indices)
        # Fitting-only common target: donor-centered low-rank scores for both views.
        fit_data = self._dataset(train_indices)
        fit_rna = np.stack([fit_data[i]["rna"].numpy() for i in range(len(fit_data))])
        fit_atac = np.stack([fit_data[i]["atac"].numpy() for i in range(len(fit_data))])
        fit_donor = self._donor_codes[train_indices]
        for value in np.unique(fit_donor):
            block = fit_donor == value
            fit_rna[block] -= fit_rna[block].mean(0, keepdims=True)
            fit_atac[block] -= fit_atac[block].mean(0, keepdims=True)
        def scores(values: np.ndarray) -> np.ndarray:
            u, singular, _ = np.linalg.svd(values, full_matrices=False)
            result = (u[:, : self.target_rank_] * singular[: self.target_rank_]).astype(np.float32)
            return (result - result.mean(0)) / np.maximum(result.std(0), 1e-6)
        target = np.concatenate((scores(fit_rna), scores(fit_atac)), axis=1)
        self._paired_target = np.zeros((self._rna.shape[0], target.shape[1]), dtype=np.float32)
        self._paired_target[train_indices] = target
        train_loader = DataLoader(
            self._dataset(train_indices), batch_size=batch_size, shuffle=True
        )

        self.device_ = _device(accelerator, devices)
        self.module_.to(self.device_)
        self.target_head_.to(self.device_)
        optimizer = torch.optim.AdamW(
            [*self.module_.parameters(), *self.target_head_.parameters()], lr=lr, weight_decay=weight_decay
        )
        del early_stopping, check_val_every_n_epoch
        self.history_ = {"train_loss": [], "validation_loss": []}
        for epoch in range(max_epochs):
            self.module_.train()
            values: list[float] = []
            kl_scale = (
                1.0
                if not n_epochs_kl_warmup
                else min(1.0, (epoch + 1) / int(n_epochs_kl_warmup))
            )
            for batch in train_loader:
                rna = batch["rna"].to(self.device_)
                atac = batch["atac"].to(self.device_)
                rna_batch = batch.get("rna_batch_idx")
                atac_batch = batch.get("atac_batch_idx")
                if rna_batch is not None:
                    rna_batch = rna_batch.to(self.device_)
                    atac_batch = atac_batch.to(self.device_)
                donor = batch.get("donor_idx")
                context = batch.get("context_idx")
                if donor is not None:
                    donor = donor.to(self.device_)
                    context = context.to(self.device_)
                out = self.module_(rna, atac, rna_batch, atac_batch)
                paired_target = torch.as_tensor(self._paired_target[batch["row_idx"].numpy()], device=self.device_)
                losses = MinuetLosses.total(rna, atac, out, donor=donor,
                    biology_context=context, paired_target=paired_target,
                    target_head=self.target_head_, private_kl_weight=7.23e-5 * kl_scale)
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.module_.parameters(), grad_clip)
                optimizer.step()
                values.append(float(losses["total"].detach()))
            train_loss = float(np.mean(values))
            self.history_["train_loss"].append(train_loss)

            validation_loss = float("nan")
            self.history_["validation_loss"].append(validation_loss)
        self.module_.eval()
        self.is_trained_ = True

    def _indices(self, adata: Any | None, indices: Sequence[int] | None) -> np.ndarray:
        n_obs = self._rna.shape[0] if adata is None else int(adata.n_obs)
        if indices is None:
            return np.arange(n_obs, dtype=np.int64)
        result = np.asarray(indices, dtype=np.int64)
        if result.ndim != 1 or (len(result) and (result.min() < 0 or result.max() >= n_obs)):
            raise IndexError("indices must be a one-dimensional in-range sequence")
        return result

    @torch.inference_mode()
    def _predict(
        self,
        adata: Any | None,
        indices: Sequence[int] | None,
        batch_size: int | None,
        key: str,
    ) -> np.ndarray:
        if not self.is_trained_:
            raise RuntimeError("Train or load the model before inference")
        selected = self._indices(adata, indices)
        loader = DataLoader(
            self._dataset(selected, adata=adata),
            batch_size=int(batch_size or 128),
        )
        self.module_.eval()
        result = []
        for batch in loader:
            rna = batch["rna"].to(self.device_)
            atac = batch["atac"].to(self.device_)
            rna_batch = batch.get("rna_batch_idx")
            atac_batch = batch.get("atac_batch_idx")
            if rna_batch is not None:
                rna_batch = rna_batch.to(self.device_)
                atac_batch = atac_batch.to(self.device_)
            out = self.module_(rna, atac, rna_batch, atac_batch)
            result.append(out[key].detach().cpu().numpy())
        width = self.config_["cell_dim"] if "shared" in key else 0
        return np.concatenate(result) if result else np.empty((0, width), dtype=np.float32)

    def get_latent_representation(
        self,
        adata: Any | None = None,
        modality: Literal["joint", "expression", "accessibility"] = "joint",
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
        return_dist: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not give_mean:
            raise NotImplementedError("only posterior means are exposed in the public API")
        keys = {
            "joint": "joint_shared",
            "expression": "rna_shared",
            "accessibility": "atac_shared",
        }
        if modality not in keys:
            raise ValueError(f"unknown modality: {modality!r}")
        mean = self._predict(adata, indices, batch_size, keys[modality])
        if return_dist:
            return mean, np.zeros_like(mean)
        return mean

    def get_normalized_expression(
        self,
        adata: Any | None = None,
        indices: Sequence[int] | None = None,
        gene_list: Sequence[str] | None = None,
        batch_size: int | None = None,
        return_numpy: bool = False,
        **_: Any,
    ) -> np.ndarray | pd.DataFrame:
        values = np.maximum(self._predict(adata, indices, batch_size, "rna_recon"), 0.0)
        names = self.rna_var_names_
        if gene_list is not None:
            lookup = {name: index for index, name in enumerate(names)}
            missing = sorted(set(gene_list) - set(lookup))
            if missing:
                raise KeyError(f"unknown genes: {missing}")
            columns = np.asarray([lookup[name] for name in gene_list])
            values = values[:, columns]
            names = np.asarray(gene_list)
        if return_numpy:
            return values
        selected = self._indices(adata, indices)
        obs_names = self.obs_names_ if adata is None else _strings(adata.obs_names)
        return pd.DataFrame(values, index=obs_names[selected], columns=names)

    def get_normalized_accessibility(
        self,
        adata: Any | None = None,
        indices: Sequence[int] | None = None,
        region_list: Sequence[str] | None = None,
        batch_size: int | None = None,
        return_numpy: bool = False,
        **_: Any,
    ) -> np.ndarray | pd.DataFrame:
        values = np.maximum(self._predict(adata, indices, batch_size, "atac_recon"), 0.0)
        names = self.atac_var_names_
        if region_list is not None:
            lookup = {name: index for index, name in enumerate(names)}
            missing = sorted(set(region_list) - set(lookup))
            if missing:
                raise KeyError(f"unknown regions: {missing}")
            columns = np.asarray([lookup[name] for name in region_list])
            values = values[:, columns]
            names = np.asarray(region_list)
        if return_numpy:
            return values
        selected = self._indices(adata, indices)
        obs_names = self.obs_names_ if adata is None else _strings(adata.obs_names)
        return pd.DataFrame(values, index=obs_names[selected], columns=names)

    def to_device(self, device: str | int | torch.device) -> None:
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        self.device_ = torch.device(device)
        self.module_.to(self.device_)
        self.target_head_.to(self.device_)

    def save(
        self,
        dir_path: str | Path,
        prefix: str | None = None,
        overwrite: bool = False,
        save_anndata: bool = False,
        **write_kwargs: Any,
    ) -> None:
        path = Path(dir_path)
        if path.exists() and any(path.iterdir()) and not overwrite:
            raise FileExistsError(f"{path} is not empty; pass overwrite=True")
        path.mkdir(parents=True, exist_ok=True)
        stem = "" if prefix is None else prefix
        payload = {
            "version": 1,
            "state_dict": self.module_.state_dict(),
            "target_head_state_dict": self.target_head_.state_dict(),
            "init_params": self.init_params_,
            "config": self.config_,
            "registry": self.registry_,
            "batch_categories": self.batch_categories_,
            "rna_var_names": self.rna_var_names_.tolist(),
            "atac_var_names": self.atac_var_names_.tolist(),
            "atac_idf": None if self.transforms_ is None else self.transforms_.atac_idf,
            "history": self.history_,
            "is_trained": self.is_trained_,
            "objective": self.objective_,
        }
        torch.save(payload, path / f"{stem}{_MODEL_FILE}")
        (path / f"{stem}{_REGISTRY_FILE}").write_text(
            json.dumps(
                {
                    "registry": self.registry_,
                    "rna_var_names": self.rna_var_names_.tolist(),
                    "atac_var_names": self.atac_var_names_.tolist(),
                    "batch_categories": self.batch_categories_,
                },
                indent=2,
            )
            + "\n"
        )
        if save_anndata:
            if self.registry_["kind"] == "mudata":
                self.adata.write(path / f"{stem}adata.h5mu", **write_kwargs)
            else:
                self.adata.write_h5ad(path / f"{stem}adata.h5ad", **write_kwargs)

    @classmethod
    def load(
        cls,
        dir_path: str | Path,
        adata: Any | None = None,
        accelerator: str = "auto",
        device: str | int = "auto",
        prefix: str | None = None,
    ) -> "Minuet":
        path = Path(dir_path)
        stem = "" if prefix is None else prefix
        payload = torch.load(path / f"{stem}{_MODEL_FILE}", map_location="cpu", weights_only=False)
        if adata is None:
            h5ad = path / f"{stem}adata.h5ad"
            h5mu = path / f"{stem}adata.h5mu"
            if h5ad.exists():
                import anndata

                adata = anndata.read_h5ad(h5ad)
            elif h5mu.exists():
                import mudata

                adata = mudata.read_h5mu(h5mu)
            else:
                raise ValueError("adata is required because the model was saved without data")
        if _SETUP_KEY not in adata.uns:
            adata.uns[_SETUP_KEY] = payload["registry"]
        model = cls(adata, **payload["init_params"])
        model.module_ = MinuetModule(MinuetConfig(**payload["config"]))
        model.config_ = dict(payload["config"])
        if model.rna_var_names_.tolist() != payload["rna_var_names"] or (
            model.atac_var_names_.tolist() != payload["atac_var_names"]
        ):
            raise ValueError("saved model and adata feature names do not match")
        model.module_.load_state_dict(payload["state_dict"])
        model.target_head_.load_state_dict(payload["target_head_state_dict"])
        model.batch_categories_ = list(payload["batch_categories"])
        saved_idf = payload["atac_idf"]
        model.transforms_ = (
            None
            if saved_idf is None
            else PairedTransforms(atac_idf=np.asarray(saved_idf, dtype=np.float32))
        )
        model.history_ = payload["history"]
        model.is_trained_ = bool(payload["is_trained"])
        model.objective_ = payload.get("objective")
        if device != "auto":
            target = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        else:
            target = _device(accelerator, "auto")
        model.to_device(target)
        model.module_.eval()
        return model

    @staticmethod
    def load_registry(dir_path: str | Path, prefix: str | None = None) -> dict[str, Any]:
        stem = "" if prefix is None else prefix
        return json.loads((Path(dir_path) / f"{stem}{_REGISTRY_FILE}").read_text())

    def view_anndata_setup(self, adata: Any | None = None) -> None:
        target = self.adata if adata is None else adata
        registry = dict(target.uns.get(_SETUP_KEY, {}))
        print(json.dumps(registry, indent=2))

    @property
    def history(self) -> dict[str, list[float]]:
        return self.history_

    @property
    def is_trained(self) -> bool:
        return self.is_trained_

    def __repr__(self) -> str:
        state = "trained" if self.is_trained_ else "untrained"
        return (
            f"Minuet(n_obs={self._rna.shape[0]}, n_genes={self.n_genes_}, "
            f"n_regions={self.n_regions_}, n_latent={self.config_['cell_dim']}, "
            f"state={state!r})"
        )
