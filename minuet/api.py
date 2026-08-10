"""Small AnnData/MuData interface for fitting and frozen-query encoding."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Sequence
import json

import numpy as np
import scipy.sparse as sp
import torch

from .data import PopulationBatchConfig, PopulationBatchSampler, dense_rows
from .fit_targets import FitOnlyLowRankTargets
from .losses import (
    COVARIANCE_START_FRACTION,
    MinuetLosses,
    PAIRED_TARGET_RANK,
)
from .model import (
    MinuetConfig,
    MinuetEncoder,
    MinuetModule,
    PRODUCTION_LATE_LR_MULTIPLIER,
    PRODUCTION_LEARNING_RATE,
    PRODUCTION_UPDATES,
    PRODUCTION_WEIGHT_DECAY,
)


_SETUP_KEY = "_minuet_setup"
_MODEL_FILE = "model.pt"
_REGISTRY_FILE = "registry.json"
_CHECKPOINT_SCHEMA = "minuet-public-frozen-query-v2"


def _as_csr(matrix: Any) -> sp.csr_matrix:
    if sp.issparse(matrix):
        result = sp.csr_matrix(matrix, dtype=np.float32)
    else:
        array = np.asarray(matrix, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("modality matrices must be two-dimensional")
        result = sp.csr_matrix(array)
    if result.data.size and (
        not np.isfinite(result.data).all() or (result.data < 0).any()
    ):
        raise ValueError("Minuet inputs must contain finite, non-negative counts")
    return result


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
    if accelerator in {"gpu", "cuda"} and not torch.cuda.is_available():
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


class Minuet:
    """Compact paired RNA--ATAC learner with a frozen per-cell query map."""

    @classmethod
    def setup_mudata(
        cls,
        mdata: Any,
        rna_layer: str | None = None,
        atac_layer: str | None = None,
        modalities: dict[str, str] | None = None,
        donor_key: str | None = None,
        context_key: str | None = None,
    ) -> None:
        modalities = dict(modalities or {})
        rna_mod = modalities.get("rna_layer", modalities.get("rna", "rna"))
        atac_mod = modalities.get("atac_layer", modalities.get("atac", "atac"))
        if rna_mod not in mdata.mod or atac_mod not in mdata.mod:
            raise KeyError(f"MuData must contain modalities {rna_mod!r} and {atac_mod!r}")
        rna, atac = mdata.mod[rna_mod], mdata.mod[atac_mod]
        if rna.n_obs != atac.n_obs or not np.array_equal(rna.obs_names, atac.obs_names):
            raise ValueError("Minuet requires paired modalities with identical obs_names")
        _layer(rna, rna_layer)
        _layer(atac, atac_layer)
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
    ) -> None:
        _layer(adata, layer)
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
    ) -> None:
        if _SETUP_KEY not in adata.uns:
            raise ValueError("Run Minuet.setup_mudata() or Minuet.setup_anndata() first")
        self.adata = adata
        self.registry_ = dict(adata.uns[_SETUP_KEY])
        rna, atac, obs_names, rna_names, atac_names = self._extract(
            adata, n_genes, n_regions, self.registry_
        )
        self._rna, self._atac = rna, atac
        self.obs_names_ = obs_names
        self.rna_var_names_, self.atac_var_names_ = rna_names, atac_names
        self.n_genes_, self.n_regions_ = rna.shape[1], atac.shape[1]
        if not len(obs_names) or len(set(obs_names)) != len(obs_names):
            raise ValueError("obs_names must be non-empty and unique")
        if len(set(rna_names)) != len(rna_names) or len(set(atac_names)) != len(atac_names):
            raise ValueError("feature names must be unique within each modality")
        self._donor_values = self._obs_values(
            adata, self.registry_.get("donor_key"), required=False
        )
        self._context_values = self._obs_values(
            adata, self.registry_.get("context_key"), required=False
        )
        self.config_ = MinuetConfig(self.n_genes_, self.n_regions_)
        self.module_ = MinuetEncoder(self.config_)
        self.module_.eval()
        self.history_: dict[str, list[float]] = {
            "step": [],
            "train_loss": [],
            "pair": [],
            "prediction": [],
            "donor_association": [],
            "covariance_agreement": [],
        }
        self.is_trained_ = False
        self.fitted_steps_ = 0
        self.objective_ = "production-v3"
        self.device_ = torch.device("cpu")

    @staticmethod
    def _extract(
        adata: Any,
        n_genes: int | None,
        n_regions: int | None,
        registry: dict[str, Any],
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
        if registry["kind"] == "mudata":
            rna = adata.mod[registry["rna_mod"]]
            atac = adata.mod[registry["atac_mod"]]
            if rna.n_obs != atac.n_obs or not np.array_equal(rna.obs_names, atac.obs_names):
                raise ValueError("query modalities must have identical obs_names")
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
            raise ValueError("n_genes must split two non-empty feature blocks")
        if n_regions is not None and int(n_regions) != inferred_regions:
            raise ValueError("n_genes + n_regions must equal adata.n_vars")
        return (
            matrix[:, :n_genes].tocsr(),
            matrix[:, n_genes:].tocsr(),
            _strings(adata.obs_names),
            _strings(adata.var_names[:n_genes]),
            _strings(adata.var_names[n_genes:]),
        )

    def _obs_values(
        self, adata: Any, key: str | None, *, required: bool
    ) -> np.ndarray | None:
        if key is None:
            if required:
                raise ValueError("donor_key and context_key are required for fitting")
            return None
        if key in adata.obs:
            return _strings(adata.obs[key])
        if self.registry_["kind"] == "mudata":
            rna = adata.mod[self.registry_["rna_mod"]]
            if key in rna.obs:
                return _strings(rna.obs[key])
        if required:
            raise KeyError(f"obs column {key!r} is not present")
        return None

    @staticmethod
    def _codes(values: np.ndarray) -> np.ndarray:
        categories = sorted(set(values.tolist()))
        lookup = {value: index for index, value in enumerate(categories)}
        return np.asarray([lookup[value] for value in values], dtype=np.int64)

    def train(
        self,
        *,
        max_steps: int = PRODUCTION_UPDATES,
        accelerator: str = "auto",
        devices: Any = "auto",
        seed: int = 0,
    ) -> None:
        """Fit the frozen production configuration; ``max_steps`` exists for smoke tests."""
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        donor_values = self._obs_values(
            self.adata, self.registry_.get("donor_key"), required=True
        )
        context_values = self._obs_values(
            self.adata, self.registry_.get("context_key"), required=True
        )
        assert donor_values is not None and context_values is not None
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.set_float32_matmul_precision("high")
        donor_codes, context_codes = self._codes(donor_values), self._codes(context_values)
        targets = FitOnlyLowRankTargets.fit(
            self._rna,
            self._atac,
            donor_values,
            rank=PAIRED_TARGET_RANK,
            seed=seed,
        )
        block = 12
        supported_contexts = sum(
            sum(
                np.sum((context_values == context) & (donor_values == donor)) >= block
                for donor in np.unique(donor_values[context_values == context])
            )
            >= 2
            for context in np.unique(context_values)
        )
        sampler = PopulationBatchSampler(
            donor_values.tolist(),
            context_values.tolist(),
            PopulationBatchConfig(
                contexts_per_batch=min(2, supported_contexts),
                donors_per_context=4,
                cells_per_block=block,
                batches_per_epoch=100,
                seed=seed,
            ),
        )
        self.device_ = _device(accelerator, devices)
        fit_module = MinuetModule(self.config_).to(self.device_)
        objective = MinuetLosses(self.config_.shared_dim, targets.rank).to(self.device_)
        parameters = [*fit_module.parameters(), *objective.parameters()]
        optimizer = torch.optim.Adam(
            parameters,
            lr=PRODUCTION_LEARNING_RATE,
            weight_decay=PRODUCTION_WEIGHT_DECAY,
        )
        covariance_start = int(round(COVARIANCE_START_FRACTION * max_steps))
        self.history_ = {key: [] for key in self.history_}
        update = 0
        epoch = 0
        while update < max_steps:
            for rows in sampler.epoch(epoch):
                rows_array = np.asarray(rows, dtype=np.int64)
                rna = dense_rows(self._rna, rows_array, self.device_)
                atac = dense_rows(self._atac, rows_array, self.device_)
                donor = torch.as_tensor(
                    donor_codes[rows_array], dtype=torch.long, device=self.device_
                )
                context = torch.as_tensor(
                    context_codes[rows_array], dtype=torch.long, device=self.device_
                )
                rna_target, atac_target = targets.batch(rows_array, self.device_)
                if update == covariance_start:
                    for group in optimizer.param_groups:
                        group["lr"] = (
                            PRODUCTION_LEARNING_RATE * PRODUCTION_LATE_LR_MULTIPLIER
                        )
                output = fit_module(rna, atac)
                losses = objective(
                    rna,
                    atac,
                    output,
                    donor=donor,
                    biology_context=context,
                    rna_target_score=rna_target,
                    atac_target_score=atac_target,
                    covariance_active=update >= covariance_start,
                )
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(parameters, 10.0)
                optimizer.step()
                if update % max(1, max_steps // 20) == 0 or update + 1 == max_steps:
                    self.history_["step"].append(float(update))
                    self.history_["train_loss"].append(float(losses["total"].detach()))
                    for name in (
                        "pair",
                        "prediction",
                        "donor_association",
                        "covariance_agreement",
                    ):
                        self.history_[name].append(float(losses[name].detach()))
                update += 1
                if update >= max_steps:
                    break
            epoch += 1
        self.module_ = fit_module.encoder().to(self.device_)
        self.module_.eval()
        self.module_.requires_grad_(False)
        self.is_trained_ = True
        self.fitted_steps_ = int(max_steps)
        del objective, fit_module, targets

    def _query_matrices(
        self, adata: Any | None
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
        if adata is None or adata is self.adata:
            return self._rna, self._atac, self.obs_names_
        rna, atac, obs, rna_names, atac_names = self._extract(
            adata, self.n_genes_, self.n_regions_, self.registry_
        )
        if not np.array_equal(rna_names, self.rna_var_names_) or not np.array_equal(
            atac_names, self.atac_var_names_
        ):
            raise ValueError("query feature names and order must match fitting data")
        return rna, atac, obs

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: Any | None = None,
        modality: Literal["joint", "expression", "accessibility"] = "joint",
        indices: Sequence[int] | None = None,
        batch_size: int = 128,
    ) -> np.ndarray:
        if not self.is_trained_:
            raise RuntimeError("Train or load the model before inference")
        if modality not in {"joint", "expression", "accessibility"}:
            raise ValueError(f"unknown modality: {modality!r}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rna, atac, _ = self._query_matrices(adata)
        if indices is None:
            selected = np.arange(rna.shape[0], dtype=np.int64)
        else:
            selected = np.asarray(indices, dtype=np.int64)
            if selected.ndim != 1 or (
                len(selected) and (selected.min() < 0 or selected.max() >= rna.shape[0])
            ):
                raise IndexError("indices must be one-dimensional and in range")
        key = {
            "joint": "joint_shared",
            "expression": "rna_shared",
            "accessibility": "atac_shared",
        }[modality]
        result: list[np.ndarray] = []
        self.module_.eval()
        for start in range(0, len(selected), batch_size):
            rows = selected[start : start + batch_size]
            output = self.module_.encode(
                dense_rows(rna, rows, self.device_),
                dense_rows(atac, rows, self.device_),
            )
            result.append(output[key].cpu().numpy())
        return (
            np.concatenate(result)
            if result
            else np.empty((0, self.config_.shared_dim), dtype=np.float32)
        )

    def to_device(self, device: str | int | torch.device) -> None:
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        self.device_ = torch.device(device)
        self.module_.to(self.device_)

    def save(
        self,
        dir_path: str | Path,
        *,
        overwrite: bool = False,
        save_anndata: bool = False,
    ) -> None:
        if not self.is_trained_:
            raise RuntimeError("only a fitted encoder can be saved")
        path = Path(dir_path)
        if path.exists() and any(path.iterdir()) and not overwrite:
            raise FileExistsError(f"{path} is not empty; pass overwrite=True")
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": _CHECKPOINT_SCHEMA,
            "encoder": self.module_.checkpoint(),
            "registry": self.registry_,
            "n_genes": self.n_genes_,
            "n_regions": self.n_regions_,
            "rna_var_names": self.rna_var_names_.tolist(),
            "atac_var_names": self.atac_var_names_.tolist(),
            "history": self.history_,
            "fitted_steps": self.fitted_steps_,
            "objective": self.objective_,
            "production_config": {
                "model": asdict(self.config_),
                "updates": PRODUCTION_UPDATES,
                "learning_rate": PRODUCTION_LEARNING_RATE,
                "weight_decay": PRODUCTION_WEIGHT_DECAY,
                "late_learning_rate_multiplier": PRODUCTION_LATE_LR_MULTIPLIER,
                "covariance_start_fraction": COVARIANCE_START_FRACTION,
            },
        }
        torch.save(payload, path / _MODEL_FILE)
        (path / _REGISTRY_FILE).write_text(
            json.dumps(
                {
                    "registry": self.registry_,
                    "rna_var_names": self.rna_var_names_.tolist(),
                    "atac_var_names": self.atac_var_names_.tolist(),
                },
                indent=2,
            )
            + "\n"
        )
        if save_anndata:
            if self.registry_["kind"] == "mudata":
                self.adata.write(path / "adata.h5mu")
            else:
                self.adata.write_h5ad(path / "adata.h5ad")

    @classmethod
    def load(
        cls,
        dir_path: str | Path,
        *,
        adata: Any | None = None,
        accelerator: str = "auto",
        device: str | int = "auto",
    ) -> "Minuet":
        path = Path(dir_path)
        payload = torch.load(path / _MODEL_FILE, map_location="cpu", weights_only=False)
        if payload.get("schema") != _CHECKPOINT_SCHEMA:
            raise ValueError(
                "incompatible checkpoint; Minuet 0.3 requires the encoder-only v2 schema"
            )
        if adata is None:
            if (path / "adata.h5ad").exists():
                import anndata

                adata = anndata.read_h5ad(path / "adata.h5ad")
            elif (path / "adata.h5mu").exists():
                import mudata

                adata = mudata.read_h5mu(path / "adata.h5mu")
            else:
                raise ValueError("adata is required because the model was saved without data")
        if _SETUP_KEY not in adata.uns:
            adata.uns[_SETUP_KEY] = payload["registry"]
        model = cls(
            adata,
            n_genes=int(payload["n_genes"]),
            n_regions=int(payload["n_regions"]),
        )
        if model.rna_var_names_.tolist() != payload["rna_var_names"] or (
            model.atac_var_names_.tolist() != payload["atac_var_names"]
        ):
            raise ValueError("saved model and adata feature names do not match")
        model.module_ = MinuetEncoder.from_checkpoint(payload["encoder"])
        model.config_ = model.module_.config
        model.history_ = payload["history"]
        model.fitted_steps_ = int(payload["fitted_steps"])
        model.objective_ = payload["objective"]
        model.is_trained_ = True
        target = (
            torch.device(f"cuda:{device}" if isinstance(device, int) else device)
            if device != "auto"
            else _device(accelerator, "auto")
        )
        model.to_device(target)
        return model

    @staticmethod
    def load_registry(dir_path: str | Path) -> dict[str, Any]:
        return json.loads((Path(dir_path) / _REGISTRY_FILE).read_text())

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
            f"n_regions={self.n_regions_}, n_latent={self.config_.shared_dim}, "
            f"state={state!r})"
        )
