"""User-facing Minuet model."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Literal

import numpy as np
import scipy.sparse as sp
import torch
from mudata import MuData
from sklearn.utils.extmath import randomized_svd
from tqdm.auto import tqdm

from .module import MinuetModule


class Minuet:
    """Minuet representation of paired single-cell RNA and ATAC profiles.

    Minuet is fitted once on paired reference cells from many donors. Its two
    encoders then embed query cells, including cells of new donors, one cell at
    a time: query mapping uses no labels, no parameter updates, and no
    statistics of the query set.

    Parameters
    ----------
    mdata
        MuData object registered with :meth:`Minuet.setup_mudata`.
    n_hidden
        Width of the residual encoder trunks.
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
        ``"batch"`` uses BatchNorm in the encoders. ``"layer"`` uses LayerNorm,
        which normalizes each cell on its own; use it when query cells come
        from a study that is absent from the reference.
    prediction_rank
        Rank of each assay's block of the prediction target.
    seed
        Seed of parameter initialization, target fitting, and batch sampling.

    Examples
    --------
    >>> mdata = mudata.read_h5mu(path_to_reference)
    >>> minuet.Minuet.setup_mudata(mdata, donor_key="donor", covariate_keys=["disease"])
    >>> model = minuet.Minuet(mdata)
    >>> model.train()
    >>> mdata.obsm["X_minuet"] = model.get_latent_representation()

    Notes
    -----
    The defaults are the configuration of the Minuet paper.
    """

    def __init__(
        self,
        mdata: MuData,
        n_hidden: int = 224,
        n_latent: int = 64,
        n_private: int = 16,
        n_layers: int = 3,
        n_decoder_hidden: int = 64,
        dropout_rate: float = 0.2,
        norm: Literal["batch", "layer"] = "batch",
        prediction_rank: int = 32,
        seed: int = 0,
    ):
        if "minuet" not in mdata.uns:
            raise ValueError("Register mdata with Minuet.setup_mudata first.")
        self.mdata = mdata
        self.setup_args = dict(mdata.uns["minuet"])
        self.var_names = {
            assay: list(mdata[self.setup_args[f"{assay}_modality"]].var_names) for assay in ("rna", "atac")
        }
        self.init_params = {
            "n_hidden": n_hidden,
            "n_latent": n_latent,
            "n_private": n_private,
            "n_layers": n_layers,
            "n_decoder_hidden": n_decoder_hidden,
            "dropout_rate": dropout_rate,
            "norm": norm,
            "prediction_rank": prediction_rank,
            "seed": seed,
        }
        torch.manual_seed(seed)
        self.module = MinuetModule(
            len(self.var_names["rna"]),
            len(self.var_names["atac"]),
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_private=n_private,
            n_layers=n_layers,
            n_decoder_hidden=n_decoder_hidden,
            dropout_rate=dropout_rate,
            norm=norm,
            prediction_rank=prediction_rank,
        )
        self.history: list[float] = []

    @staticmethod
    def setup_mudata(
        mdata: MuData,
        donor_key: str,
        covariate_keys: Sequence[str] | None = None,
        rna_modality: str = "rna",
        atac_modality: str = "atac",
        rna_layer: str | None = None,
        atac_layer: str | None = None,
    ) -> None:
        """Register a MuData object of paired reference cells.

        Parameters
        ----------
        mdata
            MuData object with raw RNA and ATAC counts of the same cells in the
            same order.
        donor_key
            Column of ``mdata.obs`` with donor identifiers.
        covariate_keys
            Columns of ``mdata.obs`` with biological covariates, such as disease
            status or sex. Donor penalties act within the groups these columns
            define, so differences between groups are kept. If ``None``, all
            cells form one group.
        rna_modality, atac_modality
            Names of the RNA and ATAC modalities.
        rna_layer, atac_layer
            Layers with the counts. If ``None``, ``.X`` is used.
        """
        if not mdata[rna_modality].obs_names.equals(mdata[atac_modality].obs_names):
            raise ValueError("RNA and ATAC must contain the same cells in the same order.")
        for key in [donor_key, *(covariate_keys or [])]:
            if key not in mdata.obs:
                raise KeyError(f"{key!r} is not a column of mdata.obs.")
        mdata.uns["minuet"] = {
            "donor_key": donor_key,
            "covariate_keys": list(covariate_keys or []),
            "rna_modality": rna_modality,
            "atac_modality": atac_modality,
            "rna_layer": rna_layer or "",
            "atac_layer": atac_layer or "",
        }

    def train(
        self,
        max_updates: int = 16000,
        lr: float = 3e-4,
        weight_decay: float = 3e-7,
        late_lr_factor: float = 0.25,
        covariance_start: float = 0.8,
        temperature: float = 0.1,
        pair_weight: float = 2.0,
        prediction_weight: float = 1.5,
        donor_weight: float = 0.1,
        covariance_weight: float = 0.35,
        kl_weight: float = 7e-5,
        groups_per_batch: int = 2,
        donors_per_group: int = 4,
        cells_per_donor: int = 12,
        device: str | None = None,
        progress_bar: bool = True,
    ) -> None:
        """Fit the model on the registered reference cells.

        Each batch draws ``groups_per_batch`` covariate groups, up to
        ``donors_per_group`` donors per group, and ``cells_per_donor`` cells per
        donor. Only donors with at least ``cells_per_donor`` cells in a group,
        and groups with at least two such donors, are sampled. The loss of every
        100th update is appended to ``history``.

        Parameters
        ----------
        max_updates
            Number of parameter updates.
        lr
            Learning rate of Adam.
        weight_decay
            L2 penalty of Adam.
        late_lr_factor
            Factor applied to the learning rate once the covariance-alignment
            penalty is switched on.
        covariance_start
            Fraction of ``max_updates`` after which the covariance-alignment
            penalty is switched on.
        temperature
            Temperature of the within-donor contrastive loss.
        pair_weight
            Weight of the within-donor contrastive loss.
        prediction_weight
            Weight of the shared prediction loss.
        donor_weight
            Weight of the donor-association penalty.
        covariance_weight
            Weight of the covariance-alignment penalty.
        kl_weight
            Weight of the KL divergence of the private variables.
        groups_per_batch
            Covariate groups per batch.
        donors_per_group
            Donors per group and batch.
        cells_per_donor
            Cells per donor and batch.
        device
            Torch device. If ``None``, CUDA is used when available.
        progress_bar
            Whether to show a progress bar.
        """
        args = self.setup_args
        seed = self.init_params["seed"]
        rna, atac = self._counts(self.mdata)
        obs = self.mdata.obs.loc[self.mdata[args["rna_modality"]].obs_names]
        donor = np.unique(obs[args["donor_key"]].astype(str), return_inverse=True)[1]
        keys = args["covariate_keys"]
        groups = obs[keys].astype(str).agg("|".join, axis=1) if keys else np.zeros(len(obs))
        group = np.unique(groups, return_inverse=True)[1]
        target = _fit_target(rna, atac, donor, self.init_params["prediction_rank"], seed)

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        module = self.module.to(device).train()
        optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
        start = round(covariance_start * max_updates)
        batches = _block_batches(
            donor, group, max_updates, seed, groups_per_batch, donors_per_group, cells_per_donor
        )
        for update, rows in enumerate(tqdm(batches, total=max_updates, disable=not progress_bar)):
            if update == start:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr * late_lr_factor
            loss = module.loss(
                torch.as_tensor(rna[rows].toarray(), device=device),
                torch.as_tensor(atac[rows].toarray(), device=device),
                torch.as_tensor(target[rows], device=device),
                torch.as_tensor(np.unique(donor[rows], return_inverse=True)[1], device=device),
                torch.as_tensor(np.unique(group[rows], return_inverse=True)[1], device=device),
                temperature=temperature,
                kl_weight=kl_weight,
                pair_weight=pair_weight,
                prediction_weight=prediction_weight,
                donor_weight=donor_weight,
                covariance_weight=covariance_weight if update >= start else 0.0,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 10.0)
            optimizer.step()
            if update % 100 == 0:
                self.history.append(loss.item())
        module.eval()

    @torch.no_grad()
    def get_latent_representation(
        self,
        mdata: MuData | None = None,
        modality: Literal["joint", "rna", "atac"] = "joint",
        batch_size: int = 512,
    ) -> np.ndarray:
        """Embed paired cells with the frozen encoders.

        Parameters
        ----------
        mdata
            MuData object with the same modalities, layers, and features as the
            reference. Donor and covariate columns are not needed. If ``None``,
            the reference is used.
        modality
            ``"joint"`` returns the average of the RNA and ATAC representations;
            ``"rna"`` and ``"atac"`` return one of them.
        batch_size
            Number of cells per forward pass.

        Returns
        -------
        Array of shape ``(n_cells, n_latent)``.
        """
        rna, atac = self._counts(self.mdata if mdata is None else mdata)
        module = self.module.eval()
        device = next(module.parameters()).device
        latent = []
        for start in range(0, rna.shape[0], batch_size):
            z_rna, z_atac = module(
                torch.as_tensor(rna[start : start + batch_size].toarray(), device=device),
                torch.as_tensor(atac[start : start + batch_size].toarray(), device=device),
            )
            z = {"joint": 0.5 * (z_rna + z_atac), "rna": z_rna, "atac": z_atac}[modality]
            latent.append(z.cpu().numpy())
        return np.concatenate(latent)

    def save(self, dir_path: str, overwrite: bool = False) -> None:
        """Save the model to ``dir_path/model.pt``.

        Parameters
        ----------
        dir_path
            Directory to create.
        overwrite
            Whether to write into an existing directory.
        """
        os.makedirs(dir_path, exist_ok=overwrite)
        torch.save(
            {
                "setup_args": self.setup_args,
                "init_params": self.init_params,
                "var_names": self.var_names,
                "state_dict": self.module.state_dict(),
            },
            os.path.join(dir_path, "model.pt"),
        )

    @classmethod
    def load(cls, dir_path: str, mdata: MuData) -> Minuet:
        """Load a saved model.

        Parameters
        ----------
        dir_path
            Directory written by :meth:`Minuet.save`.
        mdata
            MuData object to attach, for example query cells. It must have the
            modalities, layers, and features of the reference.

        Examples
        --------
        >>> model = minuet.Minuet.load("minuet_model", query_mdata)
        >>> query_mdata.obsm["X_minuet"] = model.get_latent_representation()
        """
        saved = torch.load(os.path.join(dir_path, "model.pt"), map_location="cpu")
        mdata.uns["minuet"] = saved["setup_args"]
        model = cls(mdata, **saved["init_params"])
        if model.var_names != saved["var_names"]:
            raise ValueError("Features of mdata differ from those of the saved model.")
        model.module.load_state_dict(saved["state_dict"])
        model.module.eval()
        return model

    def _counts(self, mdata: MuData) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        counts = []
        for assay in ("rna", "atac"):
            adata = mdata[self.setup_args[f"{assay}_modality"]]
            if list(adata.var_names) != self.var_names[assay]:
                raise ValueError(f"{assay.upper()} features differ from those of the reference.")
            layer = self.setup_args[f"{assay}_layer"]
            counts.append(sp.csr_matrix(adata.layers[layer] if layer else adata.X, dtype=np.float32))
        if not mdata[self.setup_args["rna_modality"]].obs_names.equals(
            mdata[self.setup_args["atac_modality"]].obs_names
        ):
            raise ValueError("RNA and ATAC must contain the same cells in the same order.")
        return counts[0], counts[1]


def _fit_target(rna: sp.csr_matrix, atac: sp.csr_matrix, donor: np.ndarray, rank: int, seed: int) -> np.ndarray:
    """Standardized scores of a donor-centered randomized SVD of each assay, concatenated."""
    blocks = []
    for offset, counts in enumerate((rna, atac)):
        x = counts.toarray()
        for d in np.unique(donor):
            cells = donor == d
            x[cells] -= x[cells].mean(axis=0, dtype=np.float64).astype(np.float32)
        u, s, _ = randomized_svd(x, rank, n_iter=5, random_state=seed + offset, flip_sign=True)
        scores = (u * s).astype(np.float32)
        mean = scores.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(scores.std(axis=0, dtype=np.float64).astype(np.float32), np.float32(1e-6))
        blocks.append((scores - mean) / scale)
    return np.concatenate(blocks, axis=1)


def _block_batches(
    donor: np.ndarray,
    group: np.ndarray,
    n_batches: int,
    seed: int,
    groups_per_batch: int,
    donors_per_group: int,
    cells_per_donor: int,
) -> Iterator[np.ndarray]:
    """Row indices of batches made of equal-size donor blocks within covariate groups."""
    blocks: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row, (d, g) in enumerate(zip(donor.tolist(), group.tolist())):
        blocks[g][d].append(row)
    eligible = []
    for g in sorted(blocks):
        donors = [cells for _, cells in sorted(blocks[g].items()) if len(cells) >= cells_per_donor]
        if len(donors) >= 2:
            eligible.append(donors)
    if not eligible:
        raise ValueError(f"No covariate group has two donors with {cells_per_donor} cells each.")
    for batch in range(n_batches):
        if batch % 100 == 0:
            generator = torch.Generator().manual_seed(seed + 1_000_003 * (batch // 100))
        rows: list[int] = []
        for g in torch.randperm(len(eligible), generator=generator)[:groups_per_batch].tolist():
            for d in torch.randperm(len(eligible[g]), generator=generator)[:donors_per_group].tolist():
                cells = eligible[g][d]
                rows += [cells[i] for i in torch.randperm(len(cells), generator=generator)[:cells_per_donor].tolist()]
        yield np.asarray(rows)
