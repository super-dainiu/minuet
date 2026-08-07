from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairedTransforms:
    rna_target_sum: float = 1.0e4
    atac_target_sum: float = 1.0e4
    atac_idf: np.ndarray | None = None


def build_atac_idf(
    atac_sources: list[tuple[sp.csr_matrix, np.ndarray | None]],
) -> np.ndarray:
    if not atac_sources:
        raise ValueError("atac_sources must not be empty")

    n_features = int(atac_sources[0][0].shape[1])
    atac_counts = np.zeros(n_features, dtype=np.float64)
    n_cells = 0.0

    for atac_csr, rows in atac_sources:
        if atac_csr.shape[1] != n_features:
            raise ValueError("All ATAC sources must share the same feature dimension")

        if rows is not None:
            view = atac_csr[np.asarray(rows, dtype=np.int64)]
        else:
            view = atac_csr

        atac_counts += np.bincount(view.indices, minlength=n_features)
        n_cells += float(view.shape[0])

    return np.log1p(n_cells / (1.0 + atac_counts)).astype(np.float32, copy=False)


def build_paired_transforms(
    atac_csr: sp.csr_matrix,
    train_indices: np.ndarray | None = None,
    rna_target_sum: float = 1.0e4,
    atac_target_sum: float = 1.0e4,
) -> PairedTransforms:
    return PairedTransforms(
        rna_target_sum=rna_target_sum,
        atac_target_sum=atac_target_sum,
        atac_idf=build_atac_idf([(atac_csr, train_indices)]),
    )


def _normalize_dense_rows(x: np.ndarray, target_sum: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        denom = float(x.sum())
        if denom <= 0.0:
            denom = 1.0
        return np.log1p((x / denom) * target_sum)

    denom = x.sum(axis=1, keepdims=True)
    denom[denom <= 0.0] = 1.0
    return np.log1p((x / denom) * target_sum)


def transform_rna_dense(x: np.ndarray, target_sum: float = 1.0e4) -> np.ndarray:
    return _normalize_dense_rows(x, target_sum)


def transform_atac_dense(x: np.ndarray, atac_idf: np.ndarray, target_sum: float = 1.0e4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    atac_idf = np.asarray(atac_idf, dtype=np.float32)

    if x.ndim == 1:
        denom = float(x.sum())
        if denom <= 0.0:
            denom = 1.0
        return np.log1p((x / denom) * target_sum * atac_idf)

    denom = x.sum(axis=1, keepdims=True)
    denom[denom <= 0.0] = 1.0
    return np.log1p((x / denom) * target_sum * atac_idf[None, :])


class PairedSparseDataset(Dataset):
    def __init__(
        self,
        rna_csr: sp.csr_matrix,
        atac_csr: sp.csr_matrix,
        indices: np.ndarray,
        transforms: PairedTransforms,
        batch_idx: np.ndarray | None = None,
        label_idx: np.ndarray | None = None,
        donor_idx: np.ndarray | None = None,
        context_idx: np.ndarray | None = None,
    ) -> None:
        self.rna = rna_csr
        self.atac = atac_csr
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transforms = transforms
        self.batch_idx = None if batch_idx is None else np.asarray(batch_idx, dtype=np.int64)
        # Optional recorded biological context for within-context association.
        self.label_idx = None if label_idx is None else np.asarray(label_idx, dtype=np.int64)
        self.donor_idx = None if donor_idx is None else np.asarray(donor_idx, dtype=np.int64)
        self.context_idx = None if context_idx is None else np.asarray(context_idx, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[idx])

        rna = self.rna[row].toarray().ravel().astype(np.float32, copy=False)
        atac = self.atac[row].toarray().ravel().astype(np.float32, copy=False)

        rna = transform_rna_dense(rna, target_sum=self.transforms.rna_target_sum)
        if self.transforms.atac_idf is None:
            raise ValueError("PairedTransforms.atac_idf must be set")
        atac = transform_atac_dense(atac, self.transforms.atac_idf, target_sum=self.transforms.atac_target_sum)

        out = {
            "rna": torch.from_numpy(rna),
            "atac": torch.from_numpy(atac),
            "row_idx": torch.tensor(row, dtype=torch.long),
        }
        if self.batch_idx is not None:
            batch_idx = torch.tensor(int(self.batch_idx[idx]), dtype=torch.long)
            out["rna_batch_idx"] = batch_idx
            out["atac_batch_idx"] = batch_idx
        if self.label_idx is not None:
            out["label_idx"] = torch.tensor(int(self.label_idx[idx]), dtype=torch.long)
        if self.donor_idx is not None:
            out["donor_idx"] = torch.tensor(int(self.donor_idx[idx]), dtype=torch.long)
        if self.context_idx is not None:
            out["context_idx"] = torch.tensor(int(self.context_idx[idx]), dtype=torch.long)
        return out
