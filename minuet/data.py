from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


def load_split_prefixes(split_csv: str | Path) -> dict[str, str]:
    frame = pd.read_csv(split_csv)
    return dict(zip(frame["sample_prefix"], frame["split"]))


def load_split_rows(split_path: str | Path, split: str) -> np.ndarray:
    split_data = np.load(split_path)
    return np.asarray(split_data[split], dtype=np.int64)


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
    ) -> None:
        self.rna = rna_csr
        self.atac = atac_csr
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transforms = transforms
        self.batch_idx = None if batch_idx is None else np.asarray(batch_idx, dtype=np.int64)
        # Optional context label (e.g. cell type) for conditional losses
        # (conditional CLUB / CCL / HSIC). -1 marks an unlabelled cell.
        self.label_idx = None if label_idx is None else np.asarray(label_idx, dtype=np.int64)

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
        }
        if self.batch_idx is not None:
            batch_idx = torch.tensor(int(self.batch_idx[idx]), dtype=torch.long)
            out["rna_batch_idx"] = batch_idx
            out["atac_batch_idx"] = batch_idx
        if self.label_idx is not None:
            out["label_idx"] = torch.tensor(int(self.label_idx[idx]), dtype=torch.long)
        return out


class SparseModalityDataset(Dataset):
    def __init__(
        self,
        matrix_csr: sp.csr_matrix,
        indices: np.ndarray,
        modality: str,
        transforms: PairedTransforms,
        batch_idx: np.ndarray | None = None,
    ) -> None:
        if modality not in {"rna", "atac"}:
            raise ValueError(f"Unsupported modality: {modality}")

        self.matrix = matrix_csr
        self.indices = np.asarray(indices, dtype=np.int64)
        self.modality = modality
        self.transforms = transforms
        self.batch_idx = None if batch_idx is None else np.asarray(batch_idx, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[idx])
        x = self.matrix[row].toarray().ravel().astype(np.float32, copy=False)

        if self.modality == "rna":
            x = transform_rna_dense(x, target_sum=self.transforms.rna_target_sum)
        else:
            if self.transforms.atac_idf is None:
                raise ValueError("PairedTransforms.atac_idf must be set for ATAC data")
            x = transform_atac_dense(x, self.transforms.atac_idf, target_sum=self.transforms.atac_target_sum)

        out = {self.modality: torch.from_numpy(x)}
        if self.batch_idx is not None:
            out[f"{self.modality}_batch_idx"] = torch.tensor(int(self.batch_idx[idx]), dtype=torch.long)
        return out


def load_sparse_npz(path: str | Path) -> sp.csr_matrix:
    mat = sp.load_npz(path)
    if not sp.isspmatrix_csr(mat):
        mat = mat.tocsr()
    return mat


def resolve_artifact_matrix(artifact_dir: str | Path, modality: str) -> Path:
    root = Path(artifact_dir)
    if modality == "rna":
        candidates = [
            root / "rna.npz",
            root / "rna_top.npz",
            root / "wang_rna_top.npz",
        ]
    elif modality == "atac":
        candidates = [
            root / "atac.npz",
            root / "atac_top.npz",
            root / "wang_atac_top.npz",
        ]
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {modality} matrix found under {root}")


def resolve_artifact_splits(artifact_dir: str | Path) -> Path:
    root = Path(artifact_dir)
    candidates = [
        root / "splits.npz",
        root / "split_indices.npz",
        root / "wang_split_indices.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No split file found under {root}")


def load_artifact_split_rows(artifact_dir: str | Path, split: str) -> np.ndarray:
    return load_split_rows(resolve_artifact_splits(artifact_dir), split)


def load_feature_names(artifact_dir: str | Path, modality: str) -> list[str] | None:
    root = Path(artifact_dir)
    if modality == "rna":
        candidates = [
            root / "selected_rna_features.json",
            root / "rna_features.json",
            root / "feature_names.json",
        ]
    elif modality == "atac":
        candidates = [
            root / "selected_atac_features.json",
            root / "atac_features.json",
            root / "feature_names.json",
        ]
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    for candidate in candidates:
        if candidate.exists():
            import json

            return list(json.loads(candidate.read_text()))
    return None


def assert_matching_feature_space(artifact_dirs: list[str | Path], modality: str) -> list[str] | None:
    reference: list[str] | None = None
    for artifact_dir in artifact_dirs:
        features = load_feature_names(artifact_dir, modality)
        if features is None:
            continue
        if reference is None:
            reference = features
            continue
        if features != reference:
            raise ValueError(
                f"{modality} feature space mismatch between artifacts; "
                "all mixed-pretraining artifacts must share the same selected feature vocabulary"
            )
    return reference
