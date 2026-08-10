"""Fitting-only low-rank targets shared by both assays."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
import torch
from torch import Tensor


def _donor_center(
    matrix: sp.spmatrix | np.ndarray, donor: np.ndarray
) -> np.ndarray:
    values = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    centered = np.asarray(values, dtype=np.float32, order="C").copy()
    for donor_value in np.unique(donor):
        block = donor == donor_value
        centered[block] -= centered[block].mean(
            axis=0, dtype=np.float64
        ).astype(np.float32)
    return centered


def _scores(centered: np.ndarray, rank: int, seed: int) -> np.ndarray:
    effective_rank = min(int(rank), centered.shape[0] - 1, centered.shape[1])
    if effective_rank < 1:
        raise ValueError("at least two fitting cells and one feature are required")
    u, singular, _ = randomized_svd(
        centered,
        n_components=effective_rank,
        n_iter=5,
        random_state=int(seed),
        flip_sign=True,
    )
    raw = (u * singular[None, :]).astype(np.float32, copy=False)
    mean = raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(
        raw.std(axis=0, dtype=np.float64).astype(np.float32), np.float32(1.0e-6)
    )
    return ((raw - mean) / scale).astype(np.float32)


@dataclass(frozen=True)
class FitOnlyLowRankTargets:
    rna_scores: np.ndarray
    atac_scores: np.ndarray

    @classmethod
    def fit(
        cls,
        rna: sp.spmatrix | np.ndarray,
        atac: sp.spmatrix | np.ndarray,
        donor: np.ndarray,
        *,
        rank: int,
        seed: int,
    ) -> "FitOnlyLowRankTargets":
        donor = np.asarray(donor)
        if rna.shape[0] != len(donor) or atac.shape[0] != len(donor):
            raise ValueError("fitting matrices and donor labels must be row aligned")
        common_rank = min(int(rank), len(donor) - 1, rna.shape[1], atac.shape[1])
        rna_scores = _scores(_donor_center(rna, donor), common_rank, seed)
        atac_scores = _scores(_donor_center(atac, donor), common_rank, seed + 1)
        return cls(rna_scores, atac_scores)

    @property
    def rank(self) -> int:
        return int(self.rna_scores.shape[1])

    def batch(
        self, rows: np.ndarray | Sequence[int], device: torch.device
    ) -> tuple[Tensor, Tensor]:
        index = np.asarray(rows, dtype=np.int64)
        return (
            torch.as_tensor(self.rna_scores[index], device=device),
            torch.as_tensor(self.atac_scores[index], device=device),
        )
