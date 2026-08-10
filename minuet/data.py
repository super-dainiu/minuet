"""Raw-count batching for Minuet fitting and frozen-query encoding."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterator, Sequence

import numpy as np
import scipy.sparse as sp
import torch


@dataclass(frozen=True)
class PopulationBatchConfig:
    contexts_per_batch: int = 2
    donors_per_context: int = 4
    cells_per_block: int = 12
    batches_per_epoch: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        if min(
            self.contexts_per_batch,
            self.donors_per_context,
            self.cells_per_block,
            self.batches_per_epoch,
        ) <= 0:
            raise ValueError("population batch settings must be positive")
        if self.donors_per_context < 2 or self.cells_per_block < 2:
            raise ValueError("population batches require at least two donors and cells")


class PopulationBatchSampler:
    """Deterministic equal-size donor-by-context block sampler."""

    def __init__(
        self,
        donor: Sequence[Hashable],
        context: Sequence[Hashable],
        config: PopulationBatchConfig,
    ) -> None:
        if len(donor) != len(context) or not donor:
            raise ValueError("donor and context must be non-empty and row aligned")
        self.config = config
        blocks: dict[Hashable, dict[Hashable, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, (donor_value, context_value) in enumerate(zip(donor, context)):
            blocks[context_value][donor_value].append(index)
        self._eligible = {
            context_value: {
                donor_value: tuple(indices)
                for donor_value, indices in donor_blocks.items()
                if len(indices) >= config.cells_per_block
            }
            for context_value, donor_blocks in blocks.items()
        }
        self._eligible = {
            context_value: donor_blocks
            for context_value, donor_blocks in self._eligible.items()
            if len(donor_blocks) >= 2
        }
        self._contexts = tuple(sorted(self._eligible, key=str))
        if len(self._contexts) < config.contexts_per_batch:
            raise ValueError(
                "insufficient donor-by-context support for production population batches"
            )

    def epoch(self, epoch: int) -> Iterator[list[int]]:
        generator = torch.Generator(device="cpu").manual_seed(
            self.config.seed + 1_000_003 * int(epoch)
        )
        for _ in range(self.config.batches_per_epoch):
            context_order = torch.randperm(
                len(self._contexts), generator=generator
            )[: self.config.contexts_per_batch]
            batch: list[int] = []
            for context_position in context_order.tolist():
                context_value = self._contexts[context_position]
                blocks = self._eligible[context_value]
                donors = tuple(sorted(blocks, key=str))
                donor_order = torch.randperm(len(donors), generator=generator)[
                    : min(self.config.donors_per_context, len(donors))
                ]
                for donor_position in donor_order.tolist():
                    indices = blocks[donors[donor_position]]
                    chosen = torch.randperm(len(indices), generator=generator)[
                        : self.config.cells_per_block
                    ]
                    batch.extend(indices[position] for position in chosen.tolist())
            if not batch or len(batch) != len(set(batch)):
                raise RuntimeError("population sampler produced an invalid batch")
            yield batch


def dense_rows(
    matrix: sp.csr_matrix,
    rows: np.ndarray | Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Materialize one raw-count batch on the requested device."""
    return torch.as_tensor(
        matrix[np.asarray(rows, dtype=np.int64)].toarray(),
        dtype=torch.float32,
        device=device,
    )
