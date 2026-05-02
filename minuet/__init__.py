"""Minuet — paired-multiome integration model.

Two-loss decomposition:
  - NT-Xent contrastive  -> RNA<->ATAC cross-modal alignment
  - CMI-BI               -> within-modality batch-conditional cluster geometry

The two losses target distinct axes of the multiome integration problem
and contribute independent improvements (cf. 2026-05-01 v61 ablation).
"""

from .factory import build_model_from_section, compute_total_losses, load_compatible_state_dict, load_model_from_checkpoint
from .losses import MinuetLosses
from .model import Minuet, MinuetConfig

__version__ = "0.1.0"

__all__ = [
    "Minuet",
    "MinuetConfig",
    "MinuetLosses",
    "build_model_from_section",
    "compute_total_losses",
    "load_compatible_state_dict",
    "load_model_from_checkpoint",
]
