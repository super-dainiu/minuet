"""Minuet: compact population-level paired RNA--ATAC representation learning."""

from .api import Minuet
from .losses import MinuetLosses
from .model import MinuetConfig, MinuetEncoder, MinuetModule

__version__ = "0.3.0"

__all__ = [
    "Minuet",
    "MinuetModule",
    "MinuetEncoder",
    "MinuetConfig",
    "MinuetLosses",
]
