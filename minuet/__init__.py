"""Minuet: paired RNA--ATAC integration with a scvi-style API."""

from .api import Minuet
from .losses import MinuetLosses
from .model import MinuetConfig, MinuetModule

__version__ = "0.2.0"

__all__ = [
    "Minuet",
    "MinuetModule",
    "MinuetConfig",
    "MinuetLosses",
]
