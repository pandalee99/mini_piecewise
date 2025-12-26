"""Minimal SGLang-style piecewise FX split + hybrid CUDA Graph runner.

This package is intentionally small:
- FX split a large forward into pieces
- Keep attention pieces eager
- Capture CUDA Graph for "safe" pieces

The focus is clarity and correctness for a minimal prototype.
"""

from .config import PiecewiseHybridConfig
from .hybrid import PiecewiseHybridModel, make_piecewise_hybrid_model

__all__ = [
    "PiecewiseHybridConfig",
    "PiecewiseHybridModel",
    "make_piecewise_hybrid_model",
]
