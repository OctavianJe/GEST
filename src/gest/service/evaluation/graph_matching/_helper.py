from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.linalg import norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Stable cosine sim with *0 fallback* for zero vectors."""
    denom = norm(a) * norm(b)
    return float((a @ b) / denom) if denom else 0.0


def aggregate(mean_first: bool, vecs: Sequence[np.ndarray]) -> np.ndarray:
    """Return either the first vector or the mean of all."""
    return vecs[0] if mean_first else np.mean(vecs, axis=0, keepdims=False)
