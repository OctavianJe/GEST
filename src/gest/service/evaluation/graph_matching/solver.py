from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np
import pygmtools as pygm

pygm.BACKEND = "numpy"


class SolverType(Enum):
    """Enumerates built-in solvers bundled with *graph_match*."""

    SPECTRAL = auto()
    NGM = auto()


class Solver(ABC):
    """Strategy interface - converts *association matrix* to matching vector."""

    @abstractmethod
    def solve(
        self, M: np.ndarray, n1: int | None = None, n2: int | None = None
    ) -> np.ndarray: ...


class SpectralSolver(Solver):
    """Classic *spectral matching* implementation."""

    def solve(self, M: np.ndarray, n1: int | None = None, n2: int | None = None):
        eigvals, eigvecs = np.linalg.eigh(M)
        vec = eigvecs[:, np.argmax(eigvals)].copy()
        vec[vec < 0] *= -1  # make non‑negative
        vec[np.isclose(vec, 0)] = 0
        return (vec > 1e-8).astype(float).reshape(-1, 1)


class NGMSolver(Solver):
    """*Neural Graph Matching* via ``pygmtools`` (Hungarian post-proc)."""

    def solve(self, M: np.ndarray, n1: int | None = None, n2: int | None = None):
        X = pygm.ngm(M, n1, n2, return_network=False, pretrain="voc")
        X = pygm.hungarian(X)
        return X.reshape(-1, 1)


class SolverFactory:
    """Factory class to create solver instances based on `SolverType`."""

    @staticmethod
    def get_solver(kind: SolverType) -> Solver:
        """Return a solver instance based on the specified `SolverType`."""

        if kind == SolverType.SPECTRAL:
            return SpectralSolver()

        if kind == SolverType.NGM:
            return NGMSolver()

        raise TypeError(
            f"Unsupported SolverType: {kind}. Available: {list(SolverType)}"
        )
