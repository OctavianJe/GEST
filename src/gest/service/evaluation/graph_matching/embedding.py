from __future__ import annotations

from abc import abstractmethod
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np

from ._embedding_loader import EmbeddingLoader
from .embedding_type_enum import EmbeddingType

_UNK_TOKEN = "<UNK>"


class EmbeddingModel(Protocol):
    @abstractmethod
    def __getitem__(self, key: str) -> np.ndarray: ...

    @abstractmethod
    def add_vectors(self, keys: Sequence[str], vectors): ...

    @property
    def index_to_key(self) -> Sequence[str]: ...

    def __contains__(self, key: str) -> bool: ...


class EmbeddingService:
    @lru_cache(maxsize=4)
    def _download(self, name: EmbeddingType) -> EmbeddingModel:
        """Download and return a read-only *word-embedding* model."""
        return EmbeddingLoader().load_vectors(name)

    def load_embedding(
        self, name: EmbeddingType = EmbeddingType.GLOVE50, add_unk: bool = True
    ) -> EmbeddingModel:
        """Return a read-only *word-embedding* model ready for cosine distance."""
        model = self._download(name)

        # Add an <UNK> token if requested and not present in the model.
        if add_unk and _UNK_TOKEN not in model.index_to_key:
            dim = model[next(iter(model.index_to_key))].shape[0]
            rng = np.random.default_rng(seed=42)
            unk = rng.uniform(-1.0, 1.0, size=(dim,))
            model.add_vectors([_UNK_TOKEN], [unk])

        return model
