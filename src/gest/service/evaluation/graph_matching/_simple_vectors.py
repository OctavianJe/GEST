from typing import Dict, Sequence

import numpy as np


class SimpleVectors:
    """Lightweight stand-in for *gensim KeyedVectors*."""

    def __init__(self, kv: Dict[str, np.ndarray]):
        self._kv = kv
        self.index_to_key = list(kv.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        return self._kv[key]

    def __contains__(self, key: str) -> bool:
        return key in self._kv

    def __iter__(self):
        return iter(self._kv)

    def add_vectors(self, keys: Sequence[str], vectors):
        for k, v in zip(keys, vectors):
            self._kv[k] = np.asarray(v, dtype=np.float32)
            self.index_to_key.append(k)
