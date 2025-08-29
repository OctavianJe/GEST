import string
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Sequence

import numpy as np

from ._helper import aggregate, cosine
from .embedding import _UNK_TOKEN, EmbeddingModel
from .graph import GESTGraph, NodeId


class NodeComparator(ABC):
    """Abstract base - compares two *node ids* across two graphs."""

    @abstractmethod
    def compare(
        self, n1: NodeId, n2: NodeId, g1: GESTGraph, g2: GESTGraph
    ) -> float: ...


class SemanticNodeComparator(NodeComparator):
    """Node similarity using *semantic* embeddings."""

    def __init__(
        self,
        model: EmbeddingModel,
        multiplicative: bool = True,
        mean_first: bool = False,
    ) -> None:
        self.model = model
        self.multiplicative = multiplicative
        self.mean_first = mean_first

    def _vectorise(self, tokens: Iterable[str]) -> np.ndarray:
        """Convert a sequence of tokens to a vector using the embedding model."""

        vecs = [
            self.model[t] if t in self.model else self.model[_UNK_TOKEN] for t in tokens
        ]

        return aggregate(self.mean_first, vecs)

    def _sim(self, t1: Sequence[str] | str, t2: Sequence[str] | str) -> float:
        """Compute cosine similarity between two sequences of tokens."""

        def clean_and_tokenize(text: str) -> list[str]:
            translator = str.maketrans("", "", string.punctuation)
            return text.lower().translate(translator).split()

        if isinstance(t1, str):
            t1 = clean_and_tokenize(t1)

        if isinstance(t2, str):
            t2 = clean_and_tokenize(t2)

        return cosine(self._vectorise(t1), self._vectorise(t2))

    def compare(self, n1: NodeId, n2: NodeId, g1: GESTGraph, g2: GESTGraph) -> float:
        """Compare two nodes by their action and entities."""

        act_sim = self._sim(g1.action(n1), g2.action(n2))
        ent_sim = self._sim(g1.entities(n1), g2.entities(n2))

        return act_sim * ent_sim if self.multiplicative else act_sim + ent_sim
