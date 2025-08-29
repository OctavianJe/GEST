import functools
import itertools
import math
from typing import Callable, Optional

import numpy as np

from .edge_comparator import (
    CompositeEdgeComparator,
    EdgeComparator,
    SpatialEdgeComparator,
    TemporalEdgeComparator,
)
from .embedding import EmbeddingModel, EmbeddingService, EmbeddingType
from .graph import GESTGraph, TimelineEdge
from .node_comparator import NodeComparator, SemanticNodeComparator
from .solver import Solver, SolverFactory, SolverType


class SimilarityEngine:
    """Engine to compute similarity between two GEST graphs using embeddings."""

    def __init__(
        self,
        embedding_type: EmbeddingType = EmbeddingType.GLOVE50,
        multiplicative_nodes: bool = True,
        use_edges: bool = True,
        solver_type: SolverType = SolverType.SPECTRAL,
        node_comparator_factory: (
            Optional[
                Callable[
                    [EmbeddingModel],
                    NodeComparator,
                ]
            ]
        ) = None,
        edge_comparator_factory: (
            Optional[
                Callable[
                    [
                        EmbeddingModel,
                        NodeComparator,
                    ],
                    EdgeComparator,
                ]
            ]
        ) = None,
    ) -> None:
        self.model: EmbeddingModel = EmbeddingService().load_embedding(embedding_type)
        self.solver: Solver = SolverFactory().get_solver(solver_type)

        self.node_cmp: NodeComparator = (
            node_comparator_factory(self.model)
            if node_comparator_factory
            else SemanticNodeComparator(
                model=self.model,
                multiplicative=multiplicative_nodes,
            )
        )

        if use_edges:
            self.edge_cmp: Optional[EdgeComparator] = (
                edge_comparator_factory(self.model, self.node_cmp)
                if edge_comparator_factory
                else CompositeEdgeComparator(
                    TemporalEdgeComparator(
                        model=self.model,
                        node_cmp=self.node_cmp,
                    ),
                    SpatialEdgeComparator(
                        model=self.model,
                        node_cmp=self.node_cmp,
                    ),
                )
            )
        else:
            self.edge_cmp = None

    def _association_matrix(
        self, g1: GESTGraph, g2: GESTGraph
    ) -> tuple[np.ndarray, int, int]:
        """Construct the association matrix for two GEST graphs."""

        n1, n2 = list(g1.action_nodes), list(g2.action_nodes)
        if not n1 or not n2:
            return np.zeros((0, 0)), 0, 0

        prod = list(itertools.product(n1, n2))
        L = list(itertools.product(prod, prod))
        dim = int(math.sqrt(len(L)))
        M = np.zeros((dim, dim), dtype=float)

        for idx, ((a1, b1), (a2, b2)) in enumerate(L):
            i, j = divmod(idx, dim)

            if a1 == a2 and b1 == b2:
                M[i, j] = self.node_cmp.compare(a1, b1, g1, g2)
            elif a1 != a2 and b1 != b2:
                val = (
                    self.edge_cmp.compare(
                        e1=TimelineEdge(src=a1, dst=a2),
                        e2=TimelineEdge(src=b1, dst=b2),
                        g1=g1,
                        g2=g2,
                    )
                    if self.edge_cmp
                    else 0.0
                )
                M[i, j] = val
            else:
                M[i, j] = 0.0

        return M, len(n1), len(n2)

    def similarity(self, g1: GESTGraph, g2: GESTGraph) -> float:
        """Compute similarity between two GEST graphs."""

        M, n1, n2 = self._association_matrix(g1, g2)
        x = self.solver.solve(M, n1, n2)
        return float((x.T @ M @ x)[0, 0])

    def similarity_normalized(self, g1: GESTGraph, g2: GESTGraph) -> float:
        """Compute normalized similarity between two GEST graphs."""

        s12 = self.similarity(g1, g2)
        s11 = self.similarity(g1, g1)
        s22 = self.similarity(g2, g2)
        return 0.0 if min(s11, s22) == 0 else s12 / np.sqrt(s11 * s22)


class SimilarityService:
    """Service to compute and cache similarity between GEST graphs."""

    def __init__(self, engine: SimilarityEngine = SimilarityEngine()) -> None:
        self._engine = engine

    def graph_similarity(self, g1: GESTGraph, g2: GESTGraph) -> float:
        """Compute similarity between two GEST graphs, caching results for efficiency."""

        return self._graph_similarity_cached(g1, g2)

    def graph_similarity_normalized(self, g1: GESTGraph, g2: GESTGraph) -> float:
        """Compute normalized similarity between two GEST graphs, caching results for efficiency."""

        return self._graph_similarity_normalized_cached(g1, g2)

    @functools.lru_cache(maxsize=512)
    def _graph_similarity_cached(self, g1: GESTGraph, g2: GESTGraph) -> float:
        """Compute similarity between two GEST graphs, caching the result."""

        return self._engine.similarity(g1, g2)

    @functools.lru_cache(maxsize=512)
    def _graph_similarity_normalized_cached(
        self, g1: GESTGraph, g2: GESTGraph
    ) -> float:
        """Compute normalized similarity between two GEST graphs, caching the result."""

        return self._engine.similarity_normalized(g1, g2)
