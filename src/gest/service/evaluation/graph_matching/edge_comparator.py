from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from ._helper import aggregate, cosine
from .embedding import _UNK_TOKEN, EmbeddingModel
from .graph import (
    ActionId,
    AllenRelation,
    GESTGraph,
    QualitativeSpatialReasoning,
    SpatialAction,
    SpatialMarker,
    SpatialMarkerId,
    TemporalAction,
    TemporalMarker,
    TemporalMarkerId,
    TimelineEdge,
)
from .node_comparator import NodeComparator


class EdgeComparator(ABC):
    """Abstract base - compares two *node-pairs* (edges) across graphs."""

    @abstractmethod
    def compare(
        self, e1: TimelineEdge, e2: TimelineEdge, g1: GESTGraph, g2: GESTGraph
    ) -> float: ...


class TemporalEdgeComparator(EdgeComparator):
    """Edge similarity using *temporal markers*."""

    def __init__(self, model: EmbeddingModel, node_cmp: NodeComparator):
        self.model = model
        self.node_cmp = node_cmp

    def _markers(
        self,
        a: ActionId,
        b: ActionId,
        g: GESTGraph,
    ) -> set[AllenRelation]:
        """Get the set of temporal markers between two actions in a GESTGraph."""

        ta_map: Dict[ActionId, TemporalAction] = g.temporal.actions
        tm_map: Dict[TemporalMarkerId, TemporalMarker] = g.temporal.temporal_markers

        marks: set[AllenRelation] = set()

        if b in ta_map.get(a, TemporalAction()).next:
            marks.add(AllenRelation.BEFORE)

        if a in ta_map.get(b, TemporalAction()).next:
            marks.add(AllenRelation.AFTER)

        def _collect(src: ActionId, tgt: ActionId) -> None:
            for rel in ta_map.get(src, TemporalAction()).relations:
                if rel.target != tgt:
                    continue
                tm = tm_map.get(rel.type)
                if tm and isinstance(tm.type, AllenRelation):
                    marks.add(tm.type)

        _collect(a, b)
        _collect(b, a)

        return marks

    def _semantic(
        self,
        m1: set[AllenRelation],
        m2: set[AllenRelation],
    ) -> float:
        """Compute cosine similarity between two sets of markers."""

        if not m1 or not m2:
            return 0.0

        def _get_avg_vector(markers: set[AllenRelation]) -> np.ndarray:
            """Converts a set of relations to a single average vector using the embedding model."""

            all_vecs = []
            for rel in markers:
                tokens = rel.value.replace("_", " ").split()
                vecs = [
                    self.model[t] if t in self.model else self.model[_UNK_TOKEN]
                    for t in tokens
                ]

                if vecs:
                    all_vecs.append(aggregate(False, vecs))

            if all_vecs:
                return np.mean(all_vecs, axis=0)

            if not self.model.index_to_key:
                return np.array([])

            key = self.model.index_to_key[0]
            dim = self.model[key].shape[0]
            return np.zeros(dim)

        v1 = _get_avg_vector(m1)
        v2 = _get_avg_vector(m2)

        return cosine(v1, v2)

    def compare(
        self,
        e1: TimelineEdge,
        e2: TimelineEdge,
        g1: GESTGraph,
        g2: GESTGraph,
    ):
        """Compare two edges by their markers and node similarities."""

        (a1, b1), (a2, b2) = e1, e2
        m1, m2 = self._markers(a1, b1, g1), self._markers(a2, b2, g2)

        if not m1 or not m2:
            return 0.0

        edge_sem = self._semantic(m1, m2)

        n1 = self.node_cmp.compare(a1, a2, g1, g2)
        n2 = self.node_cmp.compare(b1, b2, g1, g2)

        return max(0.0, n1 * n2 * edge_sem)


class SpatialEdgeComparator(EdgeComparator):
    """Edge similarity using *spatial markers*."""

    def __init__(self, model: EmbeddingModel, node_cmp: NodeComparator):
        self.model = model
        self.node_cmp = node_cmp

    def _markers(
        self,
        a: ActionId,
        b: ActionId,
        g: GESTGraph,
    ) -> set[QualitativeSpatialReasoning]:
        """Get the set of spatial markers between two actions in a GESTGraph."""

        sa_map: dict[ActionId, SpatialAction] = g.spatial.actions
        sm_map: dict[SpatialMarkerId, SpatialMarker] = g.spatial.spatial_markers

        marks: set[QualitativeSpatialReasoning] = set()

        def _collect(src: ActionId, tgt: ActionId) -> None:
            for rel in sa_map.get(src, SpatialAction()).relations:
                if rel.target != tgt:
                    continue
                m = sm_map.get(rel.type)
                if m and m.type:
                    marks.add(m.type)

        _collect(a, b)
        _collect(b, a)

        return marks

    def _semantic(
        self,
        m1: set[QualitativeSpatialReasoning],
        m2: set[QualitativeSpatialReasoning],
    ) -> float:
        """Compute cosine similarity between two sets of markers."""

        if not m1 or not m2:
            return 0.0

        def _get_avg_vector(markers: set[QualitativeSpatialReasoning]) -> np.ndarray:
            """Converts a set of relations to a single average vector using the embedding model."""

            all_vecs = []
            for phrase in markers:
                tokens = phrase.replace("_", " ").split()
                vecs = [
                    self.model[t] if t in self.model else self.model[_UNK_TOKEN]
                    for t in tokens
                ]
                if vecs:
                    all_vecs.append(aggregate(False, vecs))

            if all_vecs:
                return np.mean(all_vecs, axis=0)

            if not self.model.index_to_key:
                return np.array([])

            key = self.model.index_to_key[0]
            dim = self.model[key].shape[0]
            return np.zeros(dim)

        v1 = _get_avg_vector(m1)
        v2 = _get_avg_vector(m2)

        return cosine(v1, v2)

    def compare(
        self,
        e1: TimelineEdge,
        e2: TimelineEdge,
        g1: GESTGraph,
        g2: GESTGraph,
    ):
        """Compare two edges by their markers and node similarities."""

        (a1, b1), (a2, b2) = e1, e2
        m1, m2 = self._markers(a1, b1, g1), self._markers(a2, b2, g2)

        if not m1 or not m2:
            return 0.0

        edge_sem = self._semantic(m1, m2)

        n1 = self.node_cmp.compare(a1, a2, g1, g2)
        n2 = self.node_cmp.compare(b1, b2, g1, g2)

        return max(0.0, n1 * n2 * edge_sem)


class CompositeEdgeComparator(EdgeComparator):
    """Wrapper that merges temporal and spatial edge comparators."""

    def __init__(self, *comparators: EdgeComparator):
        self._cmps = comparators

    def compare(self, e1, e2, g1, g2) -> float:
        return max(c.compare(e1, e2, g1, g2) for c in self._cmps)
