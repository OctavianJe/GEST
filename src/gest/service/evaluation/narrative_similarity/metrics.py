from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from gest.service.evaluation.graph_matching.embedding_type_enum import EmbeddingType
from gest.service.evaluation.graph_matching.similarity import (
    SimilarityEngine,
    SimilarityService,
)
from gest.service.evaluation.graph_matching.solver import SolverType
from gest.service.evaluation.graph_matching.edge_comparator import (
    CompositeEdgeComparator,
    SpatialEdgeComparator,
    TemporalEdgeComparator,
)

from .gest_store import GestStore


class Metric(Protocol):
    name: str

    def score(
        self,
        anchor_text: str,
        candidate_text: str,
        *,
        anchor_id: str | None = None,
        candidate_id: str | None = None,
    ) -> float: ...


@dataclass(slots=True)
class GraphMatchingMetric:
    name: str
    gest_store: GestStore
    similarity_service: SimilarityService
    normalized: bool = True
    missing_value: float = 0.0

    def score(
        self,
        anchor_text: str,
        candidate_text: str,
        *,
        anchor_id: str | None = None,
        candidate_id: str | None = None,
    ) -> float:
        if not anchor_id or not candidate_id:
            return self.missing_value
        g1 = self.gest_store.get(anchor_id)
        g2 = self.gest_store.get(candidate_id)
        if g1 is None or g2 is None:
            return self.missing_value
        try:
            if self.normalized:
                return float(self.similarity_service.graph_similarity_normalized(g1, g2))
            return float(self.similarity_service.graph_similarity(g1, g2))
        except Exception:
            return self.missing_value


def build_graph_metric(
    name: str,
    gest_store: GestStore,
    *,
    solver_type: SolverType,
    embedding_type: EmbeddingType,
    use_edges: bool,
    edge_mode: str | None = None,
    normalized: bool = True,
    missing_value: float = 0.0,
) -> GraphMatchingMetric:
    edge_factory = None
    if use_edges:
        if edge_mode is None or edge_mode == "temporal_spatial":
            edge_factory = lambda model, node_cmp: CompositeEdgeComparator(  # noqa: E731
                TemporalEdgeComparator(model=model, node_cmp=node_cmp),
                SpatialEdgeComparator(model=model, node_cmp=node_cmp),
            )
        elif edge_mode == "temporal":
            edge_factory = lambda model, node_cmp: TemporalEdgeComparator(  # noqa: E731
                model=model, node_cmp=node_cmp
            )
        elif edge_mode == "spatial":
            edge_factory = lambda model, node_cmp: SpatialEdgeComparator(  # noqa: E731
                model=model, node_cmp=node_cmp
            )
        else:
            raise ValueError(
                f"Unknown edge_mode '{edge_mode}'. Expected temporal, spatial, temporal_spatial."
            )

    engine = SimilarityEngine(
        solver_type=solver_type,
        embedding_type=embedding_type,
        use_edges=use_edges,
        edge_comparator_factory=edge_factory,
    )
    service = SimilarityService(engine=engine)
    return GraphMatchingMetric(
        name=name,
        gest_store=gest_store,
        similarity_service=service,
        normalized=normalized,
        missing_value=missing_value,
    )


@dataclass(slots=True)
class SbertCosineMetric:
    name: str = "sbert_cosine"
    evaluator: "TextSimilarityEvaluator | None" = None

    def __post_init__(self) -> None:
        if self.evaluator is None:
            from gest.service.other.text_similarity.text_similarity_evaluator import (
                TextSimilarityEvaluator,
            )

            self.evaluator = TextSimilarityEvaluator()

    def score(
        self,
        anchor_text: str,
        candidate_text: str,
        *,
        anchor_id: str | None = None,
        candidate_id: str | None = None,
    ) -> float:
        return float(self.evaluator.compute_text_similarity(anchor_text, candidate_text))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


@dataclass(slots=True)
class BleuMetric:
    name: str = "bleu"
    max_order: int = 4
    smooth: bool = True

    def score(
        self,
        anchor_text: str,
        candidate_text: str,
        *,
        anchor_id: str | None = None,
        candidate_id: str | None = None,
    ) -> float:
        reference = _tokenize(anchor_text)
        hypothesis = _tokenize(candidate_text)
        if not reference or not hypothesis:
            return 0.0

        precisions = []
        for n in range(1, self.max_order + 1):
            ref_counts = Counter(_ngrams(reference, n))
            hyp_counts = Counter(_ngrams(hypothesis, n))
            overlap = sum((ref_counts & hyp_counts).values())
            total = max(sum(hyp_counts.values()), 1)
            if self.smooth:
                precisions.append((overlap + 1) / (total + 1))
            else:
                precisions.append(overlap / total if total else 0.0)

        log_precisions = sum(math.log(p) for p in precisions if p > 0)
        geo_mean = math.exp(log_precisions / self.max_order)

        ref_len = len(reference)
        hyp_len = len(hypothesis)
        if hyp_len == 0:
            return 0.0
        brevity_penalty = math.exp(1 - ref_len / hyp_len) if hyp_len < ref_len else 1.0
        return float(brevity_penalty * geo_mean)


@dataclass(slots=True)
class RougeLMetric:
    name: str = "rouge_l"

    def score(
        self,
        anchor_text: str,
        candidate_text: str,
        *,
        anchor_id: str | None = None,
        candidate_id: str | None = None,
    ) -> float:
        reference = _tokenize(anchor_text)
        hypothesis = _tokenize(candidate_text)
        if not reference or not hypothesis:
            return 0.0

        lcs_len = _lcs_length(reference, hypothesis)
        precision = lcs_len / len(hypothesis) if hypothesis else 0.0
        recall = lcs_len / len(reference) if reference else 0.0
        if precision + recall == 0:
            return 0.0
        return float((2 * precision * recall) / (precision + recall))


@dataclass(slots=True)
class BleurtMetric:
    name: str = "bleurt"
    checkpoint: str = "lucadiliello/BLEURT-20"
    backend: str = "pytorch"
    device: str | None = "cpu"
    max_length: int = 256
    _bleurt_metric: object | None = None
    _bleurt_scorer: object | None = None
    _bleurt_model: object | None = None
    _bleurt_tokenizer: object | None = None
    _device: object | None = None

    def __post_init__(self) -> None:
        backend = self.backend.lower()
        if backend != "pytorch":
            raise RuntimeError(
                "Only the PyTorch BLEURT backend is supported in this setup."
            )

        import torch
        from bleurt_pytorch import (
            BleurtForSequenceClassification,
            BleurtTokenizer,
        )

        checkpoint = _resolve_bleurt_checkpoint(self.checkpoint)
        self._bleurt_tokenizer = BleurtTokenizer.from_pretrained(checkpoint)
        self._bleurt_model = BleurtForSequenceClassification.from_pretrained(
            checkpoint
        )

        if self.device:
            self._device = torch.device(self.device)
        else:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self._bleurt_model.to(self._device)
        self._bleurt_model.eval()

    def score(
        self,
        anchor_text: str,
        candidate_text: str,
        *,
        anchor_id: str | None = None,
        candidate_id: str | None = None,
    ) -> float:
        import torch

        if self._bleurt_model is None or self._bleurt_tokenizer is None:
            raise RuntimeError("BLEURT PyTorch model is not initialized.")

        inputs = self._bleurt_tokenizer(
            [anchor_text],
            [candidate_text],
            padding="longest",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._bleurt_model(**inputs)
        scores = outputs.logits.squeeze(-1).detach().cpu().numpy()
        return float(scores[0])


def _resolve_bleurt_checkpoint(checkpoint: str) -> str:
    alias = checkpoint.strip()
    mapping = {
        "BLEURT-20": "lucadiliello/BLEURT-20",
        "bleurt-20": "lucadiliello/BLEURT-20",
        "bleurt-base-128": "Elron/bleurt-base-128",
        "bleurt-base-512": "Elron/bleurt-base-512",
        "bleurt-large-512": "Elron/bleurt-large-512",
    }
    return mapping.get(alias, alias)


def _ngrams(tokens: Iterable[str], n: int) -> Iterable[tuple[str, ...]]:
    if n <= 0:
        return []
    return zip(*(tokens[i:] for i in range(n)))


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = np.zeros(len(short) + 1, dtype=int)
    curr = np.zeros(len(short) + 1, dtype=int)
    for token in long:
        curr[0] = 0
        for j, tok in enumerate(short, start=1):
            if token == tok:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return int(prev[-1])
