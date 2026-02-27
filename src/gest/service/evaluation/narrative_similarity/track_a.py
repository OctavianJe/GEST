from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from gest.service.evaluation.graph_matching.embedding_type_enum import EmbeddingType
from gest.service.evaluation.graph_matching.solver import SolverType

from .gest_store import GestStore
from .metrics import (
    BleuMetric,
    BleurtMetric,
    Metric,
    RougeLMetric,
    SbertCosineMetric,
    build_graph_metric,
)

DATASET_ROOT = Path("miscellaneous") / "datasets" / "Narrative Similarity Task"
RESULTS_ROOT = Path("results") / "Narrative Similarity Task"
SUBMISSIONS_ROOT = RESULTS_ROOT / "submissions"
DEFAULT_DATASETS: Dict[str, Path] = {
    "sample": DATASET_ROOT / "sample" / "sample_track_a.jsonl",
    "dev": DATASET_ROOT / "development" / "dev_track_a.jsonl",
    "synthetic": DATASET_ROOT
    / "synthetic-training"
    / "synthetic_data_for_classification.jsonl",
    "test": DATASET_ROOT / "test" / "test_track_a.jsonl",
}

EDGE_MODES = ("temporal", "spatial", "temporal_spatial")
GRAPH_SOLVERS = {
    "spectral": SolverType.SPECTRAL,
    "ngm": SolverType.NGM,
}
GRAPH_EMBEDDINGS = {
    "glove50": EmbeddingType.GLOVE50,
    "glove300": EmbeddingType.GLOVE300,
    "w2v_google": EmbeddingType.W2V_GOOGLE,
}
TEXT_METRICS = ("bleu", "rouge_l", "sbert_cosine", "bleurt")
LLM_METRICS = ("llm_chosen", "llm_score")


def _graph_metric_name(solver_name: str, embedding_name: str, edge_mode: str) -> str:
    return f"gest_{solver_name}_{embedding_name}_{edge_mode}"


def _all_graph_metric_names() -> List[str]:
    names: List[str] = []
    for solver in GRAPH_SOLVERS:
        for embedding in GRAPH_EMBEDDINGS:
            for edge_mode in EDGE_MODES:
                names.append(_graph_metric_name(solver, embedding, edge_mode))
    return names


KNOWN_METRICS = (
    [
        "gest_spectral_glove300",
        "gest_ngm_glove300",
        "gest_spectral_glove300_no_edges",
        "sbert_cosine",
    ]
    + list(TEXT_METRICS)
    + list(LLM_METRICS)
    + _all_graph_metric_names()
)


@dataclass(frozen=True, slots=True)
class TrackARecord:
    split_id: str
    row_index: int
    anchor_text: str
    text_a: str
    text_b: str
    text_a_is_closer: Optional[bool]
    anchor_id: str
    text_a_id: str
    text_b_id: str


@dataclass(slots=True)
class PrecomputedCandidateMetric:
    name: str
    candidate_scores: Dict[str, float]
    missing_value: float = 0.0

    def score(
        self,
        anchor_text: str,
        candidate_text: str,
        *,
        anchor_id: str | None = None,
        candidate_id: str | None = None,
    ) -> float:
        if not candidate_id:
            return self.missing_value
        return float(self.candidate_scores.get(candidate_id, self.missing_value))


def _row_id(split_id: str, row_index: int, role: str) -> str:
    return f"{split_id}:{row_index}:{role}"


def _normalize_text(text: Optional[str]) -> str:
    if not isinstance(text, str) or not text:
        return ""
    return text.replace("\n", " ").strip()


def load_track_a_records(
    path: Path,
    *,
    drop_invalid: bool = False,
    logger: Optional[Callable[[str], None]] = None,
) -> List[TrackARecord]:
    if not path.exists():
        raise FileNotFoundError(f"Track A dataset not found: {path}")

    split_id = f"{path.parent.name}/{path.stem}"
    records: List[TrackARecord] = []
    skipped = 0

    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            anchor_text = _normalize_text(payload.get("anchor_text", ""))
            text_a = _normalize_text(payload.get("text_a", ""))
            text_b = _normalize_text(payload.get("text_b", ""))
            if drop_invalid and (not anchor_text or not text_a or not text_b):
                skipped += 1
                continue
            label = payload.get("text_a_is_closer")
            if isinstance(label, str):
                label = label.strip().lower() == "true"
            records.append(
                TrackARecord(
                    split_id=split_id,
                    row_index=idx,
                    anchor_text=anchor_text,
                    text_a=text_a,
                    text_b=text_b,
                    text_a_is_closer=label,
                    anchor_id=_row_id(split_id, idx, "anchor_text"),
                    text_a_id=_row_id(split_id, idx, "text_a"),
                    text_b_id=_row_id(split_id, idx, "text_b"),
                )
            )
    if drop_invalid and skipped and logger:
        logger(f"Skipped {skipped} invalid records from {path}")
    return records


def resolve_dataset_path(arg: str) -> Path:
    if arg in DEFAULT_DATASETS:
        return DEFAULT_DATASETS[arg]
    return Path(arg)


def build_metric_registry(
    gest_store: GestStore,
    *,
    bleurt_checkpoint: str,
    bleurt_backend: str,
    bleurt_device: str | None,
    bleurt_max_length: int,
    requested: Optional[Iterable[str]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Metric], Dict[str, str]]:
    metrics: Dict[str, Metric] = {}
    missing: Dict[str, str] = {}
    requested_set = set(requested or [])
    load_all = not requested_set

    for solver_name, solver_type in GRAPH_SOLVERS.items():
        for embedding_name, embedding_type in GRAPH_EMBEDDINGS.items():
            for edge_mode in EDGE_MODES:
                name = _graph_metric_name(solver_name, embedding_name, edge_mode)
                if load_all or name in requested_set:
                    if logger:
                        logger(f"Initializing graph metric '{name}'")
                    metrics[name] = build_graph_metric(
                        name=name,
                        gest_store=gest_store,
                        solver_type=solver_type,
                        embedding_type=embedding_type,
                        use_edges=True,
                        edge_mode=edge_mode,
                    )

    if load_all or "gest_spectral_glove300" in requested_set:
        if logger:
            logger("Initializing graph metric 'gest_spectral_glove300'")
        metrics["gest_spectral_glove300"] = build_graph_metric(
            name="gest_spectral_glove300",
            gest_store=gest_store,
            solver_type=SolverType.SPECTRAL,
            embedding_type=EmbeddingType.GLOVE300,
            use_edges=True,
            edge_mode="temporal_spatial",
        )
    if load_all or "gest_ngm_glove300" in requested_set:
        if logger:
            logger("Initializing graph metric 'gest_ngm_glove300'")
        metrics["gest_ngm_glove300"] = build_graph_metric(
            name="gest_ngm_glove300",
            gest_store=gest_store,
            solver_type=SolverType.NGM,
            embedding_type=EmbeddingType.GLOVE300,
            use_edges=True,
            edge_mode="temporal_spatial",
        )
    if load_all or "gest_spectral_glove300_no_edges" in requested_set:
        if logger:
            logger("Initializing graph metric 'gest_spectral_glove300_no_edges'")
        metrics["gest_spectral_glove300_no_edges"] = build_graph_metric(
            name="gest_spectral_glove300_no_edges",
            gest_store=gest_store,
            solver_type=SolverType.SPECTRAL,
            embedding_type=EmbeddingType.GLOVE300,
            use_edges=False,
        )

    if load_all or "sbert_cosine" in requested_set:
        if logger:
            logger("Initializing text metric 'sbert_cosine'")
        metrics["sbert_cosine"] = SbertCosineMetric()
    if load_all or "bleu" in requested_set:
        if logger:
            logger("Initializing text metric 'bleu'")
        metrics["bleu"] = BleuMetric()
    if load_all or "rouge_l" in requested_set:
        if logger:
            logger("Initializing text metric 'rouge_l'")
        metrics["rouge_l"] = RougeLMetric()

    if load_all or "bleurt" in requested_set:
        try:
            if logger:
                logger("Initializing text metric 'bleurt'")
            metrics["bleurt"] = BleurtMetric(
                checkpoint=bleurt_checkpoint,
                backend=bleurt_backend,
                device=bleurt_device,
                max_length=bleurt_max_length,
            )
        except Exception as exc:
            missing["bleurt"] = str(exc)
            if logger:
                logger(f"BLEURT unavailable: {exc}")

    return metrics, missing


def build_feature_frame(
    records: Sequence[TrackARecord],
    metrics: Sequence[Metric],
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    rows: List[dict] = []
    iterator = records
    if show_progress:
        iterator = tqdm(records, desc="Scoring", total=len(records))

    if show_progress:
        metric_names = ", ".join(m.name for m in metrics)
        tqdm.write(f"Scoring metrics: {metric_names}")

    for idx, record in enumerate(iterator, start=1):
        row = {
            "split_id": record.split_id,
            "row_index": record.row_index,
        }
        for metric in metrics:
            if show_progress and idx == 1:
                tqdm.write(f"Computing metric: {metric.name}")
            row[f"{metric.name}_a"] = metric.score(
                record.anchor_text,
                record.text_a,
                anchor_id=record.anchor_id,
                candidate_id=record.text_a_id,
            )
            row[f"{metric.name}_b"] = metric.score(
                record.anchor_text,
                record.text_b,
                anchor_id=record.anchor_id,
                candidate_id=record.text_b_id,
            )
        if record.text_a_is_closer is not None:
            row["label"] = bool(record.text_a_is_closer)
        rows.append(row)

    return pd.DataFrame(rows)


def _log_step(message: str) -> None:
    tqdm.write(f"[track_a] {message}")


def _score_metric(
    records: Sequence[TrackARecord],
    metric: Metric,
    *,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Optional[bool]]]:
    scores_a: List[float] = []
    scores_b: List[float] = []
    labels: List[Optional[bool]] = []
    iterator = records
    if show_progress:
        iterator = tqdm(
            records,
            desc=f"Scoring {metric.name}",
            total=len(records),
        )
    for record in iterator:
        scores_a.append(
            metric.score(
                record.anchor_text,
                record.text_a,
                anchor_id=record.anchor_id,
                candidate_id=record.text_a_id,
            )
        )
        scores_b.append(
            metric.score(
                record.anchor_text,
                record.text_b,
                anchor_id=record.anchor_id,
                candidate_id=record.text_b_id,
            )
        )
        labels.append(record.text_a_is_closer)
    return (
        np.asarray(scores_a, dtype=float),
        np.asarray(scores_b, dtype=float),
        labels,
    )


def _filter_records_with_gest(
    records: Sequence[TrackARecord],
    gest_store: GestStore,
    *,
    logger: Optional[Callable[[str], None]] = None,
) -> List[TrackARecord]:
    filtered: List[TrackARecord] = []
    skipped = 0
    for record in records:
        if (
            gest_store.get(record.anchor_id) is None
            or gest_store.get(record.text_a_id) is None
            or gest_store.get(record.text_b_id) is None
        ):
            skipped += 1
            continue
        filtered.append(record)
    if skipped and logger:
        logger(f"Skipped {skipped} records without GEST graphs")
    return filtered


def _parse_llm_model_output(
    model_output: object,
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    def _extract(payload: object) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        if not isinstance(payload, dict):
            return None, None, None
        chosen = payload.get("chosen_text")
        similarity_a = payload.get("similarity_A")
        similarity_b = payload.get("similarity_B")

        parsed_chosen: Optional[str] = None
        if isinstance(chosen, str):
            u = chosen.strip().upper()
            if u in {"A", "B"}:
                parsed_chosen = u

        parsed_a = float(similarity_a) if isinstance(similarity_a, (int, float)) else None
        parsed_b = float(similarity_b) if isinstance(similarity_b, (int, float)) else None
        return parsed_chosen, parsed_a, parsed_b

    if isinstance(model_output, dict):
        return _extract(model_output)

    if not isinstance(model_output, str):
        return None, None, None

    raw = model_output.strip()
    try:
        payload = json.loads(raw)
        chosen, similarity_a, similarity_b = _extract(payload)
        if chosen is not None or similarity_a is not None or similarity_b is not None:
            return chosen, similarity_a, similarity_b
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            chosen, similarity_a, similarity_b = _extract(payload)
            if chosen is not None or similarity_a is not None or similarity_b is not None:
                return chosen, similarity_a, similarity_b
        except Exception:
            pass

    chosen_match = re.search(r'"chosen_text"\s*:\s*"([ABab])"', raw)
    chosen = chosen_match.group(1).upper() if chosen_match else None
    sim_a_match = re.search(r'"similarity_A"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
    sim_b_match = re.search(r'"similarity_B"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
    similarity_a = float(sim_a_match.group(1)) if sim_a_match else None
    similarity_b = float(sim_b_match.group(1)) if sim_b_match else None
    return chosen, similarity_a, similarity_b


def load_llm_track_a_outputs(
    path: Path,
    *,
    logger: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"LLM output file not found: {path}")

    rows: List[dict] = []
    parse_errors = 0
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            chosen, similarity_a, similarity_b = _parse_llm_model_output(
                payload.get("model_output")
            )
            if chosen is None:
                parse_errors += 1
            rows.append(
                {
                    "row_index": idx,
                    "anchor_text": _normalize_text(payload.get("anchor_text", "")),
                    "text_a": _normalize_text(payload.get("text_a", "")),
                    "text_b": _normalize_text(payload.get("text_b", "")),
                    "chosen_text": chosen,
                    "similarity_A": similarity_a,
                    "similarity_B": similarity_b,
                }
            )
    if logger:
        logger(
            f"Loaded {len(rows)} LLM rows from {path} "
            f"(chosen_text parse errors: {parse_errors})"
        )
    return pd.DataFrame(rows)


def _assert_llm_alignment(
    records: Sequence[TrackARecord],
    llm_df: pd.DataFrame,
) -> None:
    if len(records) != len(llm_df):
        raise ValueError(
            f"Row count mismatch: dataset={len(records)} vs llm={len(llm_df)}"
        )
    for idx, record in enumerate(records):
        llm_row = llm_df.iloc[idx]
        if (
            record.anchor_text != llm_row["anchor_text"]
            or record.text_a != llm_row["text_a"]
            or record.text_b != llm_row["text_b"]
        ):
            raise ValueError(
                f"LLM row {idx + 1} does not align with dataset text fields."
            )


def _build_llm_hybrid_features(
    *,
    d_gest: np.ndarray,
    llm_df: pd.DataFrame,
    llm_mode: str,
    include_confidence: bool,
    include_interaction: bool,
) -> Tuple[np.ndarray, List[str]]:
    if llm_mode == "chosen":
        d_llm = np.zeros(len(llm_df), dtype=float)
        chosen = llm_df["chosen_text"].fillna("")
        d_llm[chosen == "A"] = 1.0
        d_llm[chosen == "B"] = -1.0
        features = [d_gest, d_llm]
        names = ["d_gest", "d_llm_choice"]
    elif llm_mode == "score":
        sim_a = llm_df["similarity_A"].to_numpy(dtype=float)
        sim_b = llm_df["similarity_B"].to_numpy(dtype=float)
        sim_a = np.nan_to_num(np.clip(sim_a, 0.0, 100.0), nan=0.0)
        sim_b = np.nan_to_num(np.clip(sim_b, 0.0, 100.0), nan=0.0)
        d_llm = (sim_a - sim_b) / 100.0
        features = [d_gest, d_llm]
        names = ["d_gest", "d_llm_score"]
        if include_confidence:
            confidence = np.abs(d_llm)
            features.append(confidence)
            names.append("llm_confidence")
    else:
        raise ValueError("llm_mode must be 'chosen' or 'score'.")

    if include_interaction:
        interaction = features[0] * features[1]
        features.append(interaction)
        names.append(f"{names[0]}_x_{names[1]}")

    X = np.vstack(features).T
    return X, names


def _fit_logreg_feature_matrix(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float,
    class_weight: str | None,
    max_iter: int = 5000,
) -> dict:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std

    model = LogisticRegression(
        class_weight=class_weight,
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
    )
    model.fit(Xs, y)
    return {
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "C": C,
    }


def _predict_logreg_feature_matrix(X: np.ndarray, payload: dict) -> np.ndarray:
    coef = np.asarray(payload["coef"], dtype=float)
    intercept = float(payload["intercept"])
    mean = np.asarray(payload["mean"], dtype=float)
    std = np.asarray(payload["std"], dtype=float)
    Xs = (X - mean) / std
    logits = Xs @ coef + intercept
    return logits >= 0.0


def _linear_weights_from_standardized_logreg(payload: dict) -> Tuple[np.ndarray, float]:
    coef = np.asarray(payload["coef"], dtype=float)
    intercept = float(payload["intercept"])
    mean = np.asarray(payload["mean"], dtype=float)
    std = np.asarray(payload["std"], dtype=float)
    raw_coef = coef / std
    raw_bias = float(intercept - np.sum(coef * mean / std))
    return raw_coef, raw_bias


def build_llm_pair_metrics(
    records: Sequence[TrackARecord],
    llm_df: pd.DataFrame,
) -> Dict[str, Metric]:
    _assert_llm_alignment(records, llm_df)

    chosen_scores: Dict[str, float] = {}
    score_scores: Dict[str, float] = {}
    for idx, record in enumerate(records):
        row = llm_df.iloc[idx]
        chosen = row.get("chosen_text")
        sim_a = row.get("similarity_A")
        sim_b = row.get("similarity_B")

        score_a = (
            float(np.clip(float(sim_a), 0.0, 100.0) / 100.0)
            if pd.notna(sim_a)
            else 0.0
        )
        score_b = (
            float(np.clip(float(sim_b), 0.0, 100.0) / 100.0)
            if pd.notna(sim_b)
            else 0.0
        )
        score_scores[record.text_a_id] = score_a
        score_scores[record.text_b_id] = score_b

        chosen_a = 1.0 if chosen == "A" else 0.0
        chosen_b = 1.0 if chosen == "B" else 0.0
        chosen_scores[record.text_a_id] = chosen_a
        chosen_scores[record.text_b_id] = chosen_b

    return {
        "llm_chosen": PrecomputedCandidateMetric(
            name="llm_chosen", candidate_scores=chosen_scores
        ),
        "llm_score": PrecomputedCandidateMetric(
            name="llm_score", candidate_scores=score_scores
        ),
    }


def _inject_llm_metrics_if_needed(
    *,
    metrics: Dict[str, Metric],
    records: Sequence[TrackARecord],
    needed_metric_names: Sequence[str],
    llm_jsonl: str,
    logger: Optional[Callable[[str], None]] = None,
) -> None:
    needed_llm = [name for name in needed_metric_names if name in LLM_METRICS]
    if not needed_llm:
        return
    if not llm_jsonl:
        raise ValueError(
            "LLM metric requested but --llm-jsonl is missing. "
            "Provide a row-aligned Track A LLM output file."
        )
    if logger:
        logger(f"Loading LLM outputs from '{llm_jsonl}' for evaluate/submit")
    llm_df = load_llm_track_a_outputs(Path(llm_jsonl), logger=logger)
    llm_metrics = build_llm_pair_metrics(records, llm_df)
    for name in needed_llm:
        metrics[name] = llm_metrics[name]


def _sanitize_metric_suffix(raw: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    return normalized or "model"


def predict_from_metric(
    df: pd.DataFrame, metric_name: str, *, tie_break: str = "a"
) -> np.ndarray:
    scores_a = df[f"{metric_name}_a"].to_numpy()
    scores_b = df[f"{metric_name}_b"].to_numpy()
    if tie_break == "a":
        return scores_a >= scores_b
    if tie_break == "b":
        return scores_a > scores_b
    raise ValueError("tie_break must be 'a' or 'b'.")


def predict_with_threshold(
    df: pd.DataFrame,
    metric_name: str,
    *,
    threshold: float,
    tie_break: str = "a",
) -> np.ndarray:
    scores_a = df[f"{metric_name}_a"].to_numpy()
    scores_b = df[f"{metric_name}_b"].to_numpy()
    diff = scores_a - scores_b
    if tie_break == "a":
        return diff >= threshold
    if tie_break == "b":
        return diff > threshold
    raise ValueError("tie_break must be 'a' or 'b'.")


def search_threshold(
    df: pd.DataFrame,
    *,
    metric_name: str,
    threshold_grid: Sequence[float],
    tie_break: str = "a",
) -> Tuple[float, float]:
    best_acc = -1.0
    best_t = 0.0
    for t in threshold_grid:
        preds = predict_with_threshold(
            df, metric_name, threshold=t, tie_break=tie_break
        )
        acc = accuracy_from_predictions(df, preds)
        if acc is None:
            raise ValueError("Labels are required for threshold search.")
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, best_acc


def predict_linear_combo(
    df: pd.DataFrame,
    *,
    metric_a: str,
    metric_b: str,
    alpha: float,
    beta: float,
    bias: float,
    tie_break: str = "a",
) -> np.ndarray:
    diff_a = df[f"{metric_a}_a"].to_numpy() - df[f"{metric_a}_b"].to_numpy()
    diff_b = df[f"{metric_b}_a"].to_numpy() - df[f"{metric_b}_b"].to_numpy()
    scores = alpha * diff_a + beta * diff_b + bias
    if tie_break == "a":
        return scores >= 0
    if tie_break == "b":
        return scores > 0
    raise ValueError("tie_break must be 'a' or 'b'.")


def accuracy_from_predictions(
    df: pd.DataFrame, predictions: np.ndarray
) -> Optional[float]:
    if "label" not in df.columns:
        return None
    labels = df["label"].to_numpy()
    valid_mask = pd.notna(labels)
    correct = (predictions == labels) & valid_mask
    return float(correct.sum() / len(labels)) if len(labels) else 0.0


def write_predictions(
    path: Path,
    predictions: Sequence[bool],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(
                json.dumps({"text_a_is_closer": bool(pred)}, ensure_ascii=False) + "\n"
            )


def parse_grid(spec: Optional[str], default: Sequence[float]) -> np.ndarray:
    if spec is None:
        return np.array(default, dtype=float)
    raw = spec.strip()
    if ":" in raw:
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Grid spec must be 'start:stop:step' or comma-separated list."
            )
        start, stop, step = map(float, parts)
        if step <= 0:
            raise ValueError("Grid step must be > 0.")
        count = int(np.floor((stop - start) / step)) + 1
        return start + step * np.arange(count)
    return np.array([float(p) for p in raw.split(",") if p], dtype=float)


def search_linear_weights(
    df: pd.DataFrame,
    *,
    metric_a: str,
    metric_b: str,
    alpha_grid: Sequence[float],
    beta_grid: Sequence[float],
    bias_grid: Sequence[float],
    tie_break: str = "a",
) -> Tuple[float, float, float, float]:
    best_acc = -1.0
    best = (0.0, 0.0, 0.0)
    for alpha in alpha_grid:
        for beta in beta_grid:
            for bias in bias_grid:
                preds = predict_linear_combo(
                    df,
                    metric_a=metric_a,
                    metric_b=metric_b,
                    alpha=alpha,
                    beta=beta,
                    bias=bias,
                    tie_break=tie_break,
                )
                acc = accuracy_from_predictions(df, preds)
                if acc is None:
                    raise ValueError("Labels are required for weight search.")
                if acc > best_acc:
                    best_acc = acc
                    best = (alpha, beta, bias)
    return (*best, best_acc)


def build_metrics_for_names(
    registry: Dict[str, Metric],
    names: Iterable[str],
    missing: Dict[str, str],
) -> List[Metric]:
    metrics: List[Metric] = []
    for name in names:
        if name == "rouge":
            name = "rouge_l"
        if name in registry:
            metrics.append(registry[name])
            continue
        if name in missing:
            raise RuntimeError(f"Metric '{name}' is unavailable: {missing[name]}")
        raise KeyError(f"Unknown metric '{name}'.")
    return metrics


def load_weights_file(path: str) -> Tuple[str, str, float, float, float, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("type") == "llm_gest_hybrid_logreg":
        interaction = bool(payload.get("interaction", False))
        llm_mode = str(payload.get("llm_mode", "chosen"))
        score_use_confidence = bool(payload.get("score_use_confidence", False))
        if interaction:
            raise ValueError(
                "llm_hybrid weights with interaction cannot be converted to "
                "metric_a/metric_b alpha/beta/bias."
            )
        if llm_mode == "score" and score_use_confidence:
            raise ValueError(
                "llm_hybrid score weights using confidence cannot be converted to "
                "metric_a/metric_b alpha/beta/bias."
            )
        raw_coef, raw_bias = _linear_weights_from_standardized_logreg(payload["model"])
        if len(raw_coef) != 2:
            raise ValueError(
                f"Expected 2 coefficients after conversion, got {len(raw_coef)}. "
                "This usually means the hybrid model was trained with extra features "
                "(e.g., confidence or interaction)."
            )
        llm_metric_name = "llm_chosen" if llm_mode == "chosen" else "llm_score"
        return (
            str(payload["gest_metric"]),
            llm_metric_name,
            float(raw_coef[0]),
            float(raw_coef[1]),
            float(raw_bias),
            "a",
        )
    return (
        payload["metric_a"],
        payload["metric_b"],
        float(payload["alpha"]),
        float(payload["beta"]),
        float(payload.get("bias", 0.0)),
        payload.get("tie_break", "a"),
    )


def load_label_vector(path: Path) -> np.ndarray:
    labels: List[bool] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            label = payload.get("text_a_is_closer")
            if isinstance(label, str):
                label = label.strip().lower() == "true"
            if not isinstance(label, bool):
                raise ValueError(
                    f"Invalid label in {path}: text_a_is_closer must be bool."
                )
            labels.append(label)
    return np.asarray(labels, dtype=bool)


def accuracy_from_labels(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    if len(predictions) != len(labels):
        raise ValueError(
            f"Prediction/label length mismatch: {len(predictions)} vs {len(labels)}"
        )
    return float((predictions == labels).mean()) if len(labels) else 0.0


def _fit_logreg_weights(
    df: pd.DataFrame,
    *,
    metric_a: str,
    metric_b: str,
    class_weight: str | None = "balanced",
    C: float = 0.1,
    max_iter: int = 2000,
) -> Tuple[float, float, float]:
    if "label" not in df.columns:
        raise ValueError("Labels are required for weight search.")

    diff_a = df[f"{metric_a}_a"].to_numpy() - df[f"{metric_a}_b"].to_numpy()
    diff_b = df[f"{metric_b}_a"].to_numpy() - df[f"{metric_b}_b"].to_numpy()
    X = np.vstack([diff_a, diff_b]).T
    y = df["label"].to_numpy().astype(int)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std

    model = LogisticRegression(
        class_weight=class_weight,
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
    )
    model.fit(Xs, y)

    coef = model.coef_[0]
    intercept = model.intercept_[0]
    alpha = float(coef[0] / std[0])
    beta = float(coef[1] / std[1])
    bias = float(
        intercept - (coef[0] * mean[0] / std[0]) - (coef[1] * mean[1] / std[1])
    )
    return alpha, beta, bias


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track A evaluation/submission pipeline for Narrative Similarity."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("experiments", help="List available metrics.")

    eval_cmd = sub.add_parser("evaluate", help="Evaluate a metric on a dataset.")
    eval_cmd.add_argument("--dataset", default="dev")
    eval_cmd.add_argument("--metric", default="gest_spectral_glove300")
    eval_cmd.add_argument("--metric-a", default="")
    eval_cmd.add_argument("--metric-b", default="")
    eval_cmd.add_argument("--alpha", type=float, default=1.0)
    eval_cmd.add_argument("--beta", type=float, default=1.0)
    eval_cmd.add_argument("--bias", type=float, default=0.0)
    eval_cmd.add_argument("--threshold", type=float, default=None)
    eval_cmd.add_argument("--weights-file", default="")
    eval_cmd.add_argument("--gest-csv", default="data/gest.csv")
    eval_cmd.add_argument("--output", default="")
    eval_cmd.add_argument("--bleurt-checkpoint", default="lucadiliello/BLEURT-20")
    eval_cmd.add_argument("--bleurt-backend", default="pytorch")
    eval_cmd.add_argument("--bleurt-device", default="cpu")
    eval_cmd.add_argument("--bleurt-max-length", type=int, default=256)
    eval_cmd.add_argument("--llm-jsonl", default="")
    eval_cmd.add_argument("--tie-break", choices=["a", "b"], default="a")
    eval_cmd.add_argument("--labels-file", default="")

    submit_cmd = sub.add_parser("submit", help="Generate track_a.jsonl for a dataset.")
    submit_cmd.add_argument("--dataset", default="test")
    submit_cmd.add_argument("--metric", default="gest_spectral_glove300")
    submit_cmd.add_argument("--metric-a", default="")
    submit_cmd.add_argument("--metric-b", default="")
    submit_cmd.add_argument("--alpha", type=float, default=1.0)
    submit_cmd.add_argument("--beta", type=float, default=1.0)
    submit_cmd.add_argument("--bias", type=float, default=0.0)
    submit_cmd.add_argument("--threshold", type=float, default=None)
    submit_cmd.add_argument("--weights-file", default="")
    submit_cmd.add_argument("--gest-csv", default="data/gest.csv")
    submit_cmd.add_argument(
        "--output", default=str(SUBMISSIONS_ROOT / "track_a.jsonl")
    )
    submit_cmd.add_argument("--bleurt-checkpoint", default="lucadiliello/BLEURT-20")
    submit_cmd.add_argument("--bleurt-backend", default="pytorch")
    submit_cmd.add_argument("--bleurt-device", default="cpu")
    submit_cmd.add_argument("--bleurt-max-length", type=int, default=256)
    submit_cmd.add_argument("--llm-jsonl", default="")
    submit_cmd.add_argument("--tie-break", choices=["a", "b"], default="a")
    submit_cmd.add_argument("--labels-file", default="")

    search_cmd = sub.add_parser(
        "search",
        help="Search alpha/beta/bias for a linear combo of two metrics.",
    )
    search_cmd.add_argument("--train", default="sample,synthetic")
    search_cmd.add_argument("--dev", default="dev")
    search_cmd.add_argument("--metric-a", default="gest_spectral_glove300")
    search_cmd.add_argument("--metric-b", default="bleurt")
    search_cmd.add_argument("--alpha-grid", default="0:1:0.05")
    search_cmd.add_argument("--beta-grid", default="0:1:0.05")
    search_cmd.add_argument("--bias-grid", default="-0.5:0.5:0.05")
    search_cmd.add_argument(
        "--logreg-c",
        type=float,
        default=0.1,
        help="Inverse regularization strength for logreg (smaller = stronger).",
    )
    search_cmd.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Class weight for logreg. Use 'none' to optimize raw accuracy.",
    )
    search_cmd.add_argument("--gest-csv", default="data/gest.csv")
    search_cmd.add_argument("--bleurt-checkpoint", default="lucadiliello/BLEURT-20")
    search_cmd.add_argument("--bleurt-backend", default="pytorch")
    search_cmd.add_argument("--bleurt-device", default="cpu")
    search_cmd.add_argument("--bleurt-max-length", type=int, default=256)
    search_cmd.add_argument(
        "--save-weights", default=str(SUBMISSIONS_ROOT / "weights.json")
    )
    search_cmd.add_argument("--tie-break", choices=["a", "b"], default="a")

    llm_hybrid_cmd = sub.add_parser(
        "llm-hybrid",
        help="Hybrid GEST + LLM features (chosen_text or similarity scores).",
    )
    llm_hybrid_cmd.add_argument("--dataset", default="dev")
    llm_hybrid_cmd.add_argument(
        "--llm-jsonl",
        default="data/Narrative Similarity Task/development/dev_track_a_Qwen3-32B.jsonl",
    )
    llm_hybrid_cmd.add_argument(
        "--gest-metric", default="gest_spectral_w2v_google_temporal"
    )
    llm_hybrid_cmd.add_argument("--llm-mode", choices=["chosen", "score"], default="chosen")
    llm_hybrid_cmd.add_argument(
        "--score-use-confidence",
        action="store_true",
        help="For llm-mode=score, add abs(similarity_A-similarity_B) as extra feature.",
    )
    llm_hybrid_cmd.add_argument(
        "--interaction",
        action="store_true",
        help="Add interaction feature between GEST and LLM margin.",
    )
    llm_hybrid_cmd.add_argument("--gest-csv", default="data/gest.csv")
    llm_hybrid_cmd.add_argument("--logreg-c", type=float, default=1.0)
    llm_hybrid_cmd.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="none",
        help="Class weight for logreg. Use 'none' to optimize raw accuracy.",
    )
    llm_hybrid_cmd.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Optional holdout split fraction for reporting.",
    )
    llm_hybrid_cmd.add_argument("--seed", type=int, default=42)
    llm_hybrid_cmd.add_argument("--weights-file", default="")
    llm_hybrid_cmd.add_argument(
        "--save-weights", default=str(SUBMISSIONS_ROOT / "weights_llm_hybrid.json")
    )
    llm_hybrid_cmd.add_argument("--save-linear-weights", default="")
    llm_hybrid_cmd.add_argument(
        "--output", default=str(RESULTS_ROOT / "track_a_llm_hybrid.jsonl")
    )
    llm_hybrid_cmd.add_argument("--labels-file", default="")

    threshold_cmd = sub.add_parser(
        "search-threshold",
        help="Search threshold for a single metric using train/dev.",
    )
    threshold_cmd.add_argument("--train", default="sample,synthetic")
    threshold_cmd.add_argument("--dev", default="dev")
    threshold_cmd.add_argument("--metric", default="gest_spectral_glove300")
    threshold_cmd.add_argument("--threshold-grid", default="-0.5:0.5:0.01")
    threshold_cmd.add_argument("--gest-csv", default="data/gest.csv")
    threshold_cmd.add_argument("--bleurt-checkpoint", default="lucadiliello/BLEURT-20")
    threshold_cmd.add_argument("--bleurt-backend", default="pytorch")
    threshold_cmd.add_argument("--bleurt-device", default="cpu")
    threshold_cmd.add_argument("--bleurt-max-length", type=int, default=256)
    threshold_cmd.add_argument("--tie-break", choices=["a", "b"], default="a")

    eval_all_cmd = sub.add_parser(
        "evaluate-all",
        help="Evaluate all metrics on a dataset.",
    )
    eval_all_cmd.add_argument("--dataset", default="dev")
    eval_all_cmd.add_argument("--gest-csv", default="data/gest.csv")
    eval_all_cmd.add_argument("--bleurt-checkpoint", default="lucadiliello/BLEURT-20")
    eval_all_cmd.add_argument("--bleurt-backend", default="pytorch")
    eval_all_cmd.add_argument("--bleurt-device", default="cpu")
    eval_all_cmd.add_argument("--bleurt-max-length", type=int, default=256)
    eval_all_cmd.add_argument("--tie-break", choices=["a", "b"], default="a")
    eval_all_cmd.add_argument(
        "--text-metrics",
        default="bleu,rouge_l,sbert_cosine,bleurt",
    )
    eval_all_cmd.add_argument(
        "--output-dir",
        default=str(RESULTS_ROOT / "metrics"),
        help="Directory to store per-metric predictions and summary.",
    )
    eval_all_cmd.add_argument(
        "--llm-output-dir",
        default="",
        help=(
            "Optional directory with Track A LLM outputs to evaluate as metrics and "
            "append to metrics_summary.csv."
        ),
    )
    eval_all_cmd.add_argument(
        "--llm-output-glob",
        default="dev_track_a_*.jsonl",
        help="Glob pattern used with --llm-output-dir.",
    )
    eval_all_cmd.add_argument(
        "--llm-modes",
        default="chosen,score",
        help="Comma-separated LLM modes to evaluate from each file: chosen,score.",
    )
    search_cmd.add_argument("--method", choices=["grid", "logreg"], default="logreg")

    sweep_cmd = sub.add_parser(
        "sweep",
        help="Evaluate all graph/text metric pairs and find best weights.",
    )
    sweep_cmd.add_argument("--train", default="sample,synthetic")
    sweep_cmd.add_argument("--dev", default="dev")
    sweep_cmd.add_argument("--graph-metrics", default="all")
    sweep_cmd.add_argument("--text-metrics", default="bleu,rouge_l,sbert_cosine,bleurt")
    sweep_cmd.add_argument("--alpha-grid", default="0:1:0.05")
    sweep_cmd.add_argument("--beta-grid", default="0:1:0.05")
    sweep_cmd.add_argument("--bias-grid", default="-0.5:0.5:0.05")
    sweep_cmd.add_argument("--method", choices=["grid", "logreg"], default="logreg")
    sweep_cmd.add_argument(
        "--logreg-c",
        type=float,
        default=0.1,
        help="Inverse regularization strength for logreg (smaller = stronger).",
    )
    sweep_cmd.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Class weight for logreg. Use 'none' to optimize raw accuracy.",
    )
    sweep_cmd.add_argument("--gest-csv", default="data/gest.csv")
    sweep_cmd.add_argument("--bleurt-checkpoint", default="lucadiliello/BLEURT-20")
    sweep_cmd.add_argument("--bleurt-backend", default="pytorch")
    sweep_cmd.add_argument("--bleurt-device", default="cpu")
    sweep_cmd.add_argument("--bleurt-max-length", type=int, default=256)
    sweep_cmd.add_argument("--top-k", type=int, default=10)
    sweep_cmd.add_argument(
        "--save-weights", default=str(SUBMISSIONS_ROOT / "weights.json")
    )
    sweep_cmd.add_argument("--tie-break", choices=["a", "b"], default="a")

    args = parser.parse_args()

    if args.command == "experiments":
        print("Available metrics:")
        for name in KNOWN_METRICS:
            print(f"- {name}")
        print("\nOptional metrics require extra dependencies: bleurt (bleurt-pytorch).")
        return

    combo_spec: Optional[Tuple[str, str, float, float, float, str]] = None
    if args.command in {"evaluate", "submit"}:
        if args.weights_file:
            combo_spec = load_weights_file(args.weights_file)
        elif args.metric_a and args.metric_b:
            combo_spec = (
                args.metric_a,
                args.metric_b,
                args.alpha,
                args.beta,
                args.bias,
                args.tie_break,
            )

    gest_store = GestStore.from_csv(Path(args.gest_csv))
    metrics: Dict[str, Metric] = {}
    missing: Dict[str, str] = {}
    if args.command not in {"sweep", "evaluate-all", "llm-hybrid"}:
        requested_metrics: List[str] = []
        if args.command in {"evaluate", "submit"}:
            if combo_spec:
                requested_metrics = [combo_spec[0], combo_spec[1]]
            else:
                requested_metrics = [args.metric]
        elif args.command == "search":
            requested_metrics = [args.metric_a, args.metric_b]
        elif args.command == "search-threshold":
            requested_metrics = [args.metric]
        metrics, missing = build_metric_registry(
            gest_store,
            bleurt_checkpoint=args.bleurt_checkpoint,
            bleurt_backend=args.bleurt_backend,
            bleurt_device=args.bleurt_device,
            bleurt_max_length=args.bleurt_max_length,
            requested=requested_metrics,
            logger=_log_step,
        )

    if args.command == "evaluate":
        _log_step(f"Loading dataset '{args.dataset}'")
        dataset_path = resolve_dataset_path(args.dataset)
        records = load_track_a_records(
            dataset_path, logger=_log_step
        )
        _log_step(f"Loaded {len(records)} records")
        if combo_spec:
            _log_step("Building combo metrics")
            metric_a, metric_b, alpha, beta, bias, tie_break = combo_spec
            _inject_llm_metrics_if_needed(
                metrics=metrics,
                records=records,
                needed_metric_names=[metric_a, metric_b],
                llm_jsonl=args.llm_jsonl,
                logger=_log_step,
            )
            metric_objs = build_metrics_for_names(
                metrics, [metric_a, metric_b], missing
            )
            _log_step("Scoring dataset")
            df = build_feature_frame(records, metric_objs)
            preds = predict_linear_combo(
                df,
                metric_a=metric_a,
                metric_b=metric_b,
                alpha=alpha,
                beta=beta,
                bias=bias,
                tie_break=tie_break,
            )
        else:
            _log_step(f"Building metric '{args.metric}'")
            _inject_llm_metrics_if_needed(
                metrics=metrics,
                records=records,
                needed_metric_names=[args.metric],
                llm_jsonl=args.llm_jsonl,
                logger=_log_step,
            )
            metric_objs = build_metrics_for_names(metrics, [args.metric], missing)
            _log_step("Scoring dataset")
            df = build_feature_frame(records, metric_objs)
            if args.threshold is not None:
                _log_step(f"Using threshold {args.threshold:.4f}")
                preds = predict_with_threshold(
                    df,
                    args.metric,
                    threshold=args.threshold,
                    tie_break=args.tie_break,
                )
            else:
                preds = predict_from_metric(df, args.metric, tie_break=args.tie_break)
        acc = accuracy_from_predictions(df, preds)
        if acc is None:
            print("No labels available for accuracy.")
        else:
            print(f"Accuracy: {acc:.4f}")
        if args.labels_file:
            labels = load_label_vector(Path(args.labels_file))
            local_acc = accuracy_from_labels(preds, labels)
            print(f"Local accuracy ({args.labels_file}): {local_acc:.4f}")
        if args.output:
            _log_step(f"Writing predictions to {args.output}")
            write_predictions(Path(args.output), preds)
        return

    if args.command == "evaluate-all":
        _log_step(f"Loading dataset '{args.dataset}'")
        dataset_path = resolve_dataset_path(args.dataset)
        records = load_track_a_records(
            dataset_path, logger=_log_step
        )
        _log_step(f"Loaded {len(records)} records")

        text_names = [n.strip() for n in args.text_metrics.split(",") if n.strip()]
        graph_names = _all_graph_metric_names()
        requested_metrics = graph_names + text_names
        _log_step("Building metric registry")
        metrics, missing = build_metric_registry(
            gest_store,
            bleurt_checkpoint=args.bleurt_checkpoint,
            bleurt_backend=args.bleurt_backend,
            bleurt_device=args.bleurt_device,
            bleurt_max_length=args.bleurt_max_length,
            requested=requested_metrics,
            logger=_log_step,
        )
        metric_objs = build_metrics_for_names(metrics, requested_metrics, missing)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        llm_modes = [m.strip() for m in args.llm_modes.split(",") if m.strip()]
        invalid_llm_modes = [m for m in llm_modes if m not in {"chosen", "score"}]
        if invalid_llm_modes:
            raise ValueError(
                f"Invalid --llm-modes values: {invalid_llm_modes}. "
                "Use a subset of: chosen,score."
            )

        if args.llm_output_dir:
            llm_dir = Path(args.llm_output_dir)
            llm_paths = sorted(llm_dir.glob(args.llm_output_glob))
            if not llm_paths:
                raise FileNotFoundError(
                    f"No LLM files matched '{args.llm_output_glob}' in {llm_dir}"
                )
            _log_step(
                f"Adding LLM metrics from {len(llm_paths)} file(s) in {llm_dir}"
            )
            llm_parse_rows: List[dict] = []
            for llm_path in llm_paths:
                _log_step(f"Loading LLM outputs: {llm_path}")
                llm_df = load_llm_track_a_outputs(llm_path, logger=_log_step)
                _assert_llm_alignment(records, llm_df)
                llm_metrics = build_llm_pair_metrics(records, llm_df)

                chosen_missing = int((llm_df["chosen_text"].fillna("") == "").sum())
                score_missing = int(
                    (llm_df["similarity_A"].isna() | llm_df["similarity_B"].isna()).sum()
                )
                llm_parse_rows.append(
                    {
                        "model_file": llm_path.name,
                        "rows": len(llm_df),
                        "chosen_missing": chosen_missing,
                        "score_missing": score_missing,
                    }
                )

                stem = llm_path.stem
                for prefix in ("dev_track_a_", "test_track_a_", "track_a_"):
                    if stem.startswith(prefix):
                        stem = stem[len(prefix) :]
                        break
                suffix = _sanitize_metric_suffix(stem)

                if "chosen" in llm_modes:
                    base = llm_metrics["llm_chosen"]
                    metric_objs.append(
                        PrecomputedCandidateMetric(
                            name=f"llm_chosen_{suffix}",
                            candidate_scores=base.candidate_scores,
                            missing_value=base.missing_value,
                        )
                    )
                if "score" in llm_modes:
                    base = llm_metrics["llm_score"]
                    metric_objs.append(
                        PrecomputedCandidateMetric(
                            name=f"llm_score_{suffix}",
                            candidate_scores=base.candidate_scores,
                            missing_value=base.missing_value,
                        )
                    )

            parse_summary_path = output_dir / "llm_parse_summary.csv"
            _log_step(f"Writing LLM parse summary to {parse_summary_path}")
            pd.DataFrame(llm_parse_rows).to_csv(parse_summary_path, index=False)

        results: List[Tuple[str, float | None]] = []
        for metric in metric_objs:
            _log_step(f"Scoring metric '{metric.name}'")
            scores_a, scores_b, labels = _score_metric(records, metric)

            metric_df = pd.DataFrame(
                {
                    "split_id": [r.split_id for r in records],
                    "row_index": [r.row_index for r in records],
                    f"{metric.name}_a": scores_a,
                    f"{metric.name}_b": scores_b,
                }
            )
            if any(label is not None for label in labels):
                metric_df["label"] = [
                    bool(label) if label is not None else None for label in labels
                ]
            metric_scores_path = output_dir / f"{metric.name}.csv"
            _log_step(f"Writing scores for {metric.name} -> {metric_scores_path}")
            metric_df.to_csv(metric_scores_path, index=False)

            if args.tie_break == "a":
                preds = scores_a >= scores_b
            else:
                preds = scores_a > scores_b
            acc = accuracy_from_predictions(metric_df, preds)
            results.append((metric.name, acc))

            metric_pred_path = output_dir / f"{metric.name}.jsonl"
            _log_step(f"Writing predictions for {metric.name} -> {metric_pred_path}")
            write_predictions(metric_pred_path, preds)

        results.sort(key=lambda item: (item[1] is None, -(item[1] or 0.0)))
        summary_path = output_dir / "metrics_summary.csv"
        _log_step(f"Writing summary to {summary_path}")
        pd.DataFrame(
            [{"metric": name, "accuracy": acc} for name, acc in results]
        ).to_csv(summary_path, index=False)

        _log_step("Combining per-metric scores into scores.csv")
        combined: Optional[pd.DataFrame] = None
        for metric in metric_objs:
            metric_scores_path = output_dir / f"{metric.name}.csv"
            metric_df = pd.read_csv(metric_scores_path)
            if combined is None:
                cols = [
                    "split_id",
                    "row_index",
                    f"{metric.name}_a",
                    f"{metric.name}_b",
                ]
                if "label" in metric_df.columns:
                    cols.append("label")
                combined = metric_df[cols]
            else:
                cols = [
                    "split_id",
                    "row_index",
                    f"{metric.name}_a",
                    f"{metric.name}_b",
                ]
                combined = combined.merge(
                    metric_df[cols],
                    on=["split_id", "row_index"],
                    how="left",
                )
        if combined is not None:
            scores_path = output_dir / "scores.csv"
            _log_step(f"Writing combined scores to {scores_path}")
            combined.to_csv(scores_path, index=False)

        for name, acc in results:
            if acc is None:
                print(f"{name}: no labels")
            else:
                print(f"{name}: {acc:.4f}")
        return

    if args.command == "llm-hybrid":
        _log_step(f"Loading dataset '{args.dataset}'")
        dataset_path = resolve_dataset_path(args.dataset)
        records = load_track_a_records(dataset_path, logger=_log_step)
        _log_step(f"Loaded {len(records)} records")

        _log_step(f"Loading LLM outputs from '{args.llm_jsonl}'")
        llm_df = load_llm_track_a_outputs(Path(args.llm_jsonl), logger=_log_step)
        _log_step("Checking dataset/LLM alignment")
        _assert_llm_alignment(records, llm_df)

        weight_payload: dict | None = None
        llm_mode = args.llm_mode
        score_use_confidence = bool(args.score_use_confidence)
        include_interaction = bool(args.interaction)
        gest_metric = args.gest_metric
        if args.weights_file:
            weight_payload = json.loads(Path(args.weights_file).read_text(encoding="utf-8"))
            if weight_payload.get("type") != "llm_gest_hybrid_logreg":
                raise ValueError(
                    "weights-file is not an llm_gest_hybrid_logreg payload."
                )
            llm_mode = str(weight_payload.get("llm_mode", llm_mode))
            score_use_confidence = bool(
                weight_payload.get("score_use_confidence", score_use_confidence)
            )
            include_interaction = bool(
                weight_payload.get("interaction", include_interaction)
            )
            gest_metric = str(weight_payload.get("gest_metric", gest_metric))
            _log_step(
                f"Loaded hybrid weights: gest_metric={gest_metric}, "
                f"llm_mode={llm_mode}, score_use_confidence={score_use_confidence}, "
                f"interaction={include_interaction}"
            )

        _log_step(f"Building GEST metric '{gest_metric}'")
        local_store = GestStore.from_csv(Path(args.gest_csv))
        local_metrics, local_missing = build_metric_registry(
            local_store,
            bleurt_checkpoint=args.bleurt_checkpoint if hasattr(args, "bleurt_checkpoint") else "lucadiliello/BLEURT-20",
            bleurt_backend=args.bleurt_backend if hasattr(args, "bleurt_backend") else "pytorch",
            bleurt_device=args.bleurt_device if hasattr(args, "bleurt_device") else "cpu",
            bleurt_max_length=args.bleurt_max_length if hasattr(args, "bleurt_max_length") else 256,
            requested=[gest_metric],
            logger=_log_step,
        )
        metric_obj = build_metrics_for_names(local_metrics, [gest_metric], local_missing)[0]

        _log_step("Scoring GEST metric")
        feature_df = build_feature_frame(records, [metric_obj], show_progress=True)
        d_gest = (
            feature_df[f"{gest_metric}_a"].to_numpy()
            - feature_df[f"{gest_metric}_b"].to_numpy()
        )
        X, feature_names = _build_llm_hybrid_features(
            d_gest=d_gest,
            llm_df=llm_df,
            llm_mode=llm_mode,
            include_confidence=score_use_confidence,
            include_interaction=include_interaction,
        )
        _log_step(f"Hybrid features: {', '.join(feature_names)}")

        holdout_acc: Optional[float] = None
        if weight_payload is None:
            if "label" not in feature_df.columns:
                raise ValueError(
                    "Labels are required to train llm-hybrid weights. "
                    "Use --weights-file for unlabeled sets."
                )
            y = feature_df["label"].to_numpy().astype(int)
            class_weight = None if args.class_weight == "none" else args.class_weight
            model_payload = _fit_logreg_feature_matrix(
                X,
                y,
                C=args.logreg_c,
                class_weight=class_weight,
            )
            preds = _predict_logreg_feature_matrix(X, model_payload)
            train_acc = accuracy_from_predictions(feature_df, preds)

            if (
                0.0 < args.holdout_fraction < 1.0
                and len(y) >= 20
                and len(np.unique(y)) > 1
            ):
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=args.holdout_fraction,
                    random_state=args.seed,
                    stratify=y,
                )
                split_payload = _fit_logreg_feature_matrix(
                    X_train,
                    y_train,
                    C=args.logreg_c,
                    class_weight=class_weight,
                )
                split_preds = _predict_logreg_feature_matrix(X_test, split_payload)
                holdout_acc = float((split_preds == y_test).mean())
            else:
                _log_step(
                    "Skipping holdout evaluation (not enough data, one class only, "
                    "or holdout-fraction disabled)."
                )

            if args.save_weights:
                payload = {
                    "type": "llm_gest_hybrid_logreg",
                    "gest_metric": gest_metric,
                    "llm_mode": llm_mode,
                    "score_use_confidence": score_use_confidence,
                    "interaction": include_interaction,
                    "feature_names": feature_names,
                    "model": model_payload,
                }
                Path(args.save_weights).write_text(
                    json.dumps(payload, indent=2), encoding="utf-8"
                )
                _log_step(f"Saved hybrid weights to {args.save_weights}")
            if args.save_linear_weights:
                if include_interaction:
                    raise ValueError(
                        "Cannot export linear alpha/beta/bias when --interaction is enabled."
                    )
                if llm_mode == "score" and score_use_confidence:
                    raise ValueError(
                        "Cannot export linear alpha/beta/bias when --score-use-confidence "
                        "is enabled. Re-run without that flag."
                    )
                if len(feature_names) != 2:
                    raise ValueError(
                        f"Expected exactly 2 features for linear export, got {len(feature_names)}."
                    )
                raw_coef, raw_bias = _linear_weights_from_standardized_logreg(model_payload)
                llm_metric_name = "llm_chosen" if llm_mode == "chosen" else "llm_score"
                linear_payload = {
                    "metric_a": gest_metric,
                    "metric_b": llm_metric_name,
                    "alpha": float(raw_coef[0]),
                    "beta": float(raw_coef[1]),
                    "bias": float(raw_bias),
                    "tie_break": "a",
                }
                Path(args.save_linear_weights).write_text(
                    json.dumps(linear_payload, indent=2), encoding="utf-8"
                )
                _log_step(f"Saved linear weights to {args.save_linear_weights}")
        else:
            model_payload = weight_payload["model"]
            expected_dim = len(model_payload["coef"])
            if X.shape[1] != expected_dim:
                raise ValueError(
                    f"Feature dimension mismatch: features={X.shape[1]} vs "
                    f"weights={expected_dim}."
                )
            preds = _predict_logreg_feature_matrix(X, model_payload)
            train_acc = accuracy_from_predictions(feature_df, preds)

        if train_acc is None:
            print("No labels available for accuracy.")
        else:
            print(f"Accuracy: {train_acc:.4f}")
        if args.labels_file:
            labels = load_label_vector(Path(args.labels_file))
            local_acc = accuracy_from_labels(preds, labels)
            print(f"Local accuracy ({args.labels_file}): {local_acc:.4f}")
        if holdout_acc is not None:
            print(f"Holdout accuracy: {holdout_acc:.4f}")

        if args.output:
            _log_step(f"Writing predictions to {args.output}")
            write_predictions(Path(args.output), preds)
        return

    if args.command == "submit":
        _log_step(f"Loading dataset '{args.dataset}'")
        dataset_path = resolve_dataset_path(args.dataset)
        records = load_track_a_records(
            dataset_path, logger=_log_step
        )
        _log_step(f"Loaded {len(records)} records")
        if combo_spec:
            _log_step("Building combo metrics")
            metric_a, metric_b, alpha, beta, bias, tie_break = combo_spec
            _inject_llm_metrics_if_needed(
                metrics=metrics,
                records=records,
                needed_metric_names=[metric_a, metric_b],
                llm_jsonl=args.llm_jsonl,
                logger=_log_step,
            )
            metric_objs = build_metrics_for_names(
                metrics, [metric_a, metric_b], missing
            )
            _log_step("Scoring dataset")
            df = build_feature_frame(records, metric_objs)
            preds = predict_linear_combo(
                df,
                metric_a=metric_a,
                metric_b=metric_b,
                alpha=alpha,
                beta=beta,
                bias=bias,
                tie_break=tie_break,
            )
        else:
            _log_step(f"Building metric '{args.metric}'")
            _inject_llm_metrics_if_needed(
                metrics=metrics,
                records=records,
                needed_metric_names=[args.metric],
                llm_jsonl=args.llm_jsonl,
                logger=_log_step,
            )
            metric_objs = build_metrics_for_names(metrics, [args.metric], missing)
            _log_step("Scoring dataset")
            df = build_feature_frame(records, metric_objs)
            if args.threshold is not None:
                _log_step(f"Using threshold {args.threshold:.4f}")
                preds = predict_with_threshold(
                    df,
                    args.metric,
                    threshold=args.threshold,
                    tie_break=args.tie_break,
                )
            else:
                preds = predict_from_metric(df, args.metric, tie_break=args.tie_break)
        _log_step(f"Writing predictions to {args.output}")
        write_predictions(Path(args.output), preds)
        print(f"Wrote predictions to {args.output}")
        if args.labels_file:
            labels = load_label_vector(Path(args.labels_file))
            local_acc = accuracy_from_labels(preds, labels)
            print(f"Local accuracy ({args.labels_file}): {local_acc:.4f}")
        return

    if args.command == "search":
        _log_step("Loading train/dev datasets")
        train_paths = [
            resolve_dataset_path(p.strip()) for p in args.train.split(",") if p.strip()
        ]
        dev_path = resolve_dataset_path(args.dev)
        metric_objs = build_metrics_for_names(
            metrics, [args.metric_a, args.metric_b], missing
        )
        _log_step("Scoring train/dev datasets")

        train_records: List[TrackARecord] = []
        for path in train_paths:
            train_records.extend(
                load_track_a_records(path, drop_invalid=True, logger=_log_step)
            )
        if args.metric_a.startswith("gest_") or args.metric_b.startswith("gest_"):
            _log_step("Filtering train records to those with GEST graphs")
            train_records = _filter_records_with_gest(
                train_records, gest_store, logger=_log_step
            )
        dev_records = load_track_a_records(dev_path, logger=_log_step)

        train_df = build_feature_frame(train_records, metric_objs)
        dev_df = build_feature_frame(dev_records, metric_objs)

        alpha_grid = parse_grid(args.alpha_grid, [0.0, 1.0])
        beta_grid = parse_grid(args.beta_grid, [0.0, 1.0])
        bias_grid = parse_grid(args.bias_grid, [0.0])
        _log_step(
            f"Search method: {args.method} | "
            f"alpha_grid={len(alpha_grid)}, beta_grid={len(beta_grid)}, bias_grid={len(bias_grid)}"
        )
        if args.method == "logreg":
            _log_step(f"logreg C={args.logreg_c}")
            _log_step(f"logreg class_weight={args.class_weight}")

        if args.method == "logreg":
            alpha, beta, bias = _fit_logreg_weights(
                train_df,
                metric_a=args.metric_a,
                metric_b=args.metric_b,
                class_weight=None
                if args.class_weight == "none"
                else args.class_weight,
                C=args.logreg_c,
            )
            train_preds = predict_linear_combo(
                train_df,
                metric_a=args.metric_a,
                metric_b=args.metric_b,
                alpha=alpha,
                beta=beta,
                bias=bias,
                tie_break=args.tie_break,
            )
            train_acc = accuracy_from_predictions(train_df, train_preds)
        else:
            alpha_grid = parse_grid(args.alpha_grid, [0.0, 1.0])
            beta_grid = parse_grid(args.beta_grid, [0.0, 1.0])
            bias_grid = parse_grid(args.bias_grid, [0.0])
            alpha, beta, bias, train_acc = search_linear_weights(
                train_df,
                metric_a=args.metric_a,
                metric_b=args.metric_b,
                alpha_grid=alpha_grid,
                beta_grid=beta_grid,
                bias_grid=bias_grid,
                tie_break=args.tie_break,
            )

        dev_preds = predict_linear_combo(
            dev_df,
            metric_a=args.metric_a,
            metric_b=args.metric_b,
            alpha=alpha,
            beta=beta,
            bias=bias,
            tie_break=args.tie_break,
        )
        dev_acc = accuracy_from_predictions(dev_df, dev_preds)

        print(f"Best weights: alpha={alpha:.4f}, beta={beta:.4f}, bias={bias:.4f}")
        print(f"Train accuracy: {train_acc:.4f}")
        if dev_acc is None:
            print("Dev set has no labels.")
        else:
            print(f"Dev accuracy: {dev_acc:.4f}")

        if args.save_weights:
            _log_step(f"Saving weights to {args.save_weights}")
            payload = {
                "metric_a": args.metric_a,
                "metric_b": args.metric_b,
                "alpha": alpha,
                "beta": beta,
                "bias": bias,
                "tie_break": args.tie_break,
            }
            Path(args.save_weights).write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            print(f"Saved weights to {args.save_weights}")
        return

    if args.command == "search-threshold":
        _log_step("Loading train/dev datasets")
        train_paths = [
            resolve_dataset_path(p.strip()) for p in args.train.split(",") if p.strip()
        ]
        dev_path = resolve_dataset_path(args.dev)

        metric_objs = build_metrics_for_names(metrics, [args.metric], missing)
        _log_step("Scoring train/dev datasets")

        train_records: List[TrackARecord] = []
        for path in train_paths:
            train_records.extend(
                load_track_a_records(path, drop_invalid=True, logger=_log_step)
            )
        if args.metric.startswith("gest_"):
            _log_step("Filtering train records to those with GEST graphs")
            train_records = _filter_records_with_gest(
                train_records, gest_store, logger=_log_step
            )
        dev_records = load_track_a_records(dev_path, logger=_log_step)

        train_df = build_feature_frame(train_records, metric_objs)
        dev_df = build_feature_frame(dev_records, metric_objs)

        threshold_grid = parse_grid(args.threshold_grid, [0.0])
        _log_step(f"Threshold grid size: {len(threshold_grid)}")
        threshold, train_acc = search_threshold(
            train_df,
            metric_name=args.metric,
            threshold_grid=threshold_grid,
            tie_break=args.tie_break,
        )
        dev_preds = predict_with_threshold(
            dev_df,
            args.metric,
            threshold=threshold,
            tie_break=args.tie_break,
        )
        dev_acc = accuracy_from_predictions(dev_df, dev_preds)

        print(f"Best threshold: {threshold:.4f}")
        print(f"Train accuracy: {train_acc:.4f}")
        if dev_acc is None:
            print("Dev set has no labels.")
        else:
            print(f"Dev accuracy: {dev_acc:.4f}")
        return

    if args.command == "sweep":
        _log_step("Loading train/dev datasets")
        train_paths = [
            resolve_dataset_path(p.strip()) for p in args.train.split(",") if p.strip()
        ]
        dev_path = resolve_dataset_path(args.dev)

        graph_names = (
            _all_graph_metric_names()
            if args.graph_metrics.strip().lower() == "all"
            else [n.strip() for n in args.graph_metrics.split(",") if n.strip()]
        )
        text_names = [n.strip() for n in args.text_metrics.split(",") if n.strip()]

        requested = list(dict.fromkeys(graph_names + text_names))
        _log_step(
            f"Graph metrics: {len(graph_names)} | Text metrics: {len(text_names)}"
        )
        _log_step("Building metric registry")
        metrics, missing = build_metric_registry(
            gest_store,
            bleurt_checkpoint=args.bleurt_checkpoint,
            bleurt_backend=args.bleurt_backend,
            bleurt_device=args.bleurt_device,
            bleurt_max_length=args.bleurt_max_length,
            requested=requested,
            logger=_log_step,
        )
        metric_objs = build_metrics_for_names(metrics, requested, missing)

        train_records: List[TrackARecord] = []
        for path in train_paths:
            train_records.extend(
                load_track_a_records(path, drop_invalid=True, logger=_log_step)
            )
        if any(name.startswith("gest_") for name in graph_names):
            _log_step("Filtering train records to those with GEST graphs")
            train_records = _filter_records_with_gest(
                train_records, gest_store, logger=_log_step
            )
        dev_records = load_track_a_records(dev_path, logger=_log_step)

        _log_step("Scoring train/dev datasets")
        train_df = build_feature_frame(train_records, metric_objs)
        dev_df = build_feature_frame(dev_records, metric_objs)

        alpha_grid = parse_grid(args.alpha_grid, [0.0, 1.0])
        beta_grid = parse_grid(args.beta_grid, [0.0, 1.0])
        bias_grid = parse_grid(args.bias_grid, [0.0])
        _log_step(
            f"Sweep method: {args.method} | "
            f"alpha_grid={len(alpha_grid)}, beta_grid={len(beta_grid)}, bias_grid={len(bias_grid)}"
        )
        if args.method == "logreg":
            _log_step(f"logreg C={args.logreg_c}")

        results: List[dict] = []
        for graph_metric in graph_names:
            for text_metric in text_names:
                _log_step(f"Evaluating pair: {graph_metric} + {text_metric}")
                if args.method == "logreg":
                    alpha, beta, bias = _fit_logreg_weights(
                        train_df,
                        metric_a=graph_metric,
                        metric_b=text_metric,
                        class_weight=None
                        if args.class_weight == "none"
                        else args.class_weight,
                        C=args.logreg_c,
                    )
                else:
                    alpha, beta, bias, _ = search_linear_weights(
                        train_df,
                        metric_a=graph_metric,
                        metric_b=text_metric,
                        alpha_grid=alpha_grid,
                        beta_grid=beta_grid,
                        bias_grid=bias_grid,
                        tie_break=args.tie_break,
                    )
                dev_preds = predict_linear_combo(
                    dev_df,
                    metric_a=graph_metric,
                    metric_b=text_metric,
                    alpha=alpha,
                    beta=beta,
                    bias=bias,
                    tie_break=args.tie_break,
                )
                dev_acc = accuracy_from_predictions(dev_df, dev_preds) or 0.0
                results.append(
                    {
                        "metric_a": graph_metric,
                        "metric_b": text_metric,
                        "alpha": alpha,
                        "beta": beta,
                        "bias": bias,
                        "dev_acc": dev_acc,
                    }
                )

        results.sort(key=lambda r: r["dev_acc"], reverse=True)
        top_k = max(1, int(args.top_k))
        print("Top results:")
        for row in results[:top_k]:
            print(
                f"{row['metric_a']} + {row['metric_b']} -> "
                f"alpha={row['alpha']:.4f}, beta={row['beta']:.4f}, "
                f"bias={row['bias']:.4f}, dev_acc={row['dev_acc']:.4f}"
            )

        if args.save_weights and results:
            _log_step(f"Saving best weights to {args.save_weights}")
            best = results[0]
            payload = {
                "metric_a": best["metric_a"],
                "metric_b": best["metric_b"],
                "alpha": best["alpha"],
                "beta": best["beta"],
                "bias": best["bias"],
                "tie_break": args.tie_break,
            }
            Path(args.save_weights).write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            print(f"Saved weights to {args.save_weights}")
        return


if __name__ == "__main__":
    main()
