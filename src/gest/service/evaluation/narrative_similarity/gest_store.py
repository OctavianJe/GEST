from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from gest.data.gest import GEST
from gest.service.evaluation.graph_matching.graph import GESTGraph


@dataclass(slots=True)
class GestStore:
    """Loads and stores GEST graphs keyed by dataset id."""

    dataset: str
    graphs: Dict[str, GESTGraph] = field(default_factory=dict)
    parse_errors: int = 0

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        *,
        dataset: str = "Narrative Similarity Task",
        strict: bool = False,
    ) -> "GestStore":
        if not csv_path.exists():
            raise FileNotFoundError(f"GEST csv not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required = {"dataset", "id", "gest"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"GEST csv missing required columns: {sorted(missing)}"
            )

        subset = df[df["dataset"] == dataset]
        graphs: Dict[str, GESTGraph] = {}
        parse_errors = 0

        for _, row in subset.iterrows():
            row_id = str(row["id"])
            try:
                gest = GEST.model_validate_json(row["gest"])
                graphs[row_id] = GESTGraph(gest=gest)
            except Exception:
                parse_errors += 1
                if strict:
                    raise

        return cls(dataset=dataset, graphs=graphs, parse_errors=parse_errors)

    def get(self, row_id: str) -> Optional[GESTGraph]:
        return self.graphs.get(row_id)
