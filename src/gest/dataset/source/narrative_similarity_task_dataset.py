import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from gest.dataset.source.base_source_dataset import DataRow, SourceDataset
from gest.dataset.source.source_dataset_enum import SourceDatasetEnum


@dataclass(frozen=True, slots=True)
class NarrativeSimilarityTaskRow(DataRow):
    split_id: str
    row_index: int
    role: str
    content: str

    def id(self) -> str:
        return f"{self.split_id}:{self.row_index}:{self.role}"

    def text(self) -> str:
        return self.content.replace("\n", " ").strip()


class NarrativeSimilarityTaskDataset(SourceDataset[NarrativeSimilarityTaskRow]):
    """Loader for SemEval-2026 Task 4 Narrative Similarity (Track A) JSONL files."""

    def __init__(self, *paths: Path):
        super().__init__(SourceDatasetEnum.NARRATIVE_SIMILARITY_TASK, *paths)

    def load(self) -> List[NarrativeSimilarityTaskRow]:
        records: List[NarrativeSimilarityTaskRow] = []
        for p in self.paths:
            split_id = f"{p.parent.name}/{p.stem}"
            with open(p, encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)

                    for role in ("anchor_text", "text_a", "text_b"):
                        content = payload.get(role, "")
                        records.append(
                            NarrativeSimilarityTaskRow(
                                split_id=split_id,
                                row_index=idx,
                                role=role,
                                content=content,
                            )
                        )
        return records