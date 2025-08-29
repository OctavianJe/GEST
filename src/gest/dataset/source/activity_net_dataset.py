import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from gest.dataset.source.base_source_dataset import DataRow, SourceDataset
from gest.dataset.source.source_dataset_enum import SourceDatasetEnum


@dataclass(frozen=True, slots=True)
class ActivityNetRow(DataRow):
    url: str
    duration: float
    timestamps: List[List[float]]
    sentences: List[str]

    def id(self) -> str:
        return self.url

    def text(self) -> str:
        return ActivityNetCaptionsDataset.flatten_sentences(self.sentences)


class ActivityNetCaptionsDataset(SourceDataset[ActivityNetRow]):
    """Loader for the original *ActivityNet Captions* split JSON files."""

    def __init__(self, *paths: Path):
        super().__init__(SourceDatasetEnum.ACTIVITY_NET_CAPTIONS, *paths)

    def load(self) -> List[ActivityNetRow]:
        records: List[ActivityNetRow] = []
        for p in self.paths:
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            for url, info in raw.items():
                row = ActivityNetRow(
                    url=url,
                    duration=info.get("duration", 0.0),
                    timestamps=info.get("timestamps", []),
                    sentences=info.get("sentences", []),
                )
                records.append(row)
        return records

    @staticmethod
    def flatten_sentences(sentences: List[str]) -> str:
        """Concatenate a list of sentences into one whitespace-normalized paragraph."""
        return " ".join(s.strip() for s in sentences)
