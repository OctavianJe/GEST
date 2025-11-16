import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from gest.dataset.source.base_source_dataset import DataRow, SourceDataset
from gest.dataset.source.source_dataset_enum import SourceDatasetEnum


@dataclass(frozen=True, slots=True)
class ImarRow(DataRow):
    row_id: str
    initial: str
    intermediate: str
    final: str

    def id(self) -> str:
        return self.row_id

    def text(self) -> str:
        """Return the 'initial' text field for GEST generation."""
        return self.initial.replace("\n", " ").strip()


class ImarDataset(SourceDataset[ImarRow]):
    """Loader for the IMAR dataset file."""

    def __init__(self, *paths: Path):
        super().__init__(SourceDatasetEnum.IMAR, *paths)

    def load(self) -> List[ImarRow]:
        records: List[ImarRow] = []
        for p in self.paths:
            with open(p, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row_dict in reader:
                    row = ImarRow(
                        row_id=row_dict.get("id", ""),
                        initial=row_dict.get("initial", ""),
                        intermediate=row_dict.get("intermediate", ""),
                        final=row_dict.get("final", ""),
                    )
                    records.append(row)
        return records
