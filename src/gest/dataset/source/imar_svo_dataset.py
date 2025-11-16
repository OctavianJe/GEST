import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from gest.dataset.source.base_source_dataset import DataRow, SourceDataset
from gest.dataset.source.source_dataset_enum import SourceDatasetEnum


@dataclass(frozen=True, slots=True)
class ImarSvoRow(DataRow):
    row_id: str
    initial: str
    intermediate: str
    final: str

    def id(self) -> str:
        return self.row_id

    def text(self) -> str:
        """Return the 'intermediate' text field for GEST generation."""
        # Split text into sentences (delimited by newlines)
        sentences = self.intermediate.split("\n")

        # Process each sentence: strip whitespace and ensure it ends with punctuation
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Add period only if sentence doesn't already end with punctuation
            if sentence[-1] not in ".!?":
                sentence += "."

            result.append(sentence)

        # Join all sentences with spaces
        return " ".join(result)


class ImarSvoDataset(SourceDataset[ImarSvoRow]):
    """Loader for the IMAR (SVO) dataset file."""

    def __init__(self, *paths: Path):
        super().__init__(SourceDatasetEnum.IMAR_SVO, *paths)

    def load(self) -> List[ImarSvoRow]:
        records: List[ImarSvoRow] = []
        for p in self.paths:
            with open(p, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row_dict in reader:
                    row = ImarSvoRow(
                        row_id=row_dict.get("id", ""),
                        initial=row_dict.get("initial", ""),
                        intermediate=row_dict.get("intermediate", ""),
                        final=row_dict.get("final", ""),
                    )
                    records.append(row)
        return records
