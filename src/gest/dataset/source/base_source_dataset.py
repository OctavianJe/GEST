from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, List, TypeVar

from gest.dataset.source.source_dataset_enum import SourceDatasetEnum


class DataRow(ABC):
    """Common behavior every dataset row must provide."""

    @abstractmethod
    def id(self) -> str: ...

    @abstractmethod
    def text(self) -> str: ...


RowT = TypeVar("RowT", bound=DataRow)


class SourceDataset(ABC, Generic[RowT]):
    """Load one or more external dataset files into a single DataFrame."""

    def __init__(self, name: SourceDatasetEnum, *paths: Path) -> None:
        if not paths:
            raise ValueError("At least one Path must be supplied")
        self.name: SourceDatasetEnum = name
        self.paths: List[Path] = list(paths)
        self.rows: List[RowT] = self.load()

    @abstractmethod
    def load(self) -> List[RowT]: ...
