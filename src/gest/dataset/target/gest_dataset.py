import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Generic, List, Optional, TypeVar

import pandas as pd
from pydantic import ConfigDict, Field

from gest.common.helpers.field_names import FieldNameMixin
from gest.data.gest import GEST
from gest.dataset.source.source_dataset_enum import SourceDatasetEnum

T = TypeVar("T", bound=FieldNameMixin)


class BaseCSVDataset(ABC, Generic[T]):
    """Abstract base for CSV-backed datasets."""

    def __init__(self, csv_path: Path, columns: list[str]):
        self.path = csv_path
        self._columns = columns
        if csv_path.exists():
            self.df = self.reload()
        else:
            self.df = pd.DataFrame(columns=self._columns)

    def to_csv(self) -> None:
        """Persist the current DataFrame to the CSV file."""
        self.df.to_csv(self.path, index=False, quoting=csv.QUOTE_ALL)

    def reload(self) -> pd.DataFrame:
        """Reload the CSV from disk."""
        self.df = pd.read_csv(self.path, quoting=csv.QUOTE_ALL, engine="python")
        return self.df

    def add_row(self, row) -> None:
        """Append a row in-memory."""
        values = self._serialize_row(row)
        self.df.loc[len(self.df)] = values

    def append_row_to_csv(self, row) -> None:
        """
        Append a row to disk. Creates the file (and parent dirs) if needed, and writes the header only for a new/empty file.
        Subclasses can override for idempotency logic.
        """
        values = self._serialize_row(row)
        single_df = pd.DataFrame([values], columns=self._columns)

        file_has_content = self.path.exists() and self.path.stat().st_size > 0

        single_df.to_csv(
            self.path,
            mode="a" if file_has_content else "w",
            index=False,
            header=not file_has_content,
            quoting=csv.QUOTE_ALL,
        )

    def intersect_as(
        self,
        other: "BaseCSVDataset",
        keys: list[str],
        ctor: Callable[..., Any],
    ) -> List[Any]:
        """
        Generic intersection over the given key columns.
        Returns a list of objects constructed by `ctor`, which is called with
        keyword args matching the column names in `keys`.

        Example:
            keys = ["dataset", "id"]
            ctor = lambda dataset, id: DatasetId(dataset=SourceDatasetEnum(dataset), id=str(id))
        """
        if not keys:
            return []

        left = self.df[keys].drop_duplicates()
        right = other.df[keys].drop_duplicates()
        merged = left.merge(right, on=keys, how="inner")

        out: List[Any] = []
        for _, row in merged.iterrows():
            kwargs = {k: row[k] for k in keys}
            out.append(ctor(**kwargs))
        return out

    @abstractmethod
    def _serialize_row(self, row: T) -> list:
        """Convert a row model to a list of column values."""
        ...


class DatasetId(FieldNameMixin):
    """Base target dataset unique identifier."""

    dataset: SourceDatasetEnum = Field(
        ..., description="Name of the originating dataset"
    )
    id: str = Field(
        ..., description="Unique sample identifier within the source dataset"
    )

    model_config = ConfigDict(frozen=True)


class GestRow(DatasetId):
    """Schema for a single row in the GEST master dataset."""

    text: str = Field(..., description="Flattened / cleaned caption or transcript")
    gest: GEST = Field(..., description="Annotations in GEST format")


class GestDataset(BaseCSVDataset[GestRow]):
    """Wrapper around a CSV-backed `pandas.DataFrame` that stores the GEST data."""

    _columns = list(GestRow.model_fields.keys())

    def __init__(self, csv_path: Path):
        super().__init__(csv_path=csv_path, columns=self._columns)

    ## Shared utility
    @staticmethod
    def _to_compact_json(data: GEST) -> str:
        """Serialize GEST model to compact JSON string."""
        return json.dumps(
            data.model_dump(exclude_none=True),
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def _serialize_row(self, row: GestRow) -> list:
        """Turn a GestRow into a list of column values."""
        return [
            row.dataset,
            row.id,
            row.text,
            self._to_compact_json(row.gest),
        ]

    ## Query helpers
    def find_row_index(
        self,
        source_dataset: SourceDatasetEnum,
        id: str,
    ) -> Optional[int]:
        """
        Locate the DataFrame index of the row with the given source_dataset and id.
        Returns the index if found, otherwise None.
        """
        dataset_col = GestRow.fields().dataset
        id_col = GestRow.fields().id

        mask = (self.df[dataset_col] == source_dataset.value) & (self.df[id_col] == id)
        matches = self.df.index[mask].tolist()
        return matches[0] if matches else None

    def intersect_dataset_ids(self, other: "GestDataset") -> List[DatasetId]:
        """
        Return the intersection of (dataset, id) keys as a list of DatasetId objects.
        """
        ds_col = GestRow.fields().dataset
        id_col = GestRow.fields().id

        def _ctor(**kw) -> DatasetId:
            return DatasetId(
                dataset=SourceDatasetEnum(kw[ds_col]),
                id=str(kw[id_col]),
            )

        return self.intersect_as(other, [ds_col, id_col], _ctor)


class GestBlacklistRow(FieldNameMixin):
    """Schema for a single row in the GEST blacklist dataset."""

    dataset: SourceDatasetEnum = Field(
        ..., description="Name of the originating dataset"
    )
    id: str = Field(
        ..., description="Unique sample identifier within the source dataset"
    )
    reason: str = Field(..., description="Reason for blacklisting this row")


class GestBlacklistDataset(BaseCSVDataset[GestBlacklistRow]):
    """Wrapper around a CSV-backed `pandas.DataFrame` that stores the GEST blacklist data."""

    _columns = list(GestBlacklistRow.model_fields.keys())

    def __init__(self, csv_path: Path):
        super().__init__(csv_path=csv_path, columns=self._columns)

    ## Shared utility
    def _serialize_row(self, row: GestBlacklistRow) -> list:
        """Turn a GestBlacklistRow into a list of column values."""
        return [
            row.dataset,
            row.id,
            row.reason,
        ]

    def append_row_to_csv(self, row: GestBlacklistRow) -> None:
        """Append a GestBlacklistRow to disk if not already present."""
        if self.is_blacklisted(row.dataset, row.id):
            return
        super().append_row_to_csv(row)

    ## Query helpers
    def is_blacklisted(self, dataset: SourceDatasetEnum, id: str) -> bool:
        """
        Check if a row with the given source_dataset and id is blacklisted.
        Returns True if blacklisted, otherwise False.
        """
        dataset_col = GestBlacklistRow.fields().dataset
        id_col = GestBlacklistRow.fields().id

        mask = (self.df[dataset_col] == dataset.value) & (self.df[id_col] == id)
        return bool(mask.any())
