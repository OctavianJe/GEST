from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel


@dataclass(frozen=True)
class _FieldAccessor:
    """
    Dynamic helper that exposes model-field names as **str**, honouring aliases.
    """

    _model: type[BaseModel]

    def __getattr__(self, item: str) -> str:
        if item in self._model.model_fields:
            info = self._model.model_fields[item]
            return info.alias or item
        raise AttributeError(f"{item!r} is not a valid field of {self._model.__name__}")


TModel = TypeVar("TModel", bound="FieldNameMixin")


class FieldNameMixin(BaseModel):
    """
    Mixin that provides a cached `fields()` accessor; each attribute access
    yields the field's name (or alias) as a **string**.
    """

    __slots__ = ()

    @classmethod
    @lru_cache(maxsize=256)
    def fields(cls: type[TModel]) -> _FieldAccessor:
        return _FieldAccessor(cls)
