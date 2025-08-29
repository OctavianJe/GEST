from __future__ import annotations

import inspect
from typing import Type

import pytest
from pydantic import Field

from gest.common.helpers.field_names import FieldNameMixin, _FieldAccessor


class Simple(FieldNameMixin):
    foo: int
    bar: str


class WithAlias(FieldNameMixin):
    x: int = Field(alias="y")
    z: str


MODELS: list[Type[FieldNameMixin]] = [Simple, WithAlias]


@pytest.mark.parametrize(
    "model, field_name, expected",
    [
        (Simple, "foo", "foo"),
        (Simple, "bar", "bar"),
        (WithAlias, "x", "y"),
        (WithAlias, "z", "z"),
    ],
)
def test_accessor_returns_correct_string(model, field_name, expected):
    """Checks that for each field (or aliased field) on a model,
    model.fields().<field_name> returns the correct string."""
    accessor = model.fields()
    assert getattr(accessor, field_name) == expected


def test_accessor_type_and_identity():
    """Ensures that Simple.fields() returns a _FieldAccessor instance
    and that it is memoized (same instance on repeated calls)."""
    accessor1 = Simple.fields()
    accessor2 = Simple.fields()
    assert isinstance(accessor1, _FieldAccessor)
    assert accessor1 is accessor2


def test_invalid_attr_raises():
    """Verifies that accessing a non-existent field on the accessor
    raises an AttributeError mentioning the invalid field and model."""
    msg_re = r"not_a_field.*Simple"
    with pytest.raises(AttributeError, match=msg_re):
        _ = Simple.fields().not_a_field


@pytest.mark.parametrize("model", MODELS)
def test_all_declared_fields_roundtrip(model):
    """Every declared field (or alias) round-trips to a str."""
    accessor = model.fields()
    for field in accessor._model.model_fields:
        assert isinstance(getattr(accessor, field), str)


def test_fields_signature_is_stable():
    """
    Public API: `FieldNameMixin.fields()` must be callable with *no* positional
    arguments when accessed from the class.
    """
    sig = inspect.signature(FieldNameMixin.fields)
    assert len(sig.parameters) == 0
