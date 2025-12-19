"""Test case definitions and decorators.

This module provides:

- A :class:`~merit.testing.case.Case` model used to represent a single test case,
  including tags and optional structured metadata.
- A :func:`~merit.testing.case.iter_cases` decorator factory that attaches a list
  of cases to a function and can optionally validate per-case SUT inputs against
  the function's argument schema.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Generic
from typing_extensions import TypeVar
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, TypeAdapter
from pydantic_core import SchemaValidator


RefsT = TypeVar("RefsT", default=dict[str, Any])


class Case(BaseModel, Generic[RefsT]):
    """Container for a single test case inputs and references.

    Parameters
    ----------
    id:
        UUID.
    tags:
        Labels for filtering or grouping cases.
    metadata:
        Extra scalar information about the case (e.g., "priority", "seed",
        "expected_status"). Values are limited to JSON-like scalar types.
    references:
        Optional reference values required for assertions.
    sut_input_values:
        Optional input values for the SUT.
    """
    id: UUID = Field(default_factory=uuid4)
    tags: set[str] = Field(default_factory=set)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    references: RefsT | None = None
    sut_input_values: dict[str, Any] = Field(default_factory=dict)


def iter_cases(
    cases: Sequence[Case[RefsT]],
    sut_for_inputs_validation: Callable[..., Any] | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Run decorated merit function for each case in the list.

    Parameters
    ----------
    cases:
        List of :class:`~merit.testing.case.Case` objects to attach.
    sut_for_inputs_validation:
        Optional callable. If provided, all cases will be checked 
        for validity against the callable's argument schema.
    """
    if sut_for_inputs_validation:
        schema = TypeAdapter(sut_for_inputs_validation).core_schema
        arg_schema = schema.get('arguments_schema', None)
        if arg_schema:
            validator = SchemaValidator(arg_schema) # type: ignore[arg-type]
            for case in cases:
                input_values = case.sut_input_values or {}
                validator.validate_python(input_values)
        else:
            raise ValueError("No arguments schema found for the function")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # Keep runtime behavior stable: attach a concrete list copy.
        fn.__merit_cases__ = list(cases)
        return fn

    return decorator