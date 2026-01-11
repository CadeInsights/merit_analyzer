"""Test case definitions and decorators."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Generic
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, TypeAdapter
from pydantic_core import SchemaValidator
from typing_extensions import TypeVar

from merit.testing.parametrize import parametrize


RefsT = TypeVar("RefsT", default=dict[str, Any])


class Case(BaseModel, Generic[RefsT]):
    """Container for a single test case inputs and references.

    Attributes:
    ----------
    id : UUID
        Unique identifier for the test case, defaults to a new UUID.
    tags : set[str]
        Set of tags for filtering or categorization of the test case.
    metadata : dict[str, str | int | float | bool | None]
        Arbitrary key-value pairs for additional context or reporting.
    references : RefsT, optional
        Reference data used for validation or comparison during testing.
    sut_input_values : dict[str, Any]
        Input arguments to be passed to the System Under Test (SUT).
    """

    id: UUID = Field(default_factory=uuid4)
    tags: set[str] = Field(default_factory=set)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    references: RefsT | None = None
    sut_input_values: dict[str, Any] = Field(default_factory=dict)


def valididate_cases_for_sut(
    cases: Sequence[Case[RefsT]], sut: Callable[..., Any], raise_on_error: bool = True
) -> Sequence[Case[RefsT]]:
    """Validate that all cases match the signature of the System Under Test.

    This function uses Pydantic to inspect the SUT's signature and ensures that
    the `sut_input_values` in each case are compatible with what the SUT expects.

    Parameters
    ----------
    cases : Sequence[Case[RefsT]]
        A collection of test cases to validate.
    sut : Callable[..., Any], optional
        The System Under Test to validate against.

    Returns:
    -------
    bool
        True if all cases are valid for the given SUT.

    Raises:
    ------
    ValidationError
        If any case's input values do not match the SUT's signature.
    """
    valid_cases = []
    schema = TypeAdapter(sut).core_schema
    arg_schema = schema.get("arguments_schema", None)
    if arg_schema:
        validator = SchemaValidator(arg_schema)  # type: ignore[arg-type]
        for case in cases:
            input_values = case.sut_input_values or {}
            try:
                validator.validate_python(input_values)
                valid_cases.append(case)
            except Exception as e:
                if raise_on_error:
                    raise e
                continue
    return valid_cases


def iter_cases(cases: Sequence[Case[RefsT]]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to run a test function for each case in the provided sequence.

    Parameters
    ----------
    cases : Sequence[Case[RefsT]]
        The sequence of test cases to iterate over.

    Returns:
    -------
    Callable
        A decorator that applies parametrization to the target function.
    """
    ids = [str(c.id) for c in cases]
    return parametrize("case", cases, ids=ids)
