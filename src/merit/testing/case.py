"""Test case definitions and decorators."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Generic
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
from pydantic.experimental.arguments_schema import generate_arguments_schema
from pydantic_core import ArgsKwargs, SchemaValidator, ValidationError
from typing_extensions import TypeVar

from merit.testing.decorators import parametrize


RefsT = TypeVar("RefsT", default=dict[str, Any])


# Data model for case values


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
    references : RefsT,
        Reference data used for validation or comparison during testing.
    sut_input_values : dict[str, Any]
        Input arguments to be passed to the System Under Test (SUT).
    """

    model_config = ConfigDict(validate_default=True)

    id: UUID = Field(default_factory=uuid4)
    tags: set[str] = Field(default_factory=set)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    references: RefsT = Field(default_factory=dict)  # type: ignore[assignment]
    sut_input_values: dict[str, Any] = Field(default_factory=dict)


# Validation engine


def validate_cases_for_sut(
    cases: Sequence[Case[RefsT]],
    sut: Callable[..., Any],
    raise_on_invalid: bool = True,
) -> Sequence[Case[RefsT]]:
    """Return only the cases that match the signature of the System Under Test.

    Parameters
    ----------
    cases : Sequence[Case[RefsT]]
        A collection of test cases to validate.
    sut : Callable[..., Any], optional
        The System Under Test to validate against.
    raise_on_invalid : bool, optional
        Whether to raise an exception if any case is invalid. Defaults to True.

    Returns:
    -------
    Sequence[Case[RefsT]]
        The cases that match the signature of the System Under Test.
    """
    valid_cases = []
    schema = generate_arguments_schema(
        sut,
        parameters_callback=(
            lambda index, name, annotation: "skip" if name in {"self", "cls"} else None
        ),
    )
    validator = SchemaValidator(schema)
    for case in cases:
        input_values = case.sut_input_values or {}
        try:
            parsed_args = ArgsKwargs(args=(), kwargs=input_values)
            validator.validate_python(parsed_args)
            # append if valid
            valid_cases.append(case)
        except ValidationError as e:
            if raise_on_invalid:
                raise e
            continue
    return valid_cases


# Iteration decorator for Merit definitions


def iter_cases(*cases: Case[RefsT]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
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
    cases_list = list(cases)

    # backwards compatibility with old API
    if len(cases_list) == 1 and isinstance(cases_list[0], (list, tuple)):
        cases_list = list(cases_list[0])

    ids = [str(c.id) for c in cases_list]
    return parametrize("case", cases_list, ids=ids)
