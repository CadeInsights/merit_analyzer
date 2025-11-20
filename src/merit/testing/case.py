"""Module for defining test cases and case sets.
"""

from collections.abc import Callable
from typing import Any, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


InputsT = TypeVar("InputsT")
OutputsT = TypeVar("OutputsT")


class Case(BaseModel):
    """A single test case with inputs, expected output, and actual output.

    Attributes:
    ----------
    uuid: UUID
        Unique identifier for the test case.
    input: InputsT
        The input data for the test case.
    expected_output: OutputsT
        The expected output for the test case.
    metadata: dict | None
        Optional metadata associated with the test case.
    """

    uuid: UUID = Field(default_factory=uuid4)
    input: InputsT
    expected_output: OutputsT
    metadata: dict | None = None

    class Config:
        arbitrary_types_allowed = True

    def execute(self, func: Callable[[Any], Any]) -> Any:
        """Execute the test case synchronously using the provided function."""
        return func(self.input)


class CaseSet(BaseModel):
    """A collection of test cases.

    Attributes:
    ----------
    cases: list[Case]
        List of test cases.
    metadata: dict | None
        Optional metadata for the case set.
    """

    cases: list[Case] = Field(
        default_factory=list,
        description="List of test cases",
    )
    metadata: dict | None = None

    def add(self, case: Case) -> None:
        """Add a case to the set."""
        self.cases.append(case)

    @classmethod
    def from_csv(cls, path: str) -> "CaseSet":
        """Load test cases from a CSV file."""
        # TODO: Implementation

    @classmethod
    def from_json(cls, path: str) -> "CaseSet":
        """Load test cases from a JSON file."""
        # TODO: Implementation
