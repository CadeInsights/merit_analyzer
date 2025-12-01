"""Module for defining test cases and case sets.
"""

from collections.abc import Callable
from typing import Any, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


InputsT = TypeVar("InputsT")
OutputsT = TypeVar("OutputsT")


class Case(BaseModel):
    """A single test case with inputs and optional case-level assertions.

    Attributes:
    ----------
    uuid: UUID
        Unique identifier for the test case.
    input: InputsT
        The input data for the test case.
    assertions: list | None
        Optional case-level assertions (evaluated in addition to suite-level).
    metadata: dict | None
        Optional metadata associated with the test case.
    """

    uuid: UUID = Field(default_factory=uuid4)
    input: InputsT
    assertions: list | None = None
    metadata: dict | None = None

    def execute(self, func: Callable[[Any], Any]) -> Any:
        """Execute the test case synchronously using the provided function."""
        return func(self.input)
