"""Base assertion classes and result types."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel


if TYPE_CHECKING:
    from merit.testing import Case


class AssertionResult(BaseModel):
    """Result of evaluating an assertion on a test case.

    Attributes:
    ----------
    assertion_name : str
        Name of the assertion that was evaluated
    passed : bool
        Whether the assertion passed
    score : float | None
        Numerical score (0.0 to 1.0), optional
    confidence : float | None
        Confidence level (0.0 to 1.0), optional
    message : str | None
        Optional message explaining the result
    """

    assertion_name: str
    passed: bool
    score: float | None = None
    confidence: float | None = None
    message: str | None = None


class Assertion(ABC):
    """Base class for test assertions.

    The 'name' attribute is automatically set to the class name,
    but can be overridden by defining it explicitly as a class variable.
    """

    def __init_subclass__(cls, **kwargs):
        """Auto-generate name from class name if not provided."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            cls.name = cls.__name__

    @abstractmethod
    def __call__(self, actual: Any, case: "Case") -> AssertionResult:
        """Evaluate the assertion on a test case.

        Parameters
        ----------
        actual : Any
            The actual output from the system under test
        case : Case
            Test case containing expected output

        Returns:
        -------
        AssertionResult
            Result of the assertion evaluation
        """
