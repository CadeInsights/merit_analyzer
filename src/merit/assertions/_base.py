"""Base assertion classes and result types."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


def _truncate(value: Any, max_len: int = 30) -> str:
    """Truncate a repr string if too long."""
    s = repr(value)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


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


class AssertionFailedError(AssertionError):
    """AssertionError with attached AssertionResult."""

    def __init__(self, result: AssertionResult):
        self.assertion_result = result
        message = f"{result.assertion_name} failed"
        if result.message:
            message += f": {result.message}"
        super().__init__(message)


class Assertion(ABC):
    """Base class for test assertions.

    The 'name' attribute is automatically set to the class name,
    but can be overridden by defining it explicitly as a class variable.

    Subclasses implement `evaluate()` which returns an AssertionResult.
    Calling the assertion raises AssertionFailedError if the result fails.
    """

    def __init_subclass__(cls, **kwargs):
        """Auto-generate name from class name if not provided."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            cls.name = cls.__name__

    def __init__(self, **kwargs):
        pass

    def __call__(self, actual: Any) -> AssertionResult:
        """Evaluate the assertion and raise on failure.

        Parameters
        ----------
        actual : Any
            The actual output from the system under test

        Returns:
        -------
        AssertionResult
            Result of the assertion evaluation

        Raises:
        ------
        AssertionFailedError
            If the assertion fails (passed=False)
        """
        result = self.evaluate(actual)
        if not result.passed:
            raise AssertionFailedError(result)
        return result

    @abstractmethod
    def evaluate(self, actual: Any) -> AssertionResult:
        """Evaluate the assertion.

        Parameters
        ----------
        actual : Any
            The actual output from the system under test

        Returns:
        -------
        AssertionResult
            Result of the assertion evaluation
        """
