"""Base assertion classes and result types."""

import inspect
import logging
from uuid import UUID, uuid4

from typing import Any, Protocol
from pydantic import BaseModel, field_serializer, SerializationInfo, Field

logger = logging.getLogger(__name__)


class Assertion(Protocol):
    """All callables used for assertions must conform to this protocol."""

    def __call__(
        self, actual: Any, reference: Any, context: str | None = None, strict: bool = True, metrics: list | None = None
    ) -> "AssertionResult": ...


class AssertionMetadata(BaseModel):
    """Metadata for an assertion.

    Attributes:
    ----------
    assertion_name: str | None
        Name of the assertion that was evaluated
    merit_name: str | None
        Name of the merit that was evaluated
    actual: str
        Actual output of the test case
    reference: str
        Reference output of the test case
    strict: bool
        Whether to use strict comparison
    """
    # Identifiers
    assertion_id: UUID = Field(default_factory=uuid4)
    assertion_name: str | None = None
    merit_name: str | None = None

    # Assertion inputs
    actual: str
    reference: str
    context: str | None = None
    strict: bool = True

    @field_serializer("actual", "reference")
    def _truncate(self, v: str, info: SerializationInfo) -> str:
        """Truncate the values in the actual and reference fields to 50 characters."""
        ctx = info.context or {}
        if ctx.get("truncate"):
            max_len = 50
            if len(v) <= max_len:
                return v
            return v[:max_len] + "..."
        return v

    def model_post_init(self, __context) -> None:
        """
        Auto-fill the assertion_name and merit_name fields if not provided.
        """
        if self.assertion_name or self.merit_name:
            return

        frame = inspect.currentframe()

        if frame is None:
            logger.warning("No frame found for assertion_name and merit_name")
            return

        frame = frame.f_back

        while frame:
            func_name = frame.f_code.co_name
            module_name = frame.f_globals.get("__name__", "")

            if module_name.startswith("pydantic"):
                frame = frame.f_back
                continue

            if func_name in {"__init__", "model_post_init", "_get_caller_and_merit_names"}:
                frame = frame.f_back
                continue

            if self.assertion_name is None:
                self.assertion_name = func_name

            if self.merit_name is None and func_name.startswith("merit_"):
                self.merit_name = func_name

            if self.assertion_name and self.merit_name:
                break

            frame = frame.f_back


class AssertionResult(BaseModel):
    """Result of evaluating an assertion on a test case.

    Attributes:
    ----------
    metadata: AssertionMetadata
        Metadata for the assertion result
    passed: bool
        Whether the assertion passed
    score: float | None
        Numerical score (0.0 to 1.0), optional
    confidence: float | None
        Confidence level (0.0 to 1.0), optional
    message: str | None
        Optional message explaining the result
    """

    metadata: AssertionMetadata
    passed: bool
    confidence: float | None = None
    message: str | None = None

    def __repr__(self) -> str:
        return self.model_dump_json(
            indent=2,
            exclude_none=True,
            context={"truncate": True},
        )

    def __bool__(self) -> bool:
        return self.passed


# TODO: check with Daniel if we still need AssertionFailedError

# class AssertionFailedError(AssertionError):
#     """AssertionError with attached AssertionResult."""

#     def __init__(self, result: AssertionResult):
#         self.assertion_result = result
#         message = f"{result.metadata.name} failed"
#         if result.message:
#             message += f": {result.message}"
#         super().__init__(message)
