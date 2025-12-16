"""Base check classes and result types."""

import inspect
import logging
from uuid import UUID, uuid4

from typing import Any, Protocol
from pydantic import BaseModel, field_serializer, SerializationInfo, Field

logger = logging.getLogger(__name__)

# Protocols for checking callables

class SyncChecker(Protocol):
    """Callable protocol for check functions.

    A `Checker` compares an ``actual`` value to a ``reference`` value, optionally
    using additional ``context`` and configuration flags, and returns a
    :class:`~merit.checkers.base.CheckerResult`.

    Parameters
    ----------
    actual
        Observed value produced by the system under test.
    reference
        Predefined value to compare against.
    context
        Optional context string to help interpret the comparison (e.g. prompt,
        instructions, domain constraints).
    strict
        Whether to enforce strict comparison semantics (checker-specific).
    metrics
        Optional list used to accumulate metric objects produced during the check.

    Returns
    -------
    CheckerResult
        The check outcome and metadata.
    """

    def __call__(
        self, actual: Any, reference: Any, context: str | None = None, strict: bool = True, metrics: list | None = None
    ) -> "CheckerResult": ...


class AsyncChecker(Protocol):
    """Callable protocol for check functions.

    A `Checker` compares an ``actual`` value to a ``reference`` value, optionally
    using additional ``context`` and configuration flags, and returns a
    :class:`~merit.checkers.base.CheckerResult`.

    Parameters
    ----------
    actual
        Observed value produced by the system under test.
    reference
        Predefined value to compare against.
    context
        Optional context string to help interpret the comparison (e.g. prompt,
        instructions, domain constraints).
    strict
        Whether to enforce strict comparison semantics (checker-specific).
    metrics
        Optional list used to accumulate metric objects produced during the check.

    Returns
    -------
    CheckerResult
        The check outcome and metadata.
    """

    async def __call__(
        self, actual: Any, reference: Any, context: str | None = None, strict: bool = True, metrics: list | None = None
    ) -> "CheckerResult": ...

Checker = AsyncChecker | SyncChecker


# Models for metadata and result

class CheckerMetadata(BaseModel):
    """Metadata describing how a check was executed.

    This model is attached to :class:`~merit.checkers.base.CheckerResult` and is
    intended to make results self-describing and debuggable.

    Notes
    -----
    - If ``checker_name`` / ``merit_name`` are not provided, they may be
      auto-filled in :meth:`model_post_init` by inspecting the call stack.

    Attributes
    ----------
    actual
        String representation of the observed value.
    reference
        String representation of the expected/ground-truth value.
    context
        Optional context string used during the check.
    strict
        Strictness flag forwarded to the checker implementation.
    checker_name
        Name of the checker callable (usually the function name).
        Read-only.
    merit_name
        Name of the enclosing "merit" function, if available (e.g. ``merit_*``).
        Read-only.
    """
    # Inputs
    actual: str
    reference: str
    context: str | None = None
    strict: bool = True

    # Auto-filled Identifiers
    checker_name: str | None = None
    merit_name: str | None = None

    @field_serializer("actual", "reference")
    def _truncate(self, v: str, info: SerializationInfo) -> str:
        ctx = info.context or {}
        if ctx.get("truncate"):
            max_len = 50
            return v if len(v) <= max_len else v[:max_len] + "..."
        return v

    def model_post_init(self, __context) -> None:
        """
        Auto-fill the checker_name and merit_name fields if not provided.
        """
        if self.checker_name or self.merit_name:
            return

        frame = inspect.currentframe()

        if frame is None:
            logger.warning("No frame found for checker_name and merit_name")
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

            if self.checker_name is None:
                self.checker_name = func_name

            if self.merit_name is None and func_name.startswith("merit_"):
                self.merit_name = func_name

            if self.checker_name and self.merit_name:
                break

            frame = frame.f_back



class CheckerResult(BaseModel):
    """Result of a single checker evaluation.

    The result carries a boolean outcome (`value`), optional human-readable
    details (`message`), and structured metadata about the check execution.

    Attributes
    ----------
    id
        Unique identifier for this result instance.
    checker
        Metadata describing inputs and configuration used for the check.
    confidence
        Confidence score in ``[0, 1]`` (checker-specific semantics).
    value
        Boolean outcome of the check.
    message
        Optional details about the outcome (e.g. mismatch explanation).

    Notes
    -----
    - ``bool(result)`` is equivalent to ``result.value``.
    - ``repr(result)`` returns JSON with ``None`` fields excluded and
      truncation enabled for long ``actual`` / ``reference`` strings.
    """
    # Metadata
    id: UUID = Field(default_factory=uuid4)
    checker_metadata: CheckerMetadata
    confidence: float = 1.0

    # Result
    value: bool
    message: str | None = None

    def __repr__(self) -> str:
        return self.model_dump_json(
            indent=2,
            exclude_none=True,
            context={"truncate": True},
        )

    def __bool__(self) -> bool:
        return self.value