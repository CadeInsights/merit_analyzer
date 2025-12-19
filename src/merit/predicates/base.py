"""Base predicate classes and result types."""

import inspect
import logging
from uuid import UUID, uuid4
from functools import wraps

from typing import Any, Protocol, overload, Callable, Awaitable, cast
from pydantic import BaseModel, field_serializer, SerializationInfo, Field

logger = logging.getLogger(__name__)

# Protocols for predicate callables

class SyncPredicate(Protocol):
    """Callable protocol for predicate functions.

    A `Predicate` compares an ``actual`` value to a ``reference`` value, optionally
    using additional ``context`` and configuration flags, and returns a
    :class:`~merit.predicates.base.PredicateResult`.

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
        Whether to enforce strict comparison semantics (predicate-specific).
    metrics
        Optional list used to accumulate metric objects produced during the check.

    Returns
    -------
    PredicateResult
        The check outcome and metadata.
    """

    def __call__(
        self,
        actual: Any,
        reference: Any,
        context: str | None = None,
        strict: bool = True,
        metrics: list | None = None,
        case_id: UUID | None = None,
    ) -> "PredicateResult": ...


class AsyncPredicate(Protocol):
    """Callable protocol for predicate functions.

    A `Predicate` compares an ``actual`` value to a ``reference`` value, optionally
    using additional ``context`` and configuration flags, and returns a
    :class:`~merit.predicates.base.PredicateResult`.

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
        Whether to enforce strict comparison semantics (predicate-specific).
    metrics
        Optional list used to accumulate metric objects produced during the check.

    Returns
    -------
    PredicateResult
        The check outcome and metadata.
    """

    async def __call__(
        self,
        actual: Any,
        reference: Any,
        context: str | None = None,
        strict: bool = True,
        metrics: list | None = None,
        case_id: UUID | None = None,
    ) -> "PredicateResult": ...

Predicate = AsyncPredicate | SyncPredicate


# Models for metadata and result

class PredicateMetadata(BaseModel):
    """Metadata describing how a predicate was executed.

    This model is attached to :class:`~merit.predicates.base.PredicateResult` and is
    intended to make results self-describing and debuggable.

    Notes
    -----
    - If ``predicate_name`` / ``merit_name`` are not provided, they may be
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
        Strictness flag forwarded to the predicate implementation.
    predicate_name
        Name of the predicate callable (usually the function name).
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
    predicate_name: str | None = None
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
        Auto-fill the predicate_name and merit_name fields if not provided.
        """
        if self.predicate_name or self.merit_name:
            return

        frame = inspect.currentframe()

        if frame is None:
            logger.warning("No frame found for predicate_name and merit_name")
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

            if self.predicate_name is None:
                self.predicate_name = func_name

            if self.merit_name is None and func_name.startswith("merit_"):
                self.merit_name = func_name

            if self.predicate_name and self.merit_name:
                break

            frame = frame.f_back



class PredicateResult(BaseModel):
    """Result of a single predicate evaluation.

    The result carries a boolean outcome (`value`), optional human-readable
    details (`message`), and structured metadata about the predicate execution.

    Attributes
    ----------
    id
        Unique identifier for this result instance.
    predicate
        Metadata describing inputs and configuration used for the check.
    confidence
        Confidence score in ``[0, 1]`` (predicate-specific semantics).
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
    case_id: UUID | None = None
    predicate_metadata: PredicateMetadata
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


def _filter_supported_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return only kwargs that `fn` can accept."""
    sig = inspect.signature(fn)
    params = sig.parameters
    accepts_varkw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_varkw:
        return kwargs

    return {k: v for k, v in kwargs.items() if k in params}


@overload
def predicate(func: Callable[[Any, Any], Awaitable[bool]]) -> AsyncPredicate: ...


@overload
def predicate(func: Callable[[Any, Any], bool]) -> SyncPredicate: ...


def predicate(func: Callable[[Any, Any], bool] | Callable[[Any, Any], Awaitable[bool]]) -> Predicate:
    """Decorator to convert a simple comparison function into a full Predicate.
    
    Wraps a function that takes (actual, reference) -> bool and converts it
    to follow the Predicate protocol, which includes optional context, strict,
    and metrics parameters and returns a PredicateResult.
    
    Args:
        func: A function that takes (actual, reference) and returns bool.
              Can be sync or async.
        
    Returns:
        A predicate callable following the Predicate protocol (SyncPredicate or AsyncPredicate).
        
    Example:
        >>> @predicate
        >>> def equals(actual, reference):
        >>>     return actual == reference
        >>>
        >>> result = equals(5, 5)
        >>> assert result.value is True
    """
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(
            actual: Any,
            reference: Any,
            context: str | None = None,
            strict: bool = True,
            metrics: list | None = None,
            case_id: UUID | None = None,
        ) -> PredicateResult:
            extra = _filter_supported_kwargs(
                func,
                {
                    "context": context,
                    "strict": strict,
                    "metrics": metrics,
                    "case_id": case_id,
                },
            )
            result = await cast(Any, func)(actual, reference, **extra)
            predicate_result = PredicateResult(
                predicate_metadata=PredicateMetadata(
                    actual=str(actual),
                    reference=str(reference),
                    context=context,
                    strict=strict,
                ),
                case_id=case_id,
                value=bool(result),
            )
            predicate_result.predicate_metadata.predicate_name = func.__name__
            return predicate_result
        return cast(AsyncPredicate, async_wrapper)
    else:
        @wraps(func)
        def sync_wrapper(
            actual: Any,
            reference: Any,
            context: str | None = None,
            strict: bool = True,
            metrics: list | None = None,
            case_id: UUID | None = None,
        ) -> PredicateResult:
            extra = _filter_supported_kwargs(
                func,
                {
                    "context": context,
                    "strict": strict,
                    "metrics": metrics,
                    "case_id": case_id,
                },
            )
            result = cast(Any, func)(actual, reference, **extra)
            predicate_result = PredicateResult(
                predicate_metadata=PredicateMetadata(
                    actual=str(actual),
                    reference=str(reference),
                    context=context,
                    strict=strict,
                ),
                case_id=case_id,
                value=bool(result),
            )
            predicate_result.predicate_metadata.predicate_name = func.__name__
            return predicate_result
        return cast(SyncPredicate, sync_wrapper)
