"""Base predicate classes and result types."""

import inspect
import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, Protocol, cast, overload
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, SerializationInfo, field_serializer

from merit.context import PREDICATE_RESULTS_COLLECTOR, TEST_CONTEXT


logger = logging.getLogger(__name__)

# Protocols for predicate callables


class SyncPredicate(Protocol):
    """Callable protocol for predicate functions.

    A `Predicate` compares an ``actual`` value to a ``reference`` value, optionally
    using configuration flags, and returns a
    :class:`~merit.predicates.base.PredicateResult`.

    Parameters
    ----------
    actual
        Observed value produced by the system under test.
    reference
        Predefined value to compare against.
    strict
        Whether to enforce strict comparison semantics (predicate-specific).

    Returns:
    -------
    PredicateResult
        The check outcome and metadata.
    """

    def __call__(
        self,
        actual: Any,
        reference: Any,
        strict: bool = False,
    ) -> "PredicateResult": ...


class AsyncPredicate(Protocol):
    """Callable protocol for predicate functions.

    A `Predicate` compares an ``actual`` value to a ``reference`` value, optionally
    using configuration flags, and returns a
    :class:`~merit.predicates.base.PredicateResult`.

    Parameters
    ----------
    actual
        Observed value produced by the system under test.
    reference
        Predefined value to compare against.
    strict
        Whether to enforce strict comparison semantics (predicate-specific).

    Returns:
    -------
    PredicateResult
        The check outcome and metadata.
    """

    async def __call__(
        self,
        actual: Any,
        reference: Any,
        strict: bool = False,
    ) -> "PredicateResult": ...


Predicate = AsyncPredicate | SyncPredicate


# Models for metadata and result


class PredicateMetadata(BaseModel):
    """Metadata describing how a predicate was executed.

    This model is attached to :class:`~merit.predicates.base.PredicateResult` and is
    intended to make results self-describing and debuggable.

    Notes:
    -----
    - If ``predicate_name`` / ``merit_name`` are not provided, they may be
      auto-filled in :meth:`model_post_init` by inspecting the call stack.

    Attributes:
    ----------
    actual
        String representation of the observed value.
    reference
        String representation of the expected/ground-truth value.
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


class PredicateResult(BaseModel):
    """Result of a single predicate evaluation.

    The result carries a boolean outcome (`value`), optional human-readable
    details (`message`), and structured metadata about the predicate execution.

    Attributes:
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

    Notes:
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

    def model_post_init(self, __context: Any) -> None:
        """Auto-fill the predicate_name and merit_name fields if not provided."""
        test_ctx = TEST_CONTEXT.get()
        if test_ctx is not None:
            if test_ctx.item.id_suffix:
                self.case_id = UUID(test_ctx.item.id_suffix)
            if test_ctx.item.name:
                self.predicate_metadata.merit_name = test_ctx.item.name

        collector = PREDICATE_RESULTS_COLLECTOR.get()
        if collector is not None:
            collector.append(self)


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


def predicate(
    func: Callable[[Any, Any], bool] | Callable[[Any, Any], Awaitable[bool]],
) -> Predicate:
    """Decorator to convert a simple comparison function into a full Predicate.

    Wraps a function that takes (actual, reference) -> bool and converts it
    to follow the Predicate protocol, which includes the optional ``strict``
    flag and returns a PredicateResult.

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
            strict: bool = False,
        ) -> PredicateResult:
            extra = _filter_supported_kwargs(
                func,
                {
                    "strict": strict,
                },
            )
            result = await cast("Any", func)(actual, reference, **extra)
            predicate_result = PredicateResult(
                predicate_metadata=PredicateMetadata(
                    actual=str(actual),
                    reference=str(reference),
                    strict=strict,
                ),
                value=bool(result),
            )
            predicate_result.predicate_metadata.predicate_name = func.__name__
            return predicate_result

        return cast("AsyncPredicate", async_wrapper)

    @wraps(func)
    def sync_wrapper(
        actual: Any,
        reference: Any,
        strict: bool = False,
    ) -> PredicateResult:
        extra = _filter_supported_kwargs(
            func,
            {
                "strict": strict,
            },
        )
        result = cast("Any", func)(actual, reference, **extra)
        predicate_result = PredicateResult(
            predicate_metadata=PredicateMetadata(
                actual=str(actual),
                reference=str(reference),
                strict=strict,
            ),
            value=bool(result),
        )
        predicate_result.predicate_metadata.predicate_name = func.__name__
        return predicate_result

    return cast("SyncPredicate", sync_wrapper)
