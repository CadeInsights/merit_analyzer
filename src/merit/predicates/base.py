"""Base predicate classes and result types."""

from asyncio import iscoroutinefunction
from collections.abc import Awaitable, Callable
from functools import wraps
from inspect import Parameter, signature
from typing import Any, Protocol, cast, overload

from pydantic import BaseModel, Field

from merit.context import PREDICATE_RESULTS_COLLECTOR


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

    @staticmethod
    def __call__(
        actual: Any,
        reference: Any,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
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

    @staticmethod
    async def __call__(
        actual: Any,
        reference: Any,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> "PredicateResult": ...


Predicate = AsyncPredicate | SyncPredicate


# Data model for predicate results


class PredicateResult(BaseModel):
    """Result of a single predicate evaluation.

    The result carries a boolean outcome (`value`), optional human-readable
    details (`message`), and structured metadata about the predicate execution.

    Attributes:
    ----------
    actual
        Observed value produced by the system under test.
    reference
        Predefined value to compare against.
    name
        Name of the predicate function.
    strict
        Whether to enforce strict comparison semantics (predicate-specific).
    confidence
        Confidence score in ``[0, 1]`` (predicate-specific semantics).
    value
        Boolean outcome of the check.
    message
        Optional details about the outcome (e.g. mismatch explanation).

    Notes:
    -----
    - ``bool(result)`` is equivalent to ``result.value``.
    - ``repr(result)`` returns JSON.
    """

    # Metadata
    actual: str
    reference: str
    name: str
    strict: bool = True
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    # Result
    value: bool
    message: str | None = None

    def __repr__(self) -> str:
        return self.model_dump_json(indent=2)

    def __bool__(self) -> bool:
        collector = PREDICATE_RESULTS_COLLECTOR.get()
        if collector is not None:
            collector.append(self)

        return self.value


# Decorator for predicate functions


@overload
def predicate(func: Callable[..., Awaitable[bool]]) -> AsyncPredicate: ...


@overload
def predicate(func: Callable[..., bool]) -> SyncPredicate: ...


def predicate(func: Callable[..., bool] | Callable[..., Awaitable[bool]]) -> Predicate:
    """Decorate a predicate function so it returns a :class:`PredicateResult`.

    In Merit, all PredicateResults evaluated inside ``assert`` statements
    are collected into :class:`~merit.assertions.base.AssertionResult`.
    That helps with error analysis, reporting and aggregation.

    The decorated function must return a boolean value, and take at least
    two arguments:
    - first positional argument (or ``actual`` keyword argument) will be parsed
    as the actual value to compare against the reference value.
    - second positional argument (or ``reference`` keyword argument) will be parsed
    as the reference value to compare against the actual value.

    Optional keyword arguments:
    - ``strict``: whether to enforce strict comparison semantics (predicate-specific).
    - ``confidence``: confidence score in ``[0, 1]`` (predicate-specific semantics).
    - ``message``: optional human-readable details about the outcome (e.g. mismatch explanation).

    Examples:
    --------
    >>> @predicate
    ... def is_even(actual: int, reference: int) -> bool:
    ...     return actual % 2 == 0
    >>> assert is_even(1, 2)
    >>> @predicate
    ... def within_tolerance(
    ...     actual: float,
    ...     reference: float,
    ...     *,
    ...     tolerance: float = 0.1,
    ...     strict: bool = True,
    ...     confidence: float = 1.0,
    ... ) -> bool:
    ...     delta = abs(actual - reference)
    ...     if strict:
    ...         return delta <= tolerance
    ...     return delta <= 2 * tolerance
    >>> assert within_tolerance(10.05, 10.0, tolerance=0.01, strict=False, confidence=0.65)
    """
    # import time signature check
    sig = signature(func)
    params = sig.parameters
    kinds = [p.kind for p in params.values()]
    positional_slots = sum(
        k in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD) for k in kinds
    )
    accepts_actual_reference = (Parameter.VAR_KEYWORD in kinds) or (
        "actual" in params
        and params["actual"].kind is not Parameter.POSITIONAL_ONLY
        and "reference" in params
        and params["reference"].kind is not Parameter.POSITIONAL_ONLY
    )
    if not (Parameter.VAR_POSITIONAL in kinds or positional_slots >= 2 or accepts_actual_reference):
        msg = f"""@predicate function '{func.__name__}' must accept 'actual' and 'reference'
            as the first two positional args, or as keywords ('actual=', 'reference=').
            Got signature: {sig}"""
        raise TypeError(msg)

    if iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> PredicateResult:
            binded_args = signature(AsyncPredicate.__call__).bind(*args, **kwargs).arguments
            result = await func(*args, **kwargs)
            return PredicateResult(
                value=bool(result),
                message=binded_args.get("kwargs", {}).get("message", None),
                actual=str(binded_args["actual"]),
                reference=str(binded_args["reference"]),
                name=func.__name__,
                strict=binded_args.get("kwargs", {}).get("strict", True),
                confidence=binded_args.get("kwargs", {}).get("confidence", 1.0),
            )

        return cast("AsyncPredicate", async_wrapper)

    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> PredicateResult:
        binded_args = signature(SyncPredicate.__call__).bind(*args, **kwargs).arguments
        result = func(*args, **kwargs)
        return PredicateResult(
            value=bool(result),
            message=binded_args.get("kwargs", {}).get("message", None),
            actual=str(binded_args["actual"]),
            reference=str(binded_args["reference"]),
            name=func.__name__,
            strict=binded_args.get("kwargs", {}).get("strict", True),
            confidence=binded_args.get("kwargs", {}).get("confidence", 1.0),
        )

    return cast("SyncPredicate", sync_wrapper)
