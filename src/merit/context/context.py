from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID


if TYPE_CHECKING:
    from merit.assertions.base import AssertionResult
    from merit.metrics_.base import Metric, MetricResult
    from merit.predicates.base import PredicateResult
    from merit.testing.models import MeritRun, MeritTestDefinition


@dataclass(frozen=True, slots=True)
class TestContext:
    """Execution context for a single discovered test item.

    This object holds a reference to the currently executing test item and
    aggregates results produced while executing that item (e.g., assertion results).

    Attributes:
    ----------
    item : TestItem
        The test item being executed.
    assertion_results : list[AssertionResult]
        Assertion results produced while executing the test item.
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    item: MeritTestDefinition
    execution_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class ResolverContext:
    """Context for resource resolution.

    Attributes:
    ----------
    consumer_name
        Name/identifier of the component currently resolving/consuming a resource.
    """

    consumer_name: str | None = None


ASSERTION_RESULTS_COLLECTOR: ContextVar[list[AssertionResult] | None] = ContextVar(
    "assertion_results_collector", default=None
)
PREDICATE_RESULTS_COLLECTOR: ContextVar[list[PredicateResult] | None] = ContextVar(
    "predicate_results_collector", default=None
)

METRIC_RESULTS_COLLECTOR: ContextVar[list[MetricResult] | None] = ContextVar(
    "metric_results_collector", default=None
)

TEST_CONTEXT: ContextVar[TestContext | None] = ContextVar("test_context", default=None)
RESOLVER_CONTEXT: ContextVar[ResolverContext | None] = ContextVar("resolver_context", default=None)
METRIC_CONTEXT: ContextVar[list[Metric] | None] = ContextVar("metric_context", default=None)
MERIT_RUN_CONTEXT: ContextVar[MeritRun | None] = ContextVar("merit_run_context", default=None)


def get_test_context() -> TestContext | None:
    """Get the current test context, or None if not in a test."""
    return TEST_CONTEXT.get()


def get_merit_run() -> MeritRun | None:
    """Get the current merit run, or None if not in a run."""
    return MERIT_RUN_CONTEXT.get()


@contextmanager
def assertions_collector(ctx: list[AssertionResult]) -> Iterator[None]:
    token = ASSERTION_RESULTS_COLLECTOR.set(ctx)
    try:
        yield
    finally:
        ASSERTION_RESULTS_COLLECTOR.reset(token)


@contextmanager
def predicate_results_collector(ctx: list[PredicateResult]) -> Iterator[None]:
    token = PREDICATE_RESULTS_COLLECTOR.set(ctx)
    try:
        yield
    finally:
        PREDICATE_RESULTS_COLLECTOR.reset(token)


@contextmanager
def metric_results_collector(ctx: list[MetricResult]) -> Iterator[None]:
    token = METRIC_RESULTS_COLLECTOR.set(ctx)
    try:
        yield
    finally:
        METRIC_RESULTS_COLLECTOR.reset(token)


@contextmanager
def test_context_scope(ctx: TestContext) -> Iterator[None]:
    """Temporarily set `TEST_CONTEXT` for the duration of the ``with`` block.

    Parameters
    ----------
    ctx : TestContext
        The context to bind as the current test context.
    """
    token = TEST_CONTEXT.set(ctx)
    try:
        yield
    finally:
        TEST_CONTEXT.reset(token)


@contextmanager
def resolver_context_scope(ctx: ResolverContext) -> Iterator[None]:
    """Temporarily set `RESOLVER_CONTEXT` for the duration of the ``with`` block.

    Parameters
    ----------
    ctx : ResolverContext
        The resolver context to bind as the current resolver context.
    """
    token = RESOLVER_CONTEXT.set(ctx)
    try:
        yield
    finally:
        RESOLVER_CONTEXT.reset(token)


@contextmanager
def metrics(*metrics: Metric) -> Iterator[None]:
    """Attach metrics to the current execution scope via `METRIC_CONTEXT`.

    Parameters
    ----------
    metrics : Metric
        Metrics to expose to the current execution scope.
    """
    metrics_list = list(metrics)
    
    # backwards compatibility with old API
    if len(metrics_list) == 1 and isinstance(metrics_list[0], (list, tuple)):
        metrics_list = list(metrics_list[0])

    token = METRIC_CONTEXT.set(metrics_list)
    try:
        yield
    finally:
        METRIC_CONTEXT.reset(token)


@contextmanager
def merit_run_scope(run: MeritRun) -> Iterator[None]:
    """Temporarily set `MERIT_RUN_CONTEXT` for the duration of the ``with`` block.

    Parameters
    ----------
    run : MeritRun
        The merit run to bind as the current run context.
    """
    token = MERIT_RUN_CONTEXT.set(run)
    try:
        yield
    finally:
        MERIT_RUN_CONTEXT.reset(token)
