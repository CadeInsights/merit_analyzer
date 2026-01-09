from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, TYPE_CHECKING, List

if TYPE_CHECKING:
    from merit.assertions.base import AssertionResult
    from merit.metrics.base import Metric
    from merit.testing.discovery import TestItem
    from merit.testing.new_runner import TestNode
    from merit.testing.runner import MeritRun


@dataclass(frozen=True, slots=True)
class TestContext:
    """Execution context for a single discovered test item.

    This object holds a reference to the currently executing test item and
    aggregates results produced while executing that item (e.g., assertion results).

    Attributes
    ----------
    item : TestItem
        The test item being executed.
    assertion_results : list[AssertionResult]
        Assertion results produced while executing the test item.
    """

    item: TestItem
    assertion_results: list[AssertionResult] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ResolverContext:
    """Context for resource resolution.

    Attributes
    ----------
    consumer_name
        Name/identifier of the component currently resolving/consuming a resource.
    """

    consumer_name: str | None = None


TEST_CONTEXT: ContextVar[TestContext | None] = ContextVar("test_context", default=None)
RESOLVER_CONTEXT: ContextVar[ResolverContext | None] = ContextVar("resolver_context", default=None)
ASSERTION_CONTEXT: ContextVar[AssertionResult | None] = ContextVar("assertion_context", default=None)
METRIC_CONTEXT: ContextVar[List[Metric] | None] = ContextVar("metric_context", default=None)
MERIT_RUN_CONTEXT: ContextVar[MeritRun | None] = ContextVar("merit_run_context", default=None)
NODE_CONTEXT: ContextVar[TestNode | None] = ContextVar("node_context", default=None)


def get_test_context() -> TestContext | None:
    """Get the current test context, or None if not in a test."""
    return TEST_CONTEXT.get()


def get_node_context() -> TestNode | None:
    """Get the currently executing test node, or None if not in a test."""
    return NODE_CONTEXT.get()


def get_merit_run() -> MeritRun | None:
    """Get the current merit run, or None if not in a run."""
    return MERIT_RUN_CONTEXT.get()


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
def assertion_context_scope(ctx: AssertionResult) -> Iterator[None]:
    """Temporarily set `ASSERTION_CONTEXT` for the duration of the ``with`` block.

    Parameters
    ----------
    ctx : AssertionResult
        The assertion result object to bind as the current assertion context.
    """
    token = ASSERTION_CONTEXT.set(ctx)
    try:
        yield
    finally:
        ASSERTION_CONTEXT.reset(token)


@contextmanager
def metrics(ctx: List[Metric]) -> Iterator[None]:
    """Attach metrics to the current execution scope via `METRIC_CONTEXT`.

    Parameters
    ----------
    ctx : list[Metric]
        Metrics to expose to the current execution scope.
    """
    token = METRIC_CONTEXT.set(ctx)
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


@contextmanager
def node_context_scope(node: TestNode) -> Iterator[None]:
    """Temporarily set `NODE_CONTEXT` for the duration of the ``with`` block.

    Parameters
    ----------
    node : TestNode
        The test node to bind as the current node context.
    """
    token = NODE_CONTEXT.set(node)
    try:
        yield
    finally:
        NODE_CONTEXT.reset(token)