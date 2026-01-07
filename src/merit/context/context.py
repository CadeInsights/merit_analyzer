from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, TYPE_CHECKING, List

if TYPE_CHECKING:
    from merit.assertions.base import AssertionResult
    from merit.metrics.base import Metric


@dataclass(frozen=True, slots=True)
class TestContext:
    """Execution context for a single discovered test item.

    This object holds metadata about the currently executing test item (read-only
    identifying information) and aggregates results produced while executing that
    item (e.g., assertion results).

    Attributes
    ----------
    test_item_name : str | None
        Display name for the test item (e.g., function/case name).
    test_item_group_name : str | None
        Optional grouping label (e.g., suite/class/collection name).
    test_item_module_path : str | None
        Import/module path or file path used to locate the test item.
    test_item_tags : list[str]
        Tags attached to the test item (used for filtering and reporting).
    test_item_params : list[str]
        Parameter values/labels for parametrized test items.
    test_item_id_suffix : str | None
        Optional extra suffix appended to an item id to ensure uniqueness.
    assertion_results : list[AssertionResult]
        Assertion results produced while executing the test item.
    """
    # read
    test_item_name: str | None = None
    test_item_group_name: str | None = None
    test_item_module_path: str | None = None
    test_item_tags: list[str] = field(default_factory=list)
    test_item_params: list[str] = field(default_factory=list)
    test_item_id_suffix: str | None = None

    # write
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