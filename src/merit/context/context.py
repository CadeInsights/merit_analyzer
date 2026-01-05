from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from merit.metrics.base import Metric
    from merit.assertions.result import AssertionResult


TEST_CONTEXT: ContextVar[TestContext | None] = ContextVar("test_context", default=None)
RESOLVER_CONTEXT: ContextVar[ResolverContext | None] = ContextVar("resolver_context", default=None)
METRICS_CONTEXT: ContextVar[list[Metric] | None] = ContextVar("metrics_context", default=None)


@dataclass(frozen=True, slots=True)
class TestContext:
    """Execution context for a single discovered test item.

    Attributes
    ----------
    test_item_name
        Display name for the test item (e.g., function/case name).
    test_item_group_name
        Optional grouping label (e.g., suite/class/collection name).
    test_item_module_path
        Import/module path or file path used to locate the test item.
    test_item_tags
        Tags attached to the test item (used for filtering and reporting).
    test_item_params
        Parameter values/labels for parametrized test items.
    test_item_id_suffix
        Optional extra suffix appended to an item id to ensure uniqueness.
    collected_assertion_results
        List of assertion results collected within the test item.
    """

    test_item_name: str | None = None
    test_item_group_name: str | None = None
    test_item_module_path: str | None = None
    test_item_tags: list[str] = field(default_factory=list)
    test_item_params: list[str] = field(default_factory=list)
    test_item_id_suffix: str | None = None
    collected_assertion_results: list[AssertionResult] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ResolverContext:
    """Context for resource resolution.

    Attributes
    ----------
    consumer_name
        Name/identifier of the component currently resolving/consuming a resource.
    """

    consumer_name: str | None = None


@contextmanager
def errors_to_metrics(ctx: list[Metric]) -> Iterator[None]:
    token = METRICS_CONTEXT.set(ctx)
    try:
        yield
    finally:
        METRICS_CONTEXT.reset(token)


@contextmanager
def test_context_scope(ctx: TestContext) -> Iterator[None]:
    token = TEST_CONTEXT.set(ctx)
    try:
        yield
    finally:
        TEST_CONTEXT.reset(token)


@contextmanager
def resolver_context_scope(ctx: ResolverContext) -> Iterator[None]:
    token = RESOLVER_CONTEXT.set(ctx)
    try:
        yield
    finally:
        RESOLVER_CONTEXT.reset(token)

