from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator


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
    """

    test_item_name: str | None = None
    test_item_group_name: str | None = None
    test_item_module_path: str | None = None
    test_item_tags: list[str] = field(default_factory=list)
    test_item_params: list[str] = field(default_factory=list)
    test_item_id_suffix: str | None = None


@dataclass(frozen=True, slots=True)
class ResolverContext:
    """Context for resource resolution.

    Attributes
    ----------
    consumer_name
        Name/identifier of the component currently resolving/consuming a resource.
    """

    consumer_name: str | None = None


TEST_CONTEXT: ContextVar[TestContext] = ContextVar("test_context", default=TestContext())
RESOLVER_CONTEXT: ContextVar[ResolverContext] = ContextVar("resolver_context", default=ResolverContext())


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

