from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from merit.assertions.base import AssertionRepr, AssertionResult
from merit.context import (
    ResolverContext,
    TestContext as Ctx,
    assertions_collector,
    metrics,
    predicate_results_collector,
    resolver_context_scope,
    test_context_scope as context_scope,
)
from merit.metrics_.base import Metric, metric
from merit.predicates.base import PredicateResult
from merit.resources import ResourceResolver, Scope, clear_registry
from merit.testing.discovery import TestItem


def _make_item(name: str = "merit_fn", id_suffix: str | None = None) -> TestItem:
    """Create a minimal TestItem for testing."""
    return TestItem(
        name=name,
        fn=lambda: None,
        module_path=Path("test.py"),
        is_async=False,
        id_suffix=id_suffix,
    )


@pytest.fixture(autouse=True)
def clean_registry():
    """Avoid cross-test leakage of globally-registered metric resources."""
    clear_registry()
    yield
    clear_registry()


def test_assertionresult_appends_to_test_context():
    assertion_results: list[AssertionResult] = []

    with assertions_collector(assertion_results):
        ar = AssertionResult(
            passed=True,
            expression_repr=AssertionRepr(
                expr="x == y",
                lines_above="",
                lines_below="",
                resolved_args={},
            ),
        )

    assert assertion_results == [ar]

    # Outside the scope, assertion results should not be auto-attached.
    ar2 = AssertionResult(
        passed=True,
        expression_repr=AssertionRepr(
            expr="a == b",
            lines_above="",
            lines_below="",
            resolved_args={},
        ),
    )
    assert assertion_results == [ar]
    assert ar2 not in assertion_results


def test_assertion_context_collects_predicate_results():
    test_ctx = Ctx(item=_make_item("merit_name"))

    # Collect predicate results into a list
    predicate_results_list = []

    with (
        context_scope(test_ctx),
        predicate_results_collector(predicate_results_list),
    ):
        # PredicateResult should attach itself to the collector
        pr = PredicateResult(
            actual="a",
            reference="b",
            name="manual_predicate",
            strict=True,
            value=True,
        )
        assert pr.name == "manual_predicate"

    # Build AssertionResult with collected data
    ar = AssertionResult(
        passed=True,
        expression_repr=AssertionRepr(
            expr="check",
            lines_above="",
            lines_below="",
            resolved_args={},
        ),
        predicate_results=predicate_results_list,
    )

    assert len(ar.predicate_results) == 1
    assert ar.predicate_results[0] == pr


def test_metrics_records_assertion_passed_and_reads_test_context_for_metadata():
    test_ctx = Ctx(item=_make_item("my_merit"))

    m1 = Metric(name="m1")
    m2 = Metric(name="m2")

    with context_scope(test_ctx), metrics([m1, m2]):
        # AssertionResult.__post_init__ calls metric.add_record(self.passed)
        ar1 = AssertionResult(
            passed=True,
            expression_repr=AssertionRepr(
                expr="first",
                lines_above="",
                lines_below="",
                resolved_args={},
            ),
        )
        ar2 = AssertionResult(
            passed=False,
            expression_repr=AssertionRepr(
                expr="second",
                lines_above="",
                lines_below="",
                resolved_args={},
            ),
        )

    assert m1.raw_values == [True, False]
    assert m2.raw_values == [True, False]

    # add_record is called from AssertionResult.__post_init__, so attribution should be captured.
    assert "my_merit" in m1.metadata.collected_from_merits


@pytest.mark.asyncio
async def test_metric_injection_reads_resolver_context():
    @metric(scope=Scope.CASE)
    def injected_metric() -> Generator[Metric, Any, Any]:
        yield Metric(name="ignored_by_on_resolve")

    resolver = ResourceResolver()
    with resolver_context_scope(ResolverContext(consumer_name="consumer_a")):
        m = await resolver.resolve("injected_metric")

    assert "consumer_a" in m.metadata.collected_from_resources
