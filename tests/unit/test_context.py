import pytest

from pathlib import Path
from uuid import UUID, uuid4

from merit.assertions.base import AssertionResult
from merit.context import (
    ASSERTION_CONTEXT,
    METRIC_CONTEXT,
    RESOLVER_CONTEXT,
    TEST_CONTEXT,
    ResolverContext,
    TestContext as Ctx,
    assertion_context_scope,
    metrics,
    resolver_context_scope,
    test_context_scope as context_scope,
)
from merit.metrics.base import Metric, metric
from merit.predicates.base import PredicateMetadata, PredicateResult
from merit.testing.discovery import TestItem
from merit.testing.resources import ResourceResolver, Scope, clear_registry


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
    ctx = Ctx(item=_make_item("merit_fn"))

    with context_scope(ctx):
        ar = AssertionResult(_passed=True, expression_repr="x == y")

    assert ctx.assertion_results == [ar]

    # Outside the scope, assertion results should not be auto-attached.
    ar2 = AssertionResult(_passed=True, expression_repr="a == b")
    assert ctx.assertion_results == [ar]
    assert ar2 not in ctx.assertion_results


def test_assertion_context_collects_predicate_results_and_metric_values():
    case_uuid = uuid4()
    test_ctx = Ctx(item=_make_item("merit_name", id_suffix=str(case_uuid)))
    ar = AssertionResult(_passed=True, expression_repr="check")

    m = Metric(name="m")
    m.add_record([1, 2, 3])

    with context_scope(test_ctx):
        with assertion_context_scope(ar):
            # PredicateResult should attach itself to ASSERTION_CONTEXT.
            pr = PredicateResult(
                predicate_metadata=PredicateMetadata(actual="a", reference="b", strict=True),
                value=True,
            )

            assert ar.predicate_results == [pr]
            assert pr.case_id == UUID(str(case_uuid))
            assert pr.predicate_metadata.merit_name == "merit_name"

            # Metric property access should push MetricValue into ASSERTION_CONTEXT.
            assert m.len == 3
            assert m.min == 1

    names = {mv.metric_full_name for mv in ar.metric_values}
    assert "m.len" in names
    assert "m.min" in names


def test_metrics_records_assertion_passed_and_reads_test_context_for_metadata():
    case_uuid = uuid4()
    test_ctx = Ctx(item=_make_item("my_merit", id_suffix=str(case_uuid)))

    m1 = Metric(name="m1")
    m2 = Metric(name="m2")
    ar = AssertionResult(_passed=False, expression_repr="initial")

    with context_scope(test_ctx):
        with metrics([m1, m2]):
            ar.passed = True
            ar.passed = False

    assert m1.raw_values == [True, False]
    assert m2.raw_values == [True, False]

    # add_record is called from AssertionResult.passed, so attribution should be captured.
    assert "my_merit" in m1.metadata.collected_from_merits
    assert str(case_uuid) in m1.metadata.collected_from_cases


@pytest.mark.asyncio
async def test_metric_injection_reads_resolver_context():
    @metric(scope=Scope.CASE)
    def injected_metric() -> Metric:
        return Metric(name="ignored_by_on_resolve")

    resolver = ResourceResolver()
    with resolver_context_scope(ResolverContext(consumer_name="consumer_a")):
        m = await resolver.resolve("injected_metric")

    assert "consumer_a" in m.metadata.collected_from_resources

