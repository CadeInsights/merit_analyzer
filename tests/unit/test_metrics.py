import math
import statistics
from pathlib import Path

import pytest

from merit.assertions.base import AssertionResult
from merit.context import (
    ResolverContext,
    TestContext as Ctx,
    metric_results_collector,
    metric_values_collector,
    resolver_context_scope,
    test_context_scope as context_scope,
)
from merit.metrics.base import Metric, MetricMetadata, MetricResult, metric
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


def test_metric_recording():
    """Test recording single and list values."""
    m = Metric("test_metric")
    m.add_record(10)
    assert m.len == 1
    assert m.raw_values == [10]

    m.add_record([20, 30])
    assert m.len == 3
    assert m.raw_values == [10, 20, 30]


def test_metric_computations():
    """Test statistical property computations."""
    m = Metric("test_stats")
    m.add_record([10, 20, 30, 40, 50])

    assert m.sum == 150.0
    assert m.min == 10.0
    assert m.max == 50.0
    assert m.mean == 30.0
    assert m.median == 30.0
    assert m.variance == 250.0
    assert math.isclose(m.std, statistics.stdev([10, 20, 30, 40, 50]))
    assert m.pvariance == 200.0  # (400 + 100 + 0 + 100 + 400) / 5
    assert math.isclose(m.pstd, statistics.pstdev([10, 20, 30, 40, 50]))


def test_metric_percentiles():
    """Test percentile computations."""
    m = Metric("test_percentiles")
    # Need enough data for quantiles(n=100)
    data = list(range(1, 101))
    m.add_record(data)

    assert m.p50 == 50.5  # median of 1..100
    assert m.p25 == 25.75
    assert m.p75 == 75.25
    assert m.p90 == 90.1
    assert m.p95 == 95.05
    assert m.p99 == 99.01


def test_metric_counter_and_distribution():
    """Test counter and distribution properties."""
    m = Metric("test_dist")
    m.add_record(list([True, True, False, 10, 10, 10]))

    assert m.counter == {True: 2, False: 1, 10: 3}
    assert m.counter[999] == 0
    assert m.distribution == {True: 2 / 6, False: 1 / 6, 10: 3 / 6}


def test_metric_confidence_intervals():
    """Test confidence interval computations."""
    m = Metric("test_ci")
    # Data with mean=30, std=15.811388, n=5
    # CI 95% = 30 +/- 1.96 * 15.811388 / sqrt(5) = 30 +/- 1.96 * 7.071 = 30 +/- 13.859 = (16.141, 43.859)
    m.add_record(list([10, 20, 30, 40, 50]))

    ci90 = m.ci_90
    ci95 = m.ci_95
    ci99 = m.ci_99

    assert ci90[0] < m.mean < ci90[1]
    assert ci95[0] < m.mean < ci95[1]
    assert ci99[0] < m.mean < ci99[1]

    # ci99 should be wider than ci95, which should be wider than ci90
    assert (ci99[1] - ci99[0]) > (ci95[1] - ci95[0]) > (ci90[1] - ci90[0])


def test_metric_timestamps():
    """Test recording timestamps."""
    m = Metric("test_time")
    assert m.metadata.first_item_recorded_at is None
    assert m.metadata.last_item_recorded_at is None

    m.add_record(10)
    t1 = m.metadata.first_item_recorded_at
    t2 = m.metadata.last_item_recorded_at
    assert t1 is not None
    assert t2 >= t1

    import time

    time.sleep(0.01)
    m.add_record(20)
    assert m.metadata.first_item_recorded_at == t1
    assert m.metadata.last_item_recorded_at is not None
    assert t2 is not None
    assert m.metadata.last_item_recorded_at > t2


def test_metric_empty_edge_cases_do_not_crash():
    m = Metric("empty")
    assert math.isnan(m.pvariance)
    assert math.isnan(m.pstd)

    ci90 = m.ci_90
    ci95 = m.ci_95
    ci99 = m.ci_99
    assert math.isnan(ci90[0]) and math.isnan(ci90[1])
    assert math.isnan(ci95[0]) and math.isnan(ci95[1])
    assert math.isnan(ci99[0]) and math.isnan(ci99[1])


def test_metric_percentiles_with_single_value_returns_nans():
    m = Metric("p")
    m.add_record(1)
    with pytest.warns(UserWarning, match="less than 2 values"):
        percentiles = m.percentiles
    assert len(percentiles) == 99
    assert all(math.isnan(x) for x in percentiles)


def test_metric_values_collector_records_property_access_with_normalized_values():
    values = []
    m = Metric("m")
    m.add_record([1, 2, 2])

    with metric_values_collector(values):
        _ = m.raw_values
        _ = m.counter
        _ = m.distribution

    by_name = {}
    for mv in values:
        by_name[mv.full_name] = mv.value

    assert by_name["m.raw_values"] == (1, 2, 2)
    assert by_name["m.counter"] == ((1, 1), (2, 2))
    assert by_name["m.distribution"][0] == (1, pytest.approx(1 / 3))
    assert by_name["m.distribution"][1] == (2, pytest.approx(2 / 3))


def test_metric_values_collector_uses_default_name_when_metric_is_unnamed():
    values = []
    m = Metric()
    with metric_values_collector(values):
        assert m.len == 0

    assert values[-1].full_name == "unnamed_metric.len"
    assert values[-1].value == 0


def test_metric_result_is_collected_when_collector_is_active():
    results = []
    with metric_results_collector(results):
        MetricResult(
            name="x",
            metadata=MetricMetadata(),
            assertion_results=[],
            value=1,
        )
    assert len(results) == 1
    assert results[0].name == "x"


@pytest.mark.asyncio
async def test_metric_decorator_no_args():
    """Test @metric decorator without explicit arguments."""
    clear_registry()

    @metric
    def default_metric():
        yield Metric("default")
        return 0

    resolver = ResourceResolver()
    m = await resolver.resolve("default_metric")
    assert m.name == "default_metric"


@pytest.mark.asyncio
async def test_metric_on_injection_hook_with_context():
    """Merit/case attribution happens in add_record; resource attribution happens on injection."""
    clear_registry()

    @metric(scope=Scope.CASE)
    def test_ctx_metric():
        yield Metric("ctx")
        return 0

    resolver = ResourceResolver()
    ctx = Ctx(item=_make_item("my_merit"))
    with context_scope(ctx):
        with resolver_context_scope(ResolverContext(consumer_name="some_resource")):
            m = await resolver.resolve("test_ctx_metric")
            # injection hook attribution
            assert "some_resource" in m.metadata.collected_from_resources
            # test data attribution is delegated to add_record
            assert "my_merit" not in m.metadata.collected_from_merits
            m.add_record(1)

    assert "my_merit" in m.metadata.collected_from_merits
    assert m.metadata.scope == Scope.CASE


@pytest.mark.asyncio
async def test_metric_on_injection_cumulative_metadata():
    """Merit attribution accumulates via add_record across multiple contexts."""
    clear_registry()

    @metric(scope=Scope.SESSION)
    def shared_metric():
        yield Metric("shared")
        return 0

    resolver = ResourceResolver()

    # First resolution with context A
    with context_scope(Ctx(item=_make_item("merit_a"))):
        m1 = await resolver.resolve("shared_metric")
        m1.add_record(1)
    assert "merit_a" in m1.metadata.collected_from_merits

    # Second resolution with context B
    with context_scope(Ctx(item=_make_item("merit_b"))):
        m2 = await resolver.resolve("shared_metric")
        m2.add_record(2)

    # Verify both are the same instance and contain accumulated metadata
    assert m1 is m2
    assert "merit_a" in m2.metadata.collected_from_merits
    assert "merit_b" in m2.metadata.collected_from_merits


@pytest.mark.asyncio
async def test_metric_decorator_emits_metric_result_on_teardown_with_assertions_and_return_value():
    clear_registry()

    @metric(scope=Scope.CASE)
    def scored_metric():
        AssertionResult(expression_repr="before", passed=True)
        yield Metric("ignored")
        AssertionResult(expression_repr="after", passed=False)
        yield 123
        return 999  # ignored: metric final value comes from the second yield

    resolver = ResourceResolver()
    metric_results = []
    with metric_results_collector(metric_results):
        m = await resolver.resolve("scored_metric")
        assert m.name == "scored_metric"
        await resolver.teardown()

    assert len(metric_results) == 1
    r = metric_results[0]
    assert r.name == "scored_metric"
    assert r.value == 123
    assert [a.expression_repr for a in r.assertion_results] == ["before", "after"]
    assert r.metadata.scope == Scope.CASE
    assert r.metadata is not m.metadata


@pytest.mark.asyncio
async def test_metric_decorator_emits_nan_when_generator_returns_value_but_does_not_yield_final_value():
    clear_registry()

    @metric(scope=Scope.CASE)
    def return_only_metric():
        yield Metric("ignored")
        return 123  # ignored: no second yield -> NaN

    resolver = ResourceResolver()
    metric_results = []
    with metric_results_collector(metric_results):
        await resolver.resolve("return_only_metric")
        await resolver.teardown()

    assert len(metric_results) == 1
    assert math.isnan(metric_results[0].value)


@pytest.mark.asyncio
async def test_metric_decorator_uses_second_yield_value_and_ignores_return_value():
    clear_registry()

    @metric(scope=Scope.CASE)
    def yield_and_return_metric():
        yield Metric("ignored")
        yield 111
        return 222  # ignored

    resolver = ResourceResolver()
    metric_results = []
    with metric_results_collector(metric_results):
        await resolver.resolve("yield_and_return_metric")
        await resolver.teardown()

    assert len(metric_results) == 1
    assert metric_results[0].name == "yield_and_return_metric"
    assert metric_results[0].value == 111
