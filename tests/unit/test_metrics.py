import pytest
import math
import statistics
from pathlib import Path
from merit.metrics.base import Metric, metric
from merit.testing.discovery import TestItem
from merit.testing.resources import Scope, ResourceResolver, clear_registry
from merit.context import (
    ResolverContext,
    TestContext as Ctx,
    resolver_context_scope,
    test_context_scope as context_scope,
)


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
    m = Metric(name="test_metric")
    m.add_record(10)
    assert m.len == 1
    assert m.raw_values == [10]

    m.add_record(([20, 30]))
    assert m.len == 3
    assert m.raw_values == [10, 20, 30]


def test_metric_computations():
    """Test statistical property computations."""
    m = Metric(name="test_stats")
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
    m = Metric(name="test_percentiles")
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
    m = Metric(name="test_dist")
    m.add_record(list([True, True, False, 10, 10, 10]))

    assert m.counter == {True: 2, False: 1, 10: 3}
    assert m.counter[999] == 0
    assert m.distribution == {True: 2 / 6, False: 1 / 6, 10: 3 / 6}


def test_metric_confidence_intervals():
    """Test confidence interval computations."""
    m = Metric(name="test_ci")
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
    m = Metric(name="test_time")
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
    m = Metric(name="empty")
    assert math.isnan(m.pvariance)
    assert math.isnan(m.pstd)

    ci90 = m.ci_90
    ci95 = m.ci_95
    ci99 = m.ci_99
    assert math.isnan(ci90[0]) and math.isnan(ci90[1])
    assert math.isnan(ci95[0]) and math.isnan(ci95[1])
    assert math.isnan(ci99[0]) and math.isnan(ci99[1])


@pytest.mark.asyncio
async def test_metric_decorator_no_args():
    """Test @metric decorator without explicit arguments."""
    clear_registry()

    @metric
    def default_metric() -> Metric:
        return Metric(name="default")

    resolver = ResourceResolver()
    m = await resolver.resolve("default_metric")
    assert m.name == "default_metric"


@pytest.mark.asyncio
async def test_metric_on_injection_hook_with_context():
    """Merit/case attribution happens in add_record; resource attribution happens on injection."""
    clear_registry()

    @metric(scope=Scope.CASE)
    def test_ctx_metric() -> Metric:
        return Metric(name="ctx")

    resolver = ResourceResolver()
    ctx = Ctx(item=_make_item("my_merit", id_suffix="case_123"))
    with context_scope(ctx):
        with resolver_context_scope(ResolverContext(consumer_name="some_resource")):
            m = await resolver.resolve("test_ctx_metric")
            # injection hook attribution
            assert "some_resource" in m.metadata.collected_from_resources
            # test data attribution is delegated to add_record
            assert "my_merit" not in m.metadata.collected_from_merits
            assert "case_123" not in m.metadata.collected_from_cases
            m.add_record(1)
    
    assert "my_merit" in m.metadata.collected_from_merits
    assert "case_123" in m.metadata.collected_from_cases
    assert m.metadata.scope == Scope.CASE


@pytest.mark.asyncio
async def test_metric_on_injection_cumulative_metadata():
    """Merit attribution accumulates via add_record across multiple contexts."""
    clear_registry()

    @metric(scope=Scope.SESSION)
    def shared_metric() -> Metric:
        return Metric(name="shared")

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
