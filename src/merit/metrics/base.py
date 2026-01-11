from __future__ import annotations


"""
Metric abstractions for the Merit testing framework.

This module provides the core classes and decorators for recording,
computing, and managing metrics during test execution.
"""

import functools
import inspect
import math
import statistics
import threading
import warnings
from collections import Counter
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ParamSpec

from pydantic import validate_call

from merit.context import (
    METRIC_RESULTS_COLLECTOR,
    METRIC_VALUES_COLLECTOR,
    RESOLVER_CONTEXT,
    TEST_CONTEXT,
    assertions_collector,
)
from merit.testing.resources import Scope, resource


if TYPE_CHECKING:
    from merit.assertions.base import AssertionResult


P = ParamSpec("P")

CalculatedValue = (
    int
    | float
    | bool
    | list[int | float | bool]
    | tuple[int | float | bool, ...]
    | tuple[tuple[int | float | bool, int], ...]
    | tuple[tuple[int | float | bool, float], ...]
    | tuple[float, float]
    | tuple[float, float, float]
)


@dataclass(slots=True, frozen=True)
class MetricSnapshot:
    """Snapshot of a metric value captured during assertion evaluation.

    Used to record *what* metric property was accessed and the
    *value* that was observed at that moment.

    Attributes:
    ----------
    full_name : str
        Fully qualified metric property name, typically of the form
        ``"<metric_name>.<property>"`` (e.g., ``"latency_ms.p95"``).
    value : CalculatedValue
        The value observed for that property (e.g., a float for ``mean``,
        a tuple for confidence intervals, or a tuple for ``raw_values``).
    """

    full_name: str
    value: CalculatedValue


@dataclass
class MetricMetadata:
    """Metadata for a metric tracking its lifecycle and origin.

    Attributes:
    ----------
    last_item_recorded_at : datetime, optional
        Timestamp of the most recently recorded value.
    first_item_recorded_at : datetime, optional
        Timestamp of the first recorded value.
    scope : Scope
        The lifecycle scope of the metric (e.g., SESSION, SUITE, CASE).
        Defaults to Scope.SESSION.
    collected_from_merits : set of str
        Names of merit tests that contributed to this metric.
    collected_from_resources : set of str
        Names of resources that contributed to this metric.
    collected_from_cases : set of str
        Identifiers of test cases that contributed to this metric.
    """

    last_item_recorded_at: datetime | None = None
    first_item_recorded_at: datetime | None = None
    scope: Scope = field(default=Scope.SESSION)
    collected_from_merits: set[str] = field(default_factory=set)
    collected_from_resources: set[str] = field(default_factory=set)
    collected_from_cases: set[str] = field(default_factory=set)


@dataclass(slots=True)
class MetricState:
    """Typed cache for computed metric values to avoid redundant calculations.

    Attributes:
    ----------
    len : int, optional
        Number of recorded values.
    sum : float, optional
        Sum of all recorded values.
    min : float, optional
        Minimum value among recorded values.
    max : float, optional
        Maximum value among recorded values.
    median : float, optional
        Median of the recorded values.
    mean : float, optional
        Arithmetic mean of the recorded values.
    variance : float, optional
        Sample variance of the recorded values.
    std : float, optional
        Sample standard deviation of the recorded values.
    pvariance : float, optional
        Population variance of the recorded values.
    pstd : float, optional
        Population standard deviation of the recorded values.
    ci_90 : tuple of (float, float), optional
        90% confidence interval (lower, upper).
    ci_95 : tuple of (float, float), optional
        95% confidence interval (lower, upper).
    ci_99 : tuple of (float, float), optional
        99% confidence interval (lower, upper).
    percentiles : list of float, optional
        List of 99 quantiles (p1 to p99) computed with n=100.
    counter : Counter, optional
        Frequency count of each unique raw value. Missing keys return 0.
    distribution : dict, optional
        Share of each unique raw value.
    """

    len: int | None = None
    sum: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    mean: float | None = None
    variance: float | None = None
    std: float | None = None
    pvariance: float | None = None
    pstd: float | None = None
    ci_90: tuple[float, float] | None = None
    ci_95: tuple[float, float] | None = None
    ci_99: tuple[float, float] | None = None
    percentiles: list[float] | None = None
    counter: Counter[int | float | bool] | None = None
    distribution: dict[int | float | bool, float] | None = None


@dataclass(slots=True)
class Metric:
    """Thread-safe class for recording data points and computing statistical metrics.

    This class maintains a list of raw values and provides properties to compute
    various statistics (mean, std, percentiles, etc.) on demand.

    Attributes:
    ----------
    name : str, optional
        Name of the metric.
    metadata : MetricMetadata
        Metadata describing the collection context.
    """

    name: str | None = None
    metadata: MetricMetadata = field(default_factory=MetricMetadata)

    _raw_values: list[int | float | bool] = field(default_factory=list, repr=False)
    _float_values: list[float] = field(default_factory=list, repr=False)
    _values_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _cache: MetricState = field(default_factory=MetricState, repr=False)

    def _push_value_to_context(self, prop_name: str, value: Any) -> None:
        """Helper to record metric property access in assertion context."""
        collector = METRIC_VALUES_COLLECTOR.get()
        if collector is not None:
            match value:
                case list() as v:
                    value = tuple(v)

                case Counter() as v:
                    value = tuple(sorted(v.items(), key=lambda kv: kv[0]))

                case dict() as v:
                    value = tuple(sorted(v.items(), key=lambda kv: kv[0]))

            full_name = f"{self.name or 'unnamed_metric'}.{prop_name}"
            mv = MetricSnapshot(full_name=full_name, value=value)
            collector.append(mv)

    @validate_call
    def add_record(self, value: CalculatedValue) -> None:
        """Record one or more new data points.

        Parameters
        ----------
        value : int, float, bool, list of these, or tuple of these
            The value(s) to add to the metric.
        """
        with self._values_lock:
            test_ctx = TEST_CONTEXT.get()
            if test_ctx is not None:
                if test_ctx.item.name:
                    self.metadata.collected_from_merits.add(test_ctx.item.name)
                if test_ctx.item.id_suffix:
                    self.metadata.collected_from_cases.add(test_ctx.item.id_suffix)

            if self.metadata.first_item_recorded_at is None:
                self.metadata.first_item_recorded_at = datetime.now(UTC)
            self.metadata.last_item_recorded_at = datetime.now(UTC)
            self._cache = MetricState()
            match value:
                case list() as values:
                    self._raw_values.extend(values)
                    self._float_values.extend(float(v) for v in values)

                case tuple() as values:
                    items: list[int | float | bool] = []
                    for v in values:
                        if isinstance(v, (int, float, bool)):
                            items.append(v)
                        else:
                            raise TypeError(
                                "add_record only supports scalar or sequences of int|float|bool."
                            )
                    self._raw_values.extend(items)
                    self._float_values.extend(float(v) for v in items)

                case int() | float() | bool() as v:
                    self._raw_values.append(v)
                    self._float_values.append(float(v))

                case _:
                    raise TypeError(
                        "add_record only supports scalar or sequences of int|float|bool."
                    )

    @property
    def raw_values(self) -> list[int | float | bool]:
        with self._values_lock:
            value = list(self._raw_values)
            self._push_value_to_context("raw_values", value)
            return value

    @property
    def len(self) -> int:
        with self._values_lock:
            if self._cache.len is None:
                self._cache.len = len(self._raw_values)
            value = self._cache.len
            self._push_value_to_context("len", value)
            return value

    @property
    def sum(self) -> float:
        with self._values_lock:
            if self._cache.sum is None:
                self._cache.sum = math.fsum(self._float_values)
            value = self._cache.sum
            self._push_value_to_context("sum", value)
            return value

    @property
    def min(self) -> float:
        with self._values_lock:
            if self._cache.min is None:
                if self.len == 0:
                    warnings.warn(
                        f"Cannot compute min for {self.name or 'unnamed metric'} - not enough values. Returning NaN.",
                        stacklevel=2,
                    )
                    self._cache.min = math.nan
                else:
                    self._cache.min = min(self._float_values)
            value = self._cache.min
            self._push_value_to_context("min", value)
            return value

    @property
    def max(self) -> float:
        with self._values_lock:
            if self._cache.max is None:
                if self.len == 0:
                    warnings.warn(
                        f"Cannot compute max for {self.name or 'unnamed metric'} - not enough values. Returning NaN.",
                        stacklevel=2,
                    )
                    value = math.nan
                    self._cache.max = value
                else:
                    self._cache.max = max(self._float_values)
            value = self._cache.max
            self._push_value_to_context("max", value)
            return value

    @property
    def median(self) -> float:
        with self._values_lock:
            if self._cache.median is None:
                if self.len == 0:
                    warnings.warn(
                        f"Cannot compute median for {self.name or 'unnamed metric'} - not enough values. Returning NaN.",
                        stacklevel=2,
                    )
                    self._cache.median = math.nan
                else:
                    self._cache.median = statistics.median(self._float_values)
            value = self._cache.median
            self._push_value_to_context("median", value)
            return value

    @property
    def mean(self) -> float:
        with self._values_lock:
            if self._cache.mean is None:
                if self.len == 0:
                    warnings.warn(
                        f"Cannot compute mean for {self.name or 'unnamed metric'} - not enough values. Returning NaN.",
                        stacklevel=2,
                    )
                    self._cache.mean = math.nan
                else:
                    self._cache.mean = statistics.mean(self._float_values)
            value = self._cache.mean
            self._push_value_to_context("mean", value)
            return value

    @property
    def variance(self) -> float:
        with self._values_lock:
            if self._cache.variance is None:
                if self.len < 2:
                    warnings.warn(
                        f"Cannot compute variance for {self.name or 'unnamed metric'} - not enough values. Returning NaN.",
                        stacklevel=2,
                    )
                    self._cache.variance = math.nan
                else:
                    self._cache.variance = statistics.variance(self._float_values, xbar=self.mean)
            value = self._cache.variance
            self._push_value_to_context("variance", value)
            return value

    @property
    def std(self) -> float:
        with self._values_lock:
            if self._cache.std is None:
                if self.len < 2:
                    warnings.warn(
                        f"Cannot compute std for {self.name or 'unnamed metric'} - not enough values. Returning NaN.",
                        stacklevel=2,
                    )
                    self._cache.std = math.nan
                else:
                    self._cache.std = statistics.stdev(self._float_values, xbar=self.mean)
            value = self._cache.std
            self._push_value_to_context("std", value)
            return value

    @property
    def pvariance(self) -> float:
        with self._values_lock:
            if self._cache.pvariance is None:
                if self.len == 0:
                    warnings.warn(
                        f"Cannot compute pvariance for {self.name or 'unnamed metric'} - not enough values. Returning NaN.",
                        stacklevel=2,
                    )
                    self._cache.pvariance = math.nan
                else:
                    self._cache.pvariance = statistics.pvariance(self._float_values, mu=self.mean)
            value = self._cache.pvariance
            self._push_value_to_context("pvariance", value)
            return value

    @property
    def pstd(self) -> float:
        with self._values_lock:
            if self._cache.pstd is None:
                if self.len == 0:
                    warnings.warn(
                        f"Cannot compute pstd for {self.name or 'unnamed metric'} - not enough values. Returning NaN.",
                        stacklevel=2,
                    )
                    self._cache.pstd = math.nan
                else:
                    self._cache.pstd = statistics.pstdev(self._float_values, mu=self.mean)
            value = self._cache.pstd
            self._push_value_to_context("pstd", value)
            return value

    @property
    def ci_90(self) -> tuple[float, float]:
        with self._values_lock:
            if self._cache.ci_90 is None:
                if self.len == 0:
                    warnings.warn(
                        f"Cannot compute ci_90 for {self.name or 'unnamed metric'} - not enough values. Returning NaN.",
                        stacklevel=2,
                    )
                    self._cache.ci_90 = (math.nan, math.nan)
                else:
                    half = 1.645 * self.std / math.sqrt(self.len)
                    self._cache.ci_90 = (self.mean - half, self.mean + half)
            value = self._cache.ci_90
            self._push_value_to_context("ci_90", value)
            return value

    @property
    def ci_95(self) -> tuple[float, float]:
        with self._values_lock:
            if self._cache.ci_95 is None:
                if self.len == 0:
                    warnings.warn(
                        f"Cannot compute ci_95 for {self.name or 'unnamed metric'} - not enough values. Returning NaN.",
                        stacklevel=2,
                    )
                    self._cache.ci_95 = (math.nan, math.nan)
                else:
                    half = 1.96 * self.std / math.sqrt(self.len)
                    self._cache.ci_95 = (self.mean - half, self.mean + half)
            value = self._cache.ci_95
            self._push_value_to_context("ci_95", value)
            return value

    @property
    def ci_99(self) -> tuple[float, float]:
        with self._values_lock:
            if self._cache.ci_99 is None:
                if self.len == 0:
                    warnings.warn(
                        f"Cannot compute ci_99 for {self.name or 'unnamed metric'} - not enough values. Returning NaN.",
                        stacklevel=2,
                    )
                    self._cache.ci_99 = (math.nan, math.nan)
                else:
                    half = 2.576 * self.std / math.sqrt(self.len)
                    self._cache.ci_99 = (self.mean - half, self.mean + half)
            value = self._cache.ci_99
            self._push_value_to_context("ci_99", value)
            return value

    @property
    def percentiles(self) -> list[float]:
        with self._values_lock:
            if self._cache.percentiles is None:
                if self.len < 2:
                    warnings.warn(
                        f"Metric '{self.name or 'unnamed'}' has less than 2 values. Cannot compute percentiles.",
                        stacklevel=2,
                    )
                    self._cache.percentiles = [math.nan] * 99
                else:
                    self._cache.percentiles = statistics.quantiles(
                        self._float_values, n=100, method="inclusive"
                    )
            value = self._cache.percentiles
            self._push_value_to_context("percentiles", value)
            return value

    @property
    def p25(self) -> float:
        with self._values_lock:
            value = self.percentiles[24]
            self._push_value_to_context("p25", value)
            return value

    @property
    def p50(self) -> float:
        return self.median

    @property
    def p75(self) -> float:
        with self._values_lock:
            value = self.percentiles[74]
            self._push_value_to_context("p75", value)
            return value

    @property
    def p90(self) -> float:
        with self._values_lock:
            value = self.percentiles[89]
            self._push_value_to_context("p90", value)
            return value

    @property
    def p95(self) -> float:
        with self._values_lock:
            value = self.percentiles[94]
            self._push_value_to_context("p95", value)
            return value

    @property
    def p99(self) -> float:
        with self._values_lock:
            value = self.percentiles[98]
            self._push_value_to_context("p99", value)
            return value

    @property
    def counter(self) -> Counter[int | float | bool]:
        with self._values_lock:
            if self._cache.counter is None:
                self._cache.counter = Counter(self._raw_values)
            value = self._cache.counter
            self._push_value_to_context("counter", value)
            return value

    @property
    def distribution(self) -> dict[int | float | bool, float]:
        with self._values_lock:
            if self._cache.distribution is None:
                total = self.len
                counts = self.counter
                self._cache.distribution = (
                    {k: v / total for k, v in counts.items()} if total > 0 else {}
                )
            value = self._cache.distribution
            self._push_value_to_context("distribution", value)
            return value


@dataclass(slots=True)
class MetricResult:
    """Result of evaluating a `Metric` resource.

    Parameters
    ----------
    name : str
        The metric name. By default this is the decorated function name.
    metadata : MetricMetadata
        Snapshot of metadata describing where/when the metric was recorded
        (scope, contributors, timestamps).
    assertion_results : list[AssertionResult]
        Assertion results collected while the metric resource was running.
    value : CalculatedValue
        The last yielded value from the metric generator. NaN if no value was yielded.
    """

    name: str
    metadata: MetricMetadata
    assertion_results: list[AssertionResult]
    value: CalculatedValue

    def __post_init__(self) -> None:
        collector = METRIC_RESULTS_COLLECTOR.get()
        if collector is not None:
            collector.append(self)


def metric(
    fn: Callable[P, Generator[Metric | CalculatedValue, Any, Any]]
    | Callable[P, AsyncGenerator[Metric | CalculatedValue, Any]]
    | None = None,
    *,
    scope: Scope | str = Scope.SESSION,
) -> Any:
    """Register a metric resource and capture its final result.

    This decorator wraps a **generator** or **async generator** function into a
    managed resource via `merit.testing.resources.resource`. The wrapped function
    must:

    - `yield` a single `Metric` instance first (this is what gets injected and
      used during the test run).
    - Optionally `yield` a **final value** (int/float/bool/list/CI tuple) later.
      When the generator completes, a `MetricResult` is emitted containing that
      final value and any assertions evaluated while the generator was running.

    Parameters
    ----------
    fn : callable, optional
        The generator/async-generator function to register. If None, returns a
        decorator.
    scope : Scope or str, default Scope.SESSION
        The lifecycle scope of the metric resource. Can be "case", "suite",
        or "session".
    """
    if fn is None:
        return lambda f: metric(f, scope=scope)

    name = fn.__name__

    is_generator = inspect.isgeneratorfunction(fn)
    is_async_generator = inspect.isasyncgenfunction(fn)

    def on_resolve_hook(m: Metric) -> Metric:
        m.name = name
        m.metadata.scope = scope if isinstance(scope, Scope) else Scope(scope)
        return m

    def on_injection_hook(m: Metric) -> Metric:
        resolver_ctx = RESOLVER_CONTEXT.get()
        if resolver_ctx is not None:
            if resolver_ctx.consumer_name:
                m.metadata.collected_from_resources.add(resolver_ctx.consumer_name)
        return m

    if is_generator:

        @functools.wraps(fn)
        def wrapped_gen(*args: Any, **kwargs: Any) -> Generator[Metric, Any, Any]:
            assertions_results = []

            with assertions_collector(assertions_results):
                gen = fn(*args, **kwargs)
                metric_instance: Metric = next(gen)

            yield metric_instance

            with assertions_collector(assertions_results):
                final_value = None
                while True:
                    try:
                        final_value = next(gen)
                    except StopIteration:
                        if final_value is None:
                            value = math.nan
                        else:
                            value = final_value
                        MetricResult(
                            name=name,
                            metadata=replace(
                                metric_instance.metadata
                            ),  # use replace to create a copy
                            assertion_results=assertions_results,
                            value=value,
                        )
                        break

        return resource(
            wrapped_gen,
            scope=scope,
            on_resolve=on_resolve_hook,
            on_injection=on_injection_hook,
        )

    if is_async_generator:

        @functools.wraps(fn)
        async def wrapped_async_gen(*args: Any, **kwargs: Any):
            assertions_results = []

            with assertions_collector(assertions_results):
                gen = fn(*args, **kwargs)
                metric_instance: Metric = await gen.__anext__()

            yield metric_instance

            with assertions_collector(assertions_results):
                final_value = None
                while True:
                    try:
                        final_value = await gen.__anext__()
                    except StopAsyncIteration:
                        if final_value is None:
                            value = math.nan
                        else:
                            value = final_value
                        MetricResult(
                            name=name,
                            metadata=replace(
                                metric_instance.metadata
                            ),  # use replace to create a copy
                            assertion_results=assertions_results,
                            value=value,
                        )
                        break

        return resource(
            wrapped_async_gen,
            scope=scope,
            on_resolve=on_resolve_hook,
            on_injection=on_injection_hook,
        )

    msg = f"""
        {fn.__name__} is not a generator or async generator and can't be wrapped as a Merit metric. 
        To fix: yield a Metric instance and optionally yield a final calculated value.
        """
    raise ValueError(msg)
