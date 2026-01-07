from __future__ import annotations

"""
Metric abstractions for the Merit analyzer.

This module provides the core classes and decorators for recording,
computing, and managing metrics during test execution.
"""

import threading
import math
import statistics
import warnings
from datetime import UTC, datetime
from collections import Counter
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any, ParamSpec
from pydantic import validate_call

from merit.context import RESOLVER_CONTEXT, TEST_CONTEXT, ASSERTION_CONTEXT
from merit.testing.resources import Scope, resource


P = ParamSpec("P")


@dataclass(slots=True, frozen=True)
class MetricValue:
    """
    Snapshot of a metric value captured during assertion evaluation.

    Used to record *what* metric property was accessed and the
    *value* that was observed at that moment.

    Attributes
    ----------
    metric_full_name : str
        Fully qualified metric property name, typically of the form
        ``"<metric_name>.<property>"`` (e.g., ``"latency_ms.p95"``).
    metric_value : int | float | bool | list[int | float | bool]
        The value observed for that property (e.g., a float for ``mean``,
        a tuple for confidence intervals, or a list for ``raw_values``).
    """

    metric_full_name: str
    metric_value: int | float | bool | list[int | float | bool] | tuple[float, float] | tuple[float, float, float]


@dataclass
class MetricMetadata:
    """
    Metadata for a metric tracking its lifecycle and origin.

    Attributes
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
    """
    Typed cache for computed metric values to avoid redundant calculations.

    Attributes
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
    counter : dict, optional
        Frequency count of each unique raw value.
    distribution : dict, optional
        Share of each unique raw value.
    final_value : int, float, bool, or list, optional
        A single value or list representing the final result of the metric
        after all records are processed.
    """

    # auto-computed values
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
    counter: dict[int | float | bool, int] | None = None
    distribution: dict[int | float | bool, float] | None = None
    
    # final value of the metric
    final_value: int | float | bool | list[int | float | bool] | None = None


@dataclass(slots=True)
class Metric:
    """
    Thread-safe class for recording data points and computing statistical metrics.

    This class maintains a list of raw values and provides properties to compute
    various statistics (mean, std, percentiles, etc.) on demand.

    Attributes
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
        if as_ctx := ASSERTION_CONTEXT.get():
            full_name = f"{self.name or 'unnamed_metric'}.{prop_name}"
            mv = MetricValue(metric_full_name=full_name, metric_value=value)
            as_ctx.metric_values.add(mv)

    @validate_call
    def add_record(self, value: int | float | bool | list[int] | list[float] | list[bool]) -> None:
        """
        Record one or more new data points.

        Parameters
        ----------
        value : int, float, bool, or list of these
            The value(s) to add to the metric.
        """
        with self._values_lock:
            if test_ctx := TEST_CONTEXT.get():
                if test_ctx.test_item_name:
                    self.metadata.collected_from_merits.add(test_ctx.test_item_name)
                if test_ctx.test_item_id_suffix:
                    self.metadata.collected_from_cases.add(test_ctx.test_item_id_suffix)

            if self.metadata.first_item_recorded_at is None:
                self.metadata.first_item_recorded_at = datetime.now(UTC)
            self.metadata.last_item_recorded_at = datetime.now(UTC)
            self._cache = MetricState()
            if isinstance(value, list):
                self._raw_values.extend(value)
                self._float_values.extend(float(v) for v in value)
            else:
                self._raw_values.append(value)
                self._float_values.append(float(value))

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
                    warnings.warn(f"Cannot compute min for {self.name or 'unnamed metric'} - not enough values. Returning NaN.", stacklevel=2)
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
                    warnings.warn(f"Cannot compute max for {self.name or 'unnamed metric'} - not enough values. Returning NaN.", stacklevel=2)
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
                    warnings.warn(f"Cannot compute median for {self.name or 'unnamed metric'} - not enough values. Returning NaN.", stacklevel=2)
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
                    warnings.warn(f"Cannot compute mean for {self.name or 'unnamed metric'} - not enough values. Returning NaN.", stacklevel=2)
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
                    warnings.warn(f"Cannot compute variance for {self.name or 'unnamed metric'} - not enough values. Returning NaN.", stacklevel=2)
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
                    warnings.warn(f"Cannot compute std for {self.name or 'unnamed metric'} - not enough values. Returning NaN.", stacklevel=2)
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
                self._cache.pvariance = statistics.pvariance(self._float_values, mu=self.mean)
            value = self._cache.pvariance
            self._push_value_to_context("pvariance", value)
            return value

    @property
    def pstd(self) -> float:
        with self._values_lock:
            if self._cache.pstd is None:
                self._cache.pstd = statistics.pstdev(self._float_values, mu=self.mean)
            value = self._cache.pstd
            self._push_value_to_context("pstd", value)
            return value

    @property
    def ci_90(self) -> tuple[float, float]:
        with self._values_lock:
            if self._cache.ci_90 is None:
                half = 1.645 * self.std / math.sqrt(self.len)
                self._cache.ci_90 = (self.mean - half, self.mean + half)
            value = self._cache.ci_90
            self._push_value_to_context("ci_90", value)
            return value

    @property
    def ci_95(self) -> tuple[float, float]:
        with self._values_lock:
            if self._cache.ci_95 is None:
                half = 1.96 * self.std / math.sqrt(self.len)
                self._cache.ci_95 = (self.mean - half, self.mean + half)
            value = self._cache.ci_95
            self._push_value_to_context("ci_95", value)
            return value

    @property
    def ci_99(self) -> tuple[float, float]:
        with self._values_lock:
            if self._cache.ci_99 is None:
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
                    warnings.warn(f"Metric '{self.name or 'unnamed'}' has less than 2 values. Cannot compute percentiles.", stacklevel=2)
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
    def counter(self) -> dict[int | float | bool, int]:
        with self._values_lock:
            if self._cache.counter is None:
                self._cache.counter = dict[int | float | bool, int](
                    Counter(self._raw_values)
                )
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

    @property
    def final_value(self) -> int | float | bool | list[int | float | bool] | None:
        with self._values_lock:
            value = self._cache.final_value
            self._push_value_to_context("final_value", value)
            return value

    @final_value.setter
    def final_value(self, value: int | float | bool | list[int | float | bool] | None) -> None:
        with self._values_lock:
            self._cache.final_value = value


def metric(
    fn: Callable[P, Metric] | Callable[P, Generator[Metric, Any, Any]] | None = None,
    *,
    scope: Scope | str = Scope.SESSION,
) -> Any:
    """
    Register a metric.

    This decorator converts a function that returns a `Metric` (or a generator
    yielding one) into a managed resource within the Merit framework. It
    automatically handles metadata collection like which tests or resources
    contributed to the metric.

    Parameters
    ----------
    fn : callable, optional
        The function or generator to register. If None, returns a decorator.
    scope : Scope or str, default Scope.SESSION
        The lifecycle scope of the metric resource. Can be "case", "suite",
        or "session".

    Returns
    -------
    callable
        The wrapped resource or a decorator if `fn` is None.
    """
    if fn is None:
        return lambda f: metric(f, scope=scope)

    name = fn.__name__

    def on_resolve_hook(metric: Metric) -> Metric:
        if not isinstance(metric, Metric):
            raise ValueError(f"Metric {metric} is not a valid Metric instance.")

        metric.name = name
        metric.metadata.scope = scope if isinstance(scope, Scope) else Scope(scope)
        return metric

    def on_injection_hook(metric: Metric) -> Metric:
        if resolver_ctx := RESOLVER_CONTEXT.get():
            if resolver_ctx.consumer_name:
                metric.metadata.collected_from_resources.add(resolver_ctx.consumer_name)
        return metric

    def on_teardown_hook(metric: Metric) -> None:
        # TODO: implement pushing data to dashboard
        pass

    return resource(
        fn,
        scope=scope,
        on_resolve=on_resolve_hook,
        on_injection=on_injection_hook,
        on_teardown=on_teardown_hook,
    )  # type: ignore
