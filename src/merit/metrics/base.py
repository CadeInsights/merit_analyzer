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

from merit.context import RESOLVER_CONTEXT, TEST_CONTEXT
from merit.testing.resources import Scope, resource
from merit.assertions.result import AssertionResult

P = ParamSpec("P")


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

    @validate_call
    def add_record(self, value: int | float | bool | list[int] | list[float] | list[bool] | AssertionResult) -> None:
        """
        Record one or more new data points.

        Parameters
        ----------
        value : int, float, bool, or list of these
            The value(s) to add to the metric.
        """
        with self._values_lock:
            if test_ctx := TEST_CONTEXT.get():
                if item_name := test_ctx.test_item_name:
                    self.metadata.collected_from_merits.add(item_name)
                if id_suffix := test_ctx.test_item_id_suffix:
                    self.metadata.collected_from_cases.add(id_suffix)

            if self.metadata.first_item_recorded_at is None:
                self.metadata.first_item_recorded_at = datetime.now(UTC)
            self.metadata.last_item_recorded_at = datetime.now(UTC)
            self._cache = MetricState()

            match value:
                case list():
                    self._raw_values.extend(value)
                    self._float_values.extend(float(v) for v in value)

                case AssertionResult():
                    self._raw_values.append(value.passed)
                    self._float_values.append(1.0 if value.passed else 0.0)
                    
                case _:
                    self._raw_values.append(value)
                    self._float_values.append(float(value))

    @property
    def raw_values(self) -> list[int | float | bool]:
        with self._values_lock:
            return list(self._raw_values)

    @property
    def len(self) -> int:
        with self._values_lock:
            if self._cache.len is None:
                self._cache.len = len(self._raw_values)
            return self._cache.len

    @property
    def sum(self) -> float:
        with self._values_lock:
            if self._cache.sum is None:
                self._cache.sum = math.fsum(self._float_values)
            return self._cache.sum

    @property
    def min(self) -> float:
        with self._values_lock:
            if self._cache.min is None:
                if self.len == 0:
                    warnings.warn(f"Cannot compute min for {self.name or 'unnamed metric'} - not enough values. Returning NaN.", stacklevel=2)
                    return math.nan
                self._cache.min = min(self._float_values)
            return self._cache.min

    @property
    def max(self) -> float:
        with self._values_lock:
            if self._cache.max is None:
                if self.len == 0:
                    warnings.warn(f"Cannot compute max for {self.name or 'unnamed metric'} - not enough values. Returning NaN.", stacklevel=2)
                    return math.nan
                self._cache.max = max(self._float_values)
            return self._cache.max

    @property
    def median(self) -> float:
        with self._values_lock:
            if self._cache.median is None:
                if self.len == 0:
                    warnings.warn(f"Cannot compute median for {self.name or 'unnamed metric'} - not enough values. Returning NaN.", stacklevel=2)
                    return math.nan
                self._cache.median = statistics.median(self._float_values)
            return self._cache.median

    @property
    def mean(self) -> float:
        with self._values_lock:
            if self._cache.mean is None:
                if self.len == 0:
                    warnings.warn(f"Cannot compute mean for {self.name or 'unnamed metric'} - not enough values. Returning NaN.", stacklevel=2)
                    return math.nan
                self._cache.mean = statistics.mean(self._float_values)
            return self._cache.mean

    @property
    def variance(self) -> float:
        with self._values_lock:
            if self._cache.variance is None:
                if self.len < 2:
                    warnings.warn(f"Cannot compute variance for {self.name or 'unnamed metric'} - not enough values. Returning NaN.", stacklevel=2)
                    return math.nan
                self._cache.variance = statistics.variance(self._float_values, xbar=self.mean)
            return self._cache.variance

    @property
    def std(self) -> float:
        with self._values_lock:
            if self._cache.std is None:
                if self.len < 2:
                    warnings.warn(f"Cannot compute std for {self.name or 'unnamed metric'} - not enough values. Returning NaN.", stacklevel=2)
                    return math.nan
                self._cache.std = statistics.stdev(self._float_values, xbar=self.mean)
            return self._cache.std

    @property
    def pvariance(self) -> float:
        with self._values_lock:
            if self._cache.pvariance is None:
                self._cache.pvariance = statistics.pvariance(self._float_values, mu=self.mean)
            return self._cache.pvariance

    @property
    def pstd(self) -> float:
        with self._values_lock:
            if self._cache.pstd is None:
                self._cache.pstd = statistics.pstdev(self._float_values, mu=self.mean)
            return self._cache.pstd

    @property
    def ci_90(self) -> tuple[float, float]:
        with self._values_lock:
            if self._cache.ci_90 is None:
                half = 1.645 * self.std / math.sqrt(self.len)
                self._cache.ci_90 = (self.mean - half, self.mean + half)
            return self._cache.ci_90

    @property
    def ci_95(self) -> tuple[float, float]:
        with self._values_lock:
            if self._cache.ci_95 is None:
                half = 1.96 * self.std / math.sqrt(self.len)
                self._cache.ci_95 = (self.mean - half, self.mean + half)
            return self._cache.ci_95

    @property
    def ci_99(self) -> tuple[float, float]:
        with self._values_lock:
            if self._cache.ci_99 is None:
                half = 2.576 * self.std / math.sqrt(self.len)
                self._cache.ci_99 = (self.mean - half, self.mean + half)
            return self._cache.ci_99

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
            return self._cache.percentiles

    @property
    def p25(self) -> float:
        with self._values_lock:
            return self.percentiles[24]

    @property
    def p50(self) -> float:
        return self.median

    @property
    def p75(self) -> float:
        with self._values_lock:
            return self.percentiles[74]

    @property
    def p90(self) -> float:
        with self._values_lock:
            return self.percentiles[89]

    @property
    def p95(self) -> float:
        with self._values_lock:
            return self.percentiles[94]

    @property
    def p99(self) -> float:
        with self._values_lock:
            return self.percentiles[98]

    @property
    def counter(self) -> dict[int | float | bool, int]:
        with self._values_lock:
            if self._cache.counter is None:
                self._cache.counter = dict[int | float | bool, int](
                    Counter(self._raw_values)
                )
            return self._cache.counter

    @property
    def distribution(self) -> dict[int | float | bool, float]:
        with self._values_lock:
            if self._cache.distribution is None:
                total = self.len
                counts = self.counter
                self._cache.distribution = (
                    {k: v / total for k, v in counts.items()} if total > 0 else {}
                )
            return self._cache.distribution

    @property
    def final_value(self) -> int | float | bool | list[int | float | bool] | None:
        with self._values_lock:
            return self._cache.final_value

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
            if consumer_name := resolver_ctx.consumer_name:
                metric.metadata.collected_from_resources.add(consumer_name)
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
