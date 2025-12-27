from __future__ import annotations

import logging
import threading
import math
import statistics
import datetime
from collections import Counter
from collections.abc import Callable, Generator as ABCGenerator
from dataclasses import dataclass, field, replace
from typing import Any, ParamSpec, TypeVar, get_type_hints, Generator, get_origin, get_args
from uuid import UUID, uuid4
from pydantic import validate_call, BaseModel, Field

from merit.testing.resources import Scope, resource

logger = logging.getLogger(__name__)


P = ParamSpec("P")


@dataclass(slots=True)
class MetricState:
    """Typed cache for computed metric values."""
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
    frequency_count: dict[int | float | bool, int] | None = None
    frequency_share: dict[int | float | bool, float] | None = None
    metric_value: int | float | bool | list[int | float | bool] | None = None

@dataclass(slots=True)
class Metric:
    name: str
    metadata: dict[str, str | int | float | bool | None] | None = None

    _raw_values: list[int | float | bool] = field(default_factory=list, repr=False)
    _float_values: list[float] = field(default_factory=list, repr=False)
    _values_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _cache: MetricState = field(default_factory=MetricState, repr=False)

    @validate_call
    def record_values(self, value: int | float | bool | list[int | float | bool]) -> None:
        with self._values_lock:
            self._cache = MetricState()
            if isinstance(value, list):
                self._raw_values.extend(value)
                self._float_values.extend(float(v) for v in value)
            else:
                self._raw_values.append(value)
                self._float_values.append(float(value))

    def push_to_dashboard(self):
        print(f"Pushing metric {self.name} to dashboard")

    @property
    def raw_values(self) -> list[int | float | bool]:
        with self._values_lock:
            return list[int | float | bool](self._raw_values)

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
                self._cache.min = min(self._float_values)
            return self._cache.min

    @property
    def max(self) -> float:
        with self._values_lock:
            if self._cache.max is None:
                self._cache.max = max(self._float_values)
            return self._cache.max

    @property
    def median(self) -> float:
        with self._values_lock:
            if self._cache.median is None:
                self._cache.median = statistics.median(self._float_values)
            return self._cache.median

    @property
    def mean(self) -> float:
        with self._values_lock:
            if self._cache.mean is None:
                self._cache.mean = statistics.mean(self._float_values)
            return self._cache.mean

    @property
    def variance(self) -> float:
        with self._values_lock:
            if self._cache.variance is None:
                self._cache.variance = statistics.variance(self._float_values, xbar=self.mean)
            return self._cache.variance

    @property
    def std(self) -> float:
        with self._values_lock:
            if self._cache.std is None:
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
        """Compute percentiles once (n=100) for p25-p99."""
        if self._cache.percentiles is None:
            self._cache.percentiles = statistics.quantiles(self._float_values, n=100, method="inclusive")
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
    def frequency_count(self) -> dict[int | float | bool, int]:
        with self._values_lock:
            if self._cache.frequency_count is None:
                self._cache.frequency_count = dict[int | float | bool, int](Counter[int | float | bool](self._raw_values))
            return self._cache.frequency_count

    @property
    def frequency_share(self) -> dict[int | float | bool, float]:
        with self._values_lock:
            if self._cache.frequency_share is None:
                total = self.len
                counts = self.frequency_count
                self._cache.frequency_share = {k: v / total for k, v in counts.items()} if total > 0 else {}
            return self._cache.frequency_share

    @property
    def metric_value(self) -> int | float | bool | list[int | float | bool] | None:
        with self._values_lock:
            return self._cache.metric_value

    @metric_value.setter
    def metric_value(self, value: int | float | bool | list[int | float | bool] | None) -> None:
        with self._values_lock:
            self._cache.metric_value = value

def metric(
    fn: Callable[P, Metric] | Callable[P, Generator[Metric, Any, Any]] | None = None,
    *,
    scope: Scope | str = Scope.SESSION,
) -> Any:
    """Register a callable as a metric resource.

    Args:
        fn: The callable to register.
        scope: Lifecycle scope - "case", "suite", or "session" (default: "session").
    """
    if fn is None:
        return lambda f: metric(f, scope=scope)

    return_type = get_type_hints(fn).get("return")
    # Allow Metric or Generator[Metric, Any, Any]
    valid = (
        return_type is Metric or
        (
            get_origin(return_type) in (Generator, ABCGenerator) and
            get_args(return_type)[0] is Metric
        )
    )
    if not valid:
        raise TypeError(
            f"Metric function '{fn.__name__}' must return a Metric or Generator[Metric, Any, Any]"
        )

    fn.__merit_metric__ = True

    return resource(fn, scope=scope) # type: ignore
