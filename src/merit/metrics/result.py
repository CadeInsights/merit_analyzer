"""Metric result types and collection for assertion capture."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from merit.types import Scope


@dataclass(slots=True)
class MetricResult:
    """Result of a metric assertion check during teardown."""

    name: str
    passed: bool
    error: AssertionError | None = None
    scope: Scope = Scope.SESSION


# Module-level collector for metric results
_metric_results: list[MetricResult] = []
_results_lock = threading.Lock()


def get_metric_results() -> list[MetricResult]:
    """Retrieve and clear collected metric assertion results."""
    with _results_lock:
        results = list(_metric_results)
        _metric_results.clear()
        return results


def add_metric_result(result: MetricResult) -> None:
    """Add a metric result to the collector."""
    with _results_lock:
        _metric_results.append(result)


def clear_metric_results() -> None:
    """Clear all collected metric results."""
    with _results_lock:
        _metric_results.clear()
