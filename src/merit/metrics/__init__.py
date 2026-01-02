"""Metrics module for aggregating predicate results."""

from .base import Metric, metric
from .result import MetricResult, get_metric_results

__all__ = ["Metric", "metric", "MetricResult", "get_metric_results"]
