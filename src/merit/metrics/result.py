from dataclasses import dataclass


@dataclass
class MetricResult:
    """Result of a metric."""

    metric_full_name: str
    metric_value: float