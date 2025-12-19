"""Basic metric implementations."""

from merit.predicates.base import PredicateResult
from merit.metrics._base import Metric


class PassRate(Metric):
    """Calculate the percentage of passed assertions."""

    name = "PassRate"

    def __call__(self, results: list[PredicateResult]) -> float:
        """Calculate pass rate."""
        if not results:
            return 0.0

        passed = sum(1 for r in results if r.value)
        return passed / len(results)


class AverageScore(Metric):
    """Calculate the average score across all assertions."""

    name = "AverageScore"

    def __call__(self, results: list[PredicateResult]) -> float:
        """Calculate average score."""
        if not results:
            return 0.0

        scores = [r.confidence for r in results if r.confidence is not None]
        if not scores:
            return 0.0

        return sum(scores) / len(scores)
