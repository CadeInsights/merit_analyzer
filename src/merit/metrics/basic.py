"""Basic metric implementations."""

from merit.assertions._base import AssertionResult
from merit.metrics._base import Metric


class PassRate(Metric):
    """Calculate the percentage of passed assertions."""

    name = "PassRate"

    def __call__(self, results: list[AssertionResult]) -> float:
        """Calculate pass rate."""
        if not results:
            return 0.0

        passed = sum(1 for r in results if r.passed)
        return passed / len(results)


class AverageScore(Metric):
    """Calculate the average score across all assertions."""

    name = "AverageScore"

    def __call__(self, results: list[AssertionResult]) -> float:
        """Calculate average score."""
        if not results:
            return 0.0

        scores = [r.score for r in results if r.score is not None]
        if not scores:
            return 0.0

        return sum(scores) / len(scores)
