"""Base metric class."""

from abc import ABC, abstractmethod

from merit.predicates.base import PredicateResult


class Metric(ABC):
    """Base class for evaluation metrics.

    Metrics aggregate assertion results into a single score.
    """

    def __init_subclass__(cls, **kwargs):
        """Auto-generate name from class name if not provided."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            cls.name = cls.__name__

    @abstractmethod
    def __call__(self, results: list[PredicateResult]) -> float:
        """Compute the metric score given assertion results."""
