"""Repeat utilities for merit tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from merit.testing.models import RepeatModifier


class RepeatDecorator:
    """Primary entry-point for repeating tests."""

    def __call__(
        self,
        count: int,
        *,
        min_passes: int | None = None,
    ) -> Callable[[Any], Any]:
        """Repeat a test multiple times, requiring min_passes to pass.

        Args:
            count: Number of times to run the test.
            min_passes: Minimum passes required. Defaults to count (all must pass).
        """
        if count < 1:
            raise ValueError(f"repeat count must be >= 1, got {count}")

        actual_min_passes = min_passes if min_passes is not None else count

        if actual_min_passes < 1:
            raise ValueError(f"min_passes must be >= 1, got {actual_min_passes}")

        if actual_min_passes > count:
            raise ValueError(f"min_passes ({actual_min_passes}) cannot exceed count ({count})")

        modifier = RepeatModifier(count=count, min_passes=actual_min_passes)

        def decorator(target: Any) -> Any:
            modifiers: list = getattr(target, "__merit_modifiers__", [])
            modifiers.append(modifier)
            target.__merit_modifiers__ = modifiers
            return target

        return decorator


repeat = RepeatDecorator()

__all__ = ["RepeatModifier", "repeat"]
