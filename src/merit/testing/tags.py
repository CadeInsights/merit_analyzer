"""Tagging utilities for merit tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TagData:
    """Tag metadata attached to callables or classes."""

    tags: set[str] = field(default_factory=set)
    skip_reason: str | None = None
    xfail_reason: str | None = None
    xfail_strict: bool = False


@dataclass
class RepeatData:
    """Configuration for repeating a test multiple times."""

    count: int
    min_passes: int


def get_repeat_data(target: Any) -> RepeatData | None:
    """Return repeat configuration for the target, if any."""
    return getattr(target, "__merit_repeat__", None)


def _ensure_tag_data(target: Any) -> TagData:
    data: TagData | None = getattr(target, "__merit_tag_data__", None)
    if data is None:
        data = TagData()
        target.__merit_tag_data__ = data
    return data


def _copy_tag_data(data: TagData | None) -> TagData:
    if data is None:
        return TagData()
    return TagData(
        tags=set(data.tags),
        skip_reason=data.skip_reason,
        xfail_reason=data.xfail_reason,
        xfail_strict=data.xfail_strict,
    )


def get_tag_data(target: Any) -> TagData:
    """Return a copy of tag metadata for the target."""
    return _copy_tag_data(getattr(target, "__merit_tag_data__", None))


def merge_tag_data(*datas: TagData | None) -> TagData:
    """Merge tag metadata, later entries overriding earlier ones."""
    merged = TagData()
    for data in datas:
        if not data:
            continue
        merged.tags.update(data.tags)
        if data.skip_reason is not None:
            merged.skip_reason = data.skip_reason
        if data.xfail_reason is not None:
            merged.xfail_reason = data.xfail_reason
            merged.xfail_strict = data.xfail_strict
    return merged


class TagDecorator:
    """Primary entry-point for tagging tests."""

    def __call__(self, *names: str) -> Callable[[Any], Any]:
        def decorator(target: Any) -> Any:
            data = _ensure_tag_data(target)
            for name in names:
                if not name:
                    continue
                data.tags.add(str(name))
            return target

        return decorator

    def skip(self, *, reason: str | None = None) -> Callable[[Any], Any]:
        def decorator(target: Any) -> Any:
            data = _ensure_tag_data(target)
            data.skip_reason = reason or "skipped via tag"
            data.tags.add("skip")
            return target

        return decorator

    def xfail(
        self,
        *,
        reason: str | None = None,
        strict: bool = False,
    ) -> Callable[[Any], Any]:
        def decorator(target: Any) -> Any:
            data = _ensure_tag_data(target)
            data.xfail_reason = reason or "expected failure"
            data.xfail_strict = strict
            data.tags.add("xfail")
            return target

        return decorator

    def repeat(
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
        
        def decorator(target: Any) -> Any:
            target.__merit_repeat__ = RepeatData(
                count=count,
                min_passes=actual_min_passes,
            )
            data = _ensure_tag_data(target)
            data.tags.add("repeat")
            return target

        return decorator


tag = TagDecorator()

__all__ = ["RepeatData", "TagData", "get_repeat_data", "get_tag_data", "merge_tag_data", "tag"]
