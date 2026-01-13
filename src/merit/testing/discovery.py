"""Test discovery for merit_* files and functions."""

import importlib.util
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from merit.testing.loader import MeritModuleLoader
from merit.testing.models import Modifier, TestItem
from merit.testing.tags import TagData, get_tag_data, merge_tag_data


def _load_module(path: Path) -> ModuleType:
    """Dynamically load a Python module from path."""
    spec = importlib.util.spec_from_file_location(
        path.stem,
        path,
        loader=MeritModuleLoader(fullname=path.stem, path=path),
    )
    if spec is None or spec.loader is None:
        msg = f"Cannot load module from {path}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _extract_test_params(fn: Callable[..., Any]) -> list[str]:
    """Extract parameter names from function signature (excluding 'self')."""
    sig = inspect.signature(fn)
    return [p for p in sig.parameters if p != "self"]


def _get_modifiers(fn: Callable[..., Any]) -> list[Modifier]:
    """Extract modifiers from function's __merit_modifiers__ attribute."""
    return getattr(fn, "__merit_modifiers__", [])


def _collect_from_module(module: ModuleType, module_path: Path) -> list[TestItem]:
    """Collect all merit_* tests from a module."""
    items: list[TestItem] = []

    for name, obj in inspect.getmembers(module):
        if name.startswith("merit_") and inspect.isfunction(obj):
            items.append(_build_item_for_callable(obj, name, module_path))

        elif name.startswith("Merit") and inspect.isclass(obj):
            class_tags = get_tag_data(obj)
            for method_name, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                if method_name.startswith("merit_"):
                    items.append(
                        _build_item_for_callable(
                            method,
                            method_name,
                            module_path,
                            class_name=name,
                            parent_tags=class_tags,
                        )
                    )

    return items


def _build_item_for_callable(
    fn: Callable[..., Any],
    name: str,
    module_path: Path,
    class_name: str | None = None,
    parent_tags: TagData | None = None,
) -> TestItem:
    """Create a single TestItem for a callable with modifiers attached."""
    combined_tags = merge_tag_data(parent_tags, get_tag_data(fn))
    modifiers = reversed(_get_modifiers(fn))

    return TestItem(
        name=name,
        fn=fn,
        module_path=module_path,
        is_async=inspect.iscoroutinefunction(fn),
        params=_extract_test_params(fn),
        class_name=class_name,
        modifiers=list(modifiers),
        tags=set(combined_tags.tags),
        skip_reason=combined_tags.skip_reason,
        xfail_reason=combined_tags.xfail_reason,
        xfail_strict=combined_tags.xfail_strict,
    )


def collect(path: Path | str | None = None) -> list[TestItem]:
    """Discover all merit_* tests from path.

    Args:
        path: File or directory to search. Defaults to current directory.

    Returns:
        List of discovered TestItem objects.

    Example:
        items = collect()  # Current directory
        items = collect("merit_agents.py")  # Specific file
        items = collect("./tests/")  # Directory
    """
    if path is None:
        path = Path.cwd()
    elif isinstance(path, str):
        path = Path(path)

    path = path.resolve()
    items: list[TestItem] = []

    if path.is_file():
        if path.name.startswith("merit_") and path.suffix == ".py":
            module = _load_module(path)
            items.extend(_collect_from_module(module, path))
    elif path.is_dir():
        for file_path in path.rglob("merit_*.py"):
            module = _load_module(file_path)
            items.extend(_collect_from_module(module, file_path))

    return items
