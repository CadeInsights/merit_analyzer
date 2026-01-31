"""Reporter registry for plugin-style reporter registration."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, TypeVar


if TYPE_CHECKING:
    from merit.reports.base import Reporter


T = TypeVar("T", bound="Reporter")

_reporter_registry: dict[str, type[Reporter]] = {}
_builtin_registry: dict[str, type[Reporter]] = {}


def reporter(
    cls: type[T] | None = None,
    *,
    enabled: bool = True,
    name: str | None = None,
) -> type[T] | Any:
    """Register a Reporter class for discovery.

    Can be used as a decorator with or without arguments:

        @reporter
        class MyReporter(Reporter): ...

        @reporter(name="custom")
        class MyReporter(Reporter): ...

    Args:
        cls: The Reporter class to register.
        enabled: Whether to register this reporter (default True).
        name: Custom name for registry lookup (defaults to class name).
    """

    def decorator(cls: type[T]) -> type[T]:
        if enabled:
            registry_name = name or cls.__name__
            _reporter_registry[registry_name] = cls
        return cls

    if cls is not None:
        return decorator(cls)
    return decorator


def get_reporter_registry() -> dict[str, type[Reporter]]:
    """Get the global reporter registry."""
    return _reporter_registry


def clear_reporter_registry() -> None:
    """Clear all registered reporters, keeping built-ins."""
    _reporter_registry.clear()
    _reporter_registry.update(_builtin_registry)


def _import_reporter_class(import_path: str) -> type[Reporter]:
    """Import a Reporter class from an import string.

    Supports formats:
        - "module.path:ClassName"
        - "module.path.ClassName"
    """
    if ":" in import_path:
        module_path, class_name = import_path.rsplit(":", 1)
    elif "." in import_path:
        module_path, class_name = import_path.rsplit(".", 1)
    else:
        msg = f"Invalid import path: {import_path}"
        raise ValueError(msg)

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    from merit.reports.base import Reporter

    if not isinstance(cls, type) or not issubclass(cls, Reporter):
        msg = f"{import_path} is not a Reporter subclass"
        raise TypeError(msg)

    return cls


def resolve_reporter(name: str, **kwargs: Any) -> Reporter:
    """Resolve a reporter by registry name or import string.

    Args:
        name: Registry name (e.g., "ConsoleReporter") or import string
              (e.g., "myapp.reports:CustomReporter").
        **kwargs: Arguments passed to the reporter constructor.

    Returns:
        Instantiated Reporter.

    Raises:
        ValueError: If reporter cannot be resolved.
    """
    if name in _reporter_registry:
        return _reporter_registry[name](**kwargs)

    if ":" in name or "." in name:
        cls = _import_reporter_class(name)
        return cls(**kwargs)

    available = ", ".join(sorted(_reporter_registry.keys()))
    msg = f"Unknown reporter: {name}. Available: {available}"
    raise ValueError(msg)


def resolve_reporters(
    names: list[str],
    options: dict[str, dict[str, Any]] | None = None,
) -> list[Reporter]:
    """Resolve multiple reporters by name with optional per-reporter options.

    Args:
        names: List of reporter names or import strings.
        options: Dict mapping reporter names to constructor kwargs.

    Returns:
        List of instantiated Reporters.
    """
    options = options or {}
    reporters = []
    for name in names:
        kwargs = options.get(name, {})
        reporters.append(resolve_reporter(name, **kwargs))
    return reporters


def register_builtin(cls: type[T]) -> type[T]:
    """Register a built-in reporter (persists through clear)."""
    _reporter_registry[cls.__name__] = cls
    _builtin_registry[cls.__name__] = cls
    return cls


__all__ = [
    "clear_reporter_registry",
    "get_reporter_registry",
    "register_builtin",
    "reporter",
    "resolve_reporter",
    "resolve_reporters",
]
