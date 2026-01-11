"""Utilities for parameterizing merit tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from merit.testing.models import ParameterSet, ParametrizeModifier


def _normalize_argnames(argnames: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(argnames, str):
        parts = [name.strip() for name in argnames.split(",") if name.strip()]
    else:
        parts = [str(name) for name in argnames]
    if not parts:
        msg = "parametrize() requires at least one argument name"
        raise ValueError(msg)
    return tuple(parts)


def _normalize_values(raw: Any, expected: int) -> tuple[Any, ...]:
    if expected == 1 and not isinstance(raw, (tuple, list)):
        return (raw,)
    if not isinstance(raw, (tuple, list)):
        msg = "parametrize() values must be tuples or lists"
        raise TypeError(msg)
    values = tuple(raw)
    if len(values) != expected:
        msg = f"parametrize() expected {expected} values, got {len(values)}"
        raise ValueError(msg)
    return values


def _format_id(names: tuple[str, ...], values: tuple[Any, ...]) -> str:
    formatted = []
    for name, value in zip(names, values):
        if isinstance(value, (int, float, str, bool)) or value is None:
            val = str(value)
        else:
            val = value.__class__.__name__
        formatted.append(f"{name}={val}")
    return "-".join(formatted)


def parametrize(
    argnames: str | Sequence[str],
    argvalues: Iterable[Any],
    *,
    ids: Sequence[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Parameterize a test function or method.

    Examples:
    --------
    >>> @parametrize("prompt,expected", [("hi", "Hello hi"), ("hey", "Hello hey")])
    ... def merit_chat(prompt, expected): ...
    """
    names = _normalize_argnames(argnames)
    values_list = tuple(_normalize_values(value, len(names)) for value in argvalues)
    if not values_list:
        msg = "parametrize() requires at least one value set"
        raise ValueError(msg)

    ids_tuple: tuple[str, ...] | None = None
    if ids is not None:
        ids_tuple = tuple(str(identifier) for identifier in ids)
        if len(ids_tuple) != len(values_list):
            msg = "parametrize() ids must match number of value sets"
            raise ValueError(msg)

    parameter_sets = tuple(
        ParameterSet(
            values=dict(zip(names, vals)),
            id_suffix=ids_tuple[i] if ids_tuple else _format_id(names, vals),
        )
        for i, vals in enumerate(values_list)
    )

    modifier = ParametrizeModifier(parameter_sets=parameter_sets)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        modifiers: list = getattr(fn, "__merit_modifiers__", [])
        modifiers.append(modifier)
        fn.__merit_modifiers__ = modifiers
        return fn

    return decorator
