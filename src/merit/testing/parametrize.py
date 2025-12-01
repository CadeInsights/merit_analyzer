"""Utilities for parameterizing merit tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class ParameterConfig:
    """Definition of a single @parametrize decorator."""

    names: tuple[str, ...]
    values: tuple[tuple[Any, ...], ...]
    ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ParameterSet:
    """Concrete parameter combination for an individual test run."""

    values: dict[str, Any]
    id_suffix: str


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

    Examples
    --------
    >>> @parametrize("prompt,expected", [("hi", "Hello hi"), ("hey", "Hello hey")])
    ... def merit_chat(prompt, expected):
    ...     ...
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

    config = ParameterConfig(names=names, values=values_list, ids=ids_tuple)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        configs: list[ParameterConfig] = getattr(fn, "__merit_parametrize__", [])
        configs.append(config)
        setattr(fn, "__merit_parametrize__", configs)
        return fn

    return decorator


def get_parameter_sets(fn: Callable[..., Any]) -> list[ParameterSet]:
    """Return all ParameterSet combinations for a callable."""

    configs: list[ParameterConfig] = getattr(fn, "__merit_parametrize__", [])
    if not configs:
        return []

    # Each decorator multiplies combinations (like pytest)
    combos: list[tuple[dict[str, Any], list[str]]] = [({}, [])]

    for config in configs:
        next_combos: list[tuple[dict[str, Any], list[str]]] = []
        for existing_values, existing_ids in combos:
            for idx, value_tuple in enumerate(config.values):
                values_dict = dict(zip(config.names, value_tuple))
                merged_values = {**existing_values, **values_dict}

                identifier: str
                if config.ids:
                    identifier = config.ids[idx]
                else:
                    identifier = _format_id(config.names, value_tuple)

                next_combos.append((merged_values, [*existing_ids, identifier]))
        combos = next_combos

    parameter_sets: list[ParameterSet] = []
    for value_dict, id_parts in combos:
        suffix = "-".join(part for part in id_parts if part)
        parameter_sets.append(ParameterSet(values=value_dict, id_suffix=suffix or "params"))

    return parameter_sets
