"""Test modifiers - pure data classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterSet:
    """Concrete parameter combination for an individual test run."""

    values: dict[str, Any]
    id_suffix: str


@dataclass(frozen=True)
class RepeatModifier:
    """Repeat the inner execution N times."""

    count: int
    min_passes: int


@dataclass(frozen=True)
class ParametrizeModifier:
    """Run the inner execution for each parameter set."""

    parameter_sets: tuple[ParameterSet, ...]


Modifier = RepeatModifier | ParametrizeModifier
