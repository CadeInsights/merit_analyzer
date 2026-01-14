"""Testing models - pure data classes."""

from merit.testing.models.definition import MeritTestDefinition
from merit.testing.models.modifiers import (
    Modifier,
    ParameterSet,
    ParametrizeModifier,
    RepeatModifier,
)
from merit.testing.models.result import TestExecution, TestResult, TestStatus
from merit.testing.models.run import MeritRun, RunEnvironment, RunResult


# Backwards compatibility alias
TestItem = MeritTestDefinition

__all__ = [
    "MeritRun",
    "MeritTestDefinition",
    "Modifier",
    "ParameterSet",
    "ParametrizeModifier",
    "RepeatModifier",
    "RunEnvironment",
    "RunResult",
    "TestExecution",
    "TestItem",  # alias
    "TestResult",
    "TestStatus",
]
