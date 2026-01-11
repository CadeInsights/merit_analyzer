"""Testing framework for AI agents.

Provides pytest-like test discovery and resource injection.
"""

from .case import Case, iter_cases, valididate_cases_for_sut
from .discovery import collect
from .environment import capture_environment
from .executor import TestExecutor
from .models import (
    MeritRun,
    Modifier,
    ParameterSet,
    ParametrizeModifier,
    RepeatModifier,
    RunEnvironment,
    RunResult,
    TestExecution,
    TestItem,
    TestResult,
    TestStatus,
)
from .parametrize import parametrize
from .repeat import repeat
from .resources import ResourceResolver, Scope, resource
from .runner import Runner, run
from .tags import tag


__all__ = [
    "Case",
    "MeritRun",
    "Modifier",
    "ParameterSet",
    "ParametrizeModifier",
    "RepeatModifier",
    "ResourceResolver",
    "RunEnvironment",
    "RunResult",
    "Runner",
    "Scope",
    "TestExecution",
    "TestExecutor",
    "TestItem",
    "TestResult",
    "TestStatus",
    "capture_environment",
    "collect",
    "iter_cases",
    "parametrize",
    "repeat",
    "resource",
    "run",
    "tag",
    "valididate_cases_for_sut",
]
