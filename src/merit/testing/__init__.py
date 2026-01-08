"""Testing framework for AI agents.

Provides pytest-like test discovery and resource injection.
"""

from .case import Case, iter_cases, valididate_cases_for_sut
from .discovery import TestItem, collect
from .parametrize import parametrize
from .repeat import RepeatData, repeat
from .resources import ResourceResolver, Scope, resource
from .runner import (
    MeritRun,
    RunEnvironment,
    Runner,
    RunResult,
    TestExecution,
    TestResult,
    TestStatus,
    run,
)
from .tags import tag


__all__ = [
    "Case",
    "iter_cases",
    "valididate_cases_for_sut",
    "MeritRun",
    "RepeatData",
    "ResourceResolver",
    "RunEnvironment",
    "RunResult",
    "Runner",
    "Scope",
    "TestExecution",
    "TestItem",
    "TestResult",
    "TestStatus",
    "collect",
    "parametrize",
    "repeat",
    "resource",
    "run",
    "tag",
]
