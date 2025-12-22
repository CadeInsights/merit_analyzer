"""Testing framework for AI agents.

Provides pytest-like test discovery and resource injection.
"""

from .case import Case, iter_cases, valididate_cases_for_sut
from .discovery import TestItem, collect
from .parametrize import parametrize
from .resources import ResourceResolver, Scope, resource
from .runner import RunEnvironment, Runner, RunResult, TestResult, TestStatus, run
from .tags import tag


__all__ = [
    "Case",
    "iter_cases",
    "valididate_cases_for_sut",
    "ResourceResolver",
    "RunEnvironment",
    "RunResult",
    "Runner",
    "Scope",
    "TestItem",
    "TestResult",
    "TestStatus",
    "collect",
    "parametrize",
    "resource",
    "run",
    "tag",
]
