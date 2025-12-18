"""Testing framework for AI agents.

Provides pytest-like test discovery and resource injection.
"""

from .case import Case
from .discovery import TestItem, collect
from .parametrize import parametrize
from .resources import ResourceResolver, Scope, resource
from .runner import RunEnvironment, Runner, RunResult, TestResult, TestStatus, run
from .suite import Suite
from .tags import tag, RepeatData


__all__ = [
    "Case",
    "RepeatData",
    "ResourceResolver",
    "RunEnvironment",
    "RunResult",
    "Runner",
    "Scope",
    "Suite",
    "TestItem",
    "TestResult",
    "TestStatus",
    "collect",
    "parametrize",
    "resource",
    "run",
    "tag",
]
