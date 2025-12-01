"""Testing framework for AI agents.

Provides pytest-like test discovery and resource injection.
"""

from .case import Case
from .discovery import TestItem, collect
from .parametrize import parametrize
from .resources import ResourceResolver, Scope, resource
from .runner import Runner, RunResult, TestResult, TestStatus, run
from .suite import Suite
from .tags import tag


__all__ = [
    "Case",
    "Suite",
    "parametrize",
    "tag",
    "TestItem",
    "collect",
    "resource",
    "ResourceResolver",
    "Scope",
    "Runner",
    "RunResult",
    "TestResult",
    "TestStatus",
    "run",
]
