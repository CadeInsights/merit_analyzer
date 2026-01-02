"""Testing framework for AI agents.

Provides pytest-like test discovery and resource injection.
"""

from .case import Case, iter_cases, valididate_cases_for_sut
from .discovery import TestItem, collect
from .parametrize import parametrize
from .repeat import RepeatData, repeat
from .context import (
    ResolverContext,
    TestContext,
    RESOLVER_CONTEXT,
    TEST_CONTEXT,
    resolver_context_scope,
    test_context_scope,
)
from .resources import ResourceResolver, Scope, resource
from .runner import RunEnvironment, Runner, RunResult, TestResult, TestStatus, run
from .tags import tag


__all__ = [
    "Case",
    "iter_cases",
    "valididate_cases_for_sut",
    "RepeatData",
    "ResourceResolver",
    "RunEnvironment",
    "RunResult",
    "Runner",
    "Scope",
    "TestItem",
    "TestContext",
    "TestResult",
    "TestStatus",
    "collect",
    "RESOLVER_CONTEXT",
    "TEST_CONTEXT",
    "ResolverContext",
    "parametrize",
    "repeat",
    "resource",
    "resolver_context_scope",
    "run",
    "tag",
    "test_context_scope",
]
