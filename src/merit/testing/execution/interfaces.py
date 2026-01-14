"""Execution base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from merit.resources.resolver import ResourceResolver
from merit.testing.models.definition import MeritTestDefinition
from merit.testing.models.result import TestResult


class MeritTest(ABC):
    """Executable test - single, repeated, or parametrized."""

    @abstractmethod
    async def execute(self, resolver: ResourceResolver) -> TestResult:
        """Execute the test and return result."""


class TestFactory(ABC):
    """Creates MeritTest instances from definitions."""

    @abstractmethod
    def build(
        self,
        definition: MeritTestDefinition,
        params: dict[str, Any] | None = None,
    ) -> MeritTest:
        """Build appropriate executable test from definition."""
