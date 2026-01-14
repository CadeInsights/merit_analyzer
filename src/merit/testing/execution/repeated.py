"""Repeated test execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from merit.resources import ResourceResolver
from merit.testing.execution.interfaces import MeritTest, TestFactory
from merit.testing.models import MeritTestDefinition, RepeatModifier, TestResult, TestStatus


@dataclass
class RepeatedMeritTest(MeritTest):
    """Executes test N times, aggregates results."""

    definition: MeritTestDefinition
    params: dict[str, Any]
    count: int
    min_passes: int
    factory: TestFactory

    def __post_init__(self) -> None:
        """Validate that the first modifier is RepeatModifier."""
        if not self.definition.modifiers or not isinstance(
            self.definition.modifiers[0], RepeatModifier
        ):
            raise ValueError("RepeatedMeritTest requires RepeatModifier as first modifier")

    async def execute(self, resolver: ResourceResolver) -> TestResult:
        """Execute test count times and aggregate results."""
        sub_runs: list[TestResult] = []

        for i in range(self.count):
            suffix = f"repeat={i}"
            child_def = replace(
                self.definition,
                modifiers=self.definition.modifiers[1:],
                id_suffix=suffix,
            )
            child = self.factory.build(child_def, self.params)
            result = await child.execute(resolver)
            result.id_suffix = suffix
            sub_runs.append(result)

        passed = sum(1 for r in sub_runs if r.status == TestStatus.PASSED)
        status = TestStatus.PASSED if passed >= self.min_passes else TestStatus.FAILED
        duration = sum(r.duration_ms for r in sub_runs)

        return TestResult(status=status, duration_ms=duration, sub_runs=sub_runs)
