"""Repeated test execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from merit.resources import ResourceResolver
from merit.testing.execution.interfaces import MeritTest, TestFactory
from merit.testing.models import (
    MeritTestDefinition,
    RepeatModifier,
    TestExecution,
    TestResult,
    TestStatus,
)


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

    async def execute(self, resolver: ResourceResolver) -> TestExecution:
        """Execute test count times and aggregate results."""
        tasks: list[asyncio.Task[TestExecution]] = []
        for i in range(self.count):
            suffix = f"repeat={i}"
            child_def = replace(
                self.definition,
                modifiers=self.definition.modifiers[1:],
                id_suffix=suffix,
            )
            child = self.factory.build(child_def, self.params)
            tasks.append(asyncio.create_task(child.execute(resolver)))

        sub_executions = await asyncio.gather(*tasks)

        passed = sum(1 for e in sub_executions if e.result.status == TestStatus.PASSED)
        status = TestStatus.PASSED if passed >= self.min_passes else TestStatus.FAILED
        duration = sum(e.result.duration_ms for e in sub_executions)

        return TestExecution(
            definition=self.definition,
            result=TestResult(status=status, duration_ms=duration),
            execution_id=uuid4(),
            sub_executions=sub_executions,
        )
