"""Parametrized test execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from merit.resources import ResourceResolver
from merit.testing.execution.interfaces import MeritTest, TestFactory
from merit.testing.models import (
    MeritTestDefinition,
    ParameterSet,
    ParametrizeModifier,
    TestExecution,
    TestResult,
    TestStatus,
)


@dataclass
class ParametrizedMeritTest(MeritTest):
    """Executes test for each parameter set, aggregates results."""

    definition: MeritTestDefinition
    params: dict[str, Any]
    parameter_sets: tuple[ParameterSet, ...]
    factory: TestFactory

    def __post_init__(self) -> None:
        """Validate that the first modifier is ParametrizeModifier."""
        if not self.definition.modifiers or not isinstance(
            self.definition.modifiers[0], ParametrizeModifier
        ):
            raise ValueError("ParametrizedMeritTest requires ParametrizeModifier as first modifier")

    async def execute(self, resolver: ResourceResolver) -> TestExecution:
        """Execute test for each parameter set and aggregate results."""
        sub_executions: list[TestExecution] = []

        for ps in self.parameter_sets:
            child_def = replace(
                self.definition,
                modifiers=self.definition.modifiers[1:],
                id_suffix=ps.id_suffix,
            )
            child_params = {**self.params, **ps.values}
            child = self.factory.build(child_def, child_params)
            child_execution = await child.execute(resolver)
            sub_executions.append(child_execution)

        has_failure = any(e.result.status.is_failure for e in sub_executions)
        status = TestStatus.FAILED if has_failure else TestStatus.PASSED
        duration = sum(e.result.duration_ms for e in sub_executions)

        return TestExecution(
            definition=self.definition,
            result=TestResult(status=status, duration_ms=duration),
            execution_id=uuid4(),
            sub_executions=sub_executions,
        )
