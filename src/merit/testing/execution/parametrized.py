"""Parametrized test execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from merit.resources import ResourceResolver
from merit.testing.execution.interfaces import MeritTest, TestFactory
from merit.testing.models import (
    MeritTestDefinition,
    ParameterSet,
    ParametrizeModifier,
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

    async def execute(self, resolver: ResourceResolver) -> TestResult:
        """Execute test for each parameter set and aggregate results."""
        sub_runs: list[TestResult] = []

        for ps in self.parameter_sets:
            child_def = replace(
                self.definition,
                modifiers=self.definition.modifiers[1:],
                id_suffix=ps.id_suffix,
            )
            child_params = {**self.params, **ps.values}
            child = self.factory.build(child_def, child_params)
            result = await child.execute(resolver)
            result.id_suffix = ps.id_suffix
            sub_runs.append(result)

        status = (
            TestStatus.FAILED if any(r.status.is_failure for r in sub_runs) else TestStatus.PASSED
        )
        duration = sum(r.duration_ms for r in sub_runs)

        return TestResult(status=status, duration_ms=duration, sub_runs=sub_runs)
