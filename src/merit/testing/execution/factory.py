"""Test factory for creating executable tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from merit.testing.execution import parametrized, repeated, single
from merit.testing.execution.interfaces import MeritTest, TestFactory
from merit.testing.execution.result_builder import ResultBuilder
from merit.testing.execution.tracer import TestTracer
from merit.testing.models import (
    MeritTestDefinition,
    ParametrizeModifier,
    RepeatModifier,
)


@dataclass
class DefaultTestFactory(TestFactory):
    """Creates test instances with shared collaborators."""

    tracer: TestTracer
    result_builder: ResultBuilder

    def build(
        self,
        definition: MeritTestDefinition,
        params: dict[str, Any] | None = None,
    ) -> MeritTest:
        """Build appropriate executable test from definition."""
        params = params or {}

        match definition.modifiers:
            case []:
                return single.SingleMeritTest(
                    definition=definition,
                    params=params,
                    tracer=self.tracer,
                    result_builder=self.result_builder,
                )
            case [RepeatModifier() as mod, *_]:
                return repeated.RepeatedMeritTest(
                    definition=definition,
                    params=params,
                    count=mod.count,
                    min_passes=mod.min_passes,
                    factory=self,
                )
            case [ParametrizeModifier() as mod, *_]:
                return parametrized.ParametrizedMeritTest(
                    definition=definition,
                    params=params,
                    parameter_sets=mod.parameter_sets,
                    factory=self,
                )
            case [mod, *_]:
                raise NotImplementedError(f"Unknown modifier: {type(mod).__name__}")
