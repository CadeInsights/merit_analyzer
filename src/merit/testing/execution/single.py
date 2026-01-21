"""Single test execution."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from merit.assertions.base import AssertionResult
from merit.context import (
    ResolverContext,
    TestContext,
    assertions_collector,
    resolver_context_scope,
    test_context_scope,
)
from merit.resources import ResourceResolver
from merit.resources.resolver import Scope
from merit.testing.execution.interfaces import MeritTest
from merit.testing.execution.result_builder import ResultBuilder
from merit.testing.execution.tracer import TestTracer
from merit.testing.models import MeritTestDefinition, TestResult, TestStatus


logger = logging.getLogger(__name__)


@dataclass
class SingleMeritTest(MeritTest):
    """Executes a single test directly."""

    definition: MeritTestDefinition
    params: dict[str, Any]
    tracer: TestTracer
    result_builder: ResultBuilder

    def __post_init__(self) -> None:
        """Validate that this test has no modifiers."""
        if self.definition.modifiers:
            raise ValueError("SingleMeritTest should not have modifiers")

    async def execute(self, resolver: ResourceResolver) -> TestResult:
        """Execute the test and return result."""
        if self.definition.skip_reason:
            return TestResult(
                status=TestStatus.SKIPPED,
                duration_ms=0,
                error=AssertionError(self.definition.skip_reason),
            )

        # fork resolver for case isolation
        forked_resolver = resolver.fork_for_case()

        ctx = TestContext(item=self.definition)
        assertion_results: list[AssertionResult] = []
        error: Exception | None = None

        with self.tracer.span(self.definition) as span:
            start = time.perf_counter()

            with test_context_scope(ctx), assertions_collector(assertion_results):
                try:
                    kwargs = await self._resolve_params(forked_resolver)
                    await self._invoke(kwargs)
                except Exception as e:  # noqa: BLE001
                    error = e
                finally:
                    # teardown case scope - must run even if test fails
                    try:
                        await forked_resolver.teardown_scope(Scope.CASE)
                    except Exception as teardown_err:
                        # teardown errors dont mask tests errors
                        logger.warning(f"Error during resource teardown: {teardown_err}")
                        if error is None:
                            error = teardown_err

            duration_ms = (time.perf_counter() - start) * 1000
            result = self.result_builder.build(
                self.definition, duration_ms, assertion_results, error
            )
            result.trace_id = self.tracer.get_trace_id(span)
            self.tracer.record(span, result)
            return result

    async def _resolve_params(self, resolver: ResourceResolver) -> dict[str, Any]:
        """Resolve test parameters from resources."""
        kwargs = dict(self.params)
        with resolver_context_scope(ResolverContext(consumer_name=self.definition.name)):
            for param in self.definition.params:
                if param not in kwargs:
                    kwargs[param] = await resolver.resolve(param)
        return kwargs

    async def _invoke(self, kwargs: dict[str, Any]) -> None:
        """Invoke the test function."""
        fn = self.definition.fn
        instance = None

        if self.definition.class_name:
            cls = fn.__globals__.get(self.definition.class_name)

            if cls is None:
                raise RuntimeError(
                    f"Test class '{self.definition.class_name}' not found for test '{self.definition.name}'"
                )

            instance = cls()

        if instance:
            if self.definition.is_async:
                await fn(instance, **kwargs)
            else:
                fn(instance, **kwargs)
        elif self.definition.is_async:
            await fn(**kwargs)
        else:
            fn(**kwargs)
