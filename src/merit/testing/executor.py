"""Test execution logic."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import StatusCode

from merit.context import (
    ResolverContext,
    TestContext,
    assertions_collector,
    resolver_context_scope,
    test_context_scope,
)
from merit.testing.models import (
    Modifier,
    ParametrizeModifier,
    RepeatModifier,
    TestItem,
    TestResult,
    TestStatus,
)
from merit.tracing import get_tracer


if TYPE_CHECKING:
    from merit.assertions.base import AssertionResult
    from merit.testing.resources import ResourceResolver


class TestExecutor:
    """Executes individual test items."""

    def __init__(self, *, enable_tracing: bool = False) -> None:
        self.enable_tracing = enable_tracing

    def create_test_context(self, item: TestItem) -> TestContext:
        """Create a TestContext from a TestItem."""
        return TestContext(item=item)

    async def run_test(
        self, item: TestItem, resolver: ResourceResolver, ctx: TestContext
    ) -> TestResult:
        """Execute a test with its modifier chain."""
        if item.modifiers:
            return await self._run_with_modifiers(item, resolver, ctx, item.modifiers, 0, {})
        return await self._execute(item, resolver, ctx, {})

    async def _run_with_modifiers(
        self,
        item: TestItem,
        resolver: ResourceResolver,
        ctx: TestContext,
        modifiers: list[Modifier],
        index: int,
        params: dict[str, Any],
    ) -> TestResult:
        """Recursively process modifiers and execute the test."""
        if index >= len(modifiers):
            return await self._execute(item, resolver, ctx, params)

        mod = modifiers[index]
        next_index = index + 1

        if isinstance(mod, RepeatModifier):
            iterations = [(f"repeat={i}", params) for i in range(mod.count)]
            sub_runs = await self._run_iterations(
                item, resolver, ctx, modifiers, next_index, iterations
            )
            passed = sum(1 for r in sub_runs if r.status == TestStatus.PASSED)
            status = TestStatus.PASSED if passed >= mod.min_passes else TestStatus.FAILED

        elif isinstance(mod, ParametrizeModifier):
            iterations = [(ps.id_suffix, {**params, **ps.values}) for ps in mod.parameter_sets]
            sub_runs = await self._run_iterations(
                item, resolver, ctx, modifiers, next_index, iterations
            )
            status = (
                TestStatus.FAILED
                if any(r.status.is_failure for r in sub_runs)
                else TestStatus.PASSED
            )

        else:
            return await self._run_with_modifiers(
                item, resolver, ctx, modifiers, next_index, params
            )

        duration = sum(r.duration_ms for r in sub_runs)
        return TestResult(status=status, duration_ms=duration, sub_runs=sub_runs)

    async def _run_iterations(
        self,
        item: TestItem,
        resolver: ResourceResolver,
        ctx: TestContext,
        modifiers: list[Modifier],
        next_index: int,
        iterations: list[tuple[str, dict[str, Any]]],
    ) -> list[TestResult]:
        """Run multiple iterations with id_suffix tracking."""
        sub_runs: list[TestResult] = []
        original_id_suffix = item.id_suffix

        for id_suffix, params in iterations:
            item.id_suffix = id_suffix
            result = await self._run_with_modifiers(
                item, resolver, ctx, modifiers, next_index, params
            )
            result.id_suffix = id_suffix
            sub_runs.append(result)

        item.id_suffix = original_id_suffix
        return sub_runs

    async def _execute(
        self,
        item: TestItem,
        resolver: ResourceResolver,
        ctx: TestContext,
        params: dict[str, Any],
    ) -> TestResult:
        """Execute a single test."""
        if item.skip_reason:
            return TestResult(
                status=TestStatus.SKIPPED,
                duration_ms=0,
                error=AssertionError(item.skip_reason),
            )

        with self._trace_span(item) as span:
            return await self._run_test_body(item, ctx, resolver, params, span)

    async def _resolve_params(
        self, item: TestItem, resolver: ResourceResolver, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve test parameters from resources."""
        kwargs = dict(params)
        with resolver_context_scope(ResolverContext(consumer_name=item.name)):
            for param in item.params:
                if param not in kwargs:
                    if param == "trace_context" and not self.enable_tracing:
                        raise RuntimeError("`trace_context` requires `--trace` to be enabled")
                    try:
                        kwargs[param] = await resolver.resolve(param)
                    except ValueError:
                        pass
        return kwargs

    @contextmanager
    def _trace_span(self, item: TestItem):
        """Context manager for optional tracing."""
        if not self.enable_tracing:
            yield None
            return

        tracer = get_tracer()
        if not tracer:
            yield None
            return

        with tracer.start_as_current_span(f"test.{item.full_name}") as span:
            span.set_attribute("test.name", item.name)
            span.set_attribute("test.module", str(item.module_path))
            if item.tags:
                span.set_attribute("test.tags", list(item.tags))
            yield span

    async def _run_test_body(
        self,
        item: TestItem,
        ctx: TestContext,
        resolver: ResourceResolver,
        params: dict[str, Any],
        span: Any,
    ) -> TestResult:
        """Execute the test function with proper context scoping."""
        start = time.perf_counter()
        assertion_results: list[AssertionResult] = []
        error: Exception | None = None

        with (
            test_context_scope(ctx),
            assertions_collector(assertion_results),
        ):
            try:
                kwargs = await self._resolve_params(item, resolver, params)
                await self._invoke(item, kwargs)
            except Exception as e:
                error = e

            duration = (time.perf_counter() - start) * 1000
            result = self._make_result(item, span, duration, assertion_results, error)

        return result

    async def _invoke(self, item: TestItem, kwargs: dict[str, Any]) -> None:
        """Invoke the test function."""
        fn = item.fn
        instance = None

        if item.class_name:
            cls = fn.__globals__.get(item.class_name)
            if cls:
                instance = cls()

        if instance:
            if item.is_async:
                await fn(instance, **kwargs)
            else:
                fn(instance, **kwargs)
        elif item.is_async:
            await fn(**kwargs)
        else:
            fn(**kwargs)

    def _make_result(
        self,
        item: TestItem,
        span: Any,
        duration: float,
        assertion_results: list[AssertionResult],
        error: Exception | None,
    ) -> TestResult:
        """Create TestResult based on assertion results and raised exceptions."""
        expect_failure = item.xfail_reason is not None
        failed_assertions = [ar for ar in assertion_results if not ar.passed]
        has_assertion_failure = len(failed_assertions) > 0

        # Non-assertion exception = ERROR
        if error is not None and not isinstance(error, AssertionError):
            status = TestStatus.XFAILED if expect_failure else TestStatus.ERROR
            self._record_span(span, "error", duration, error)

        # Assertion failure from collector or raised AssertionError
        elif has_assertion_failure or isinstance(error, AssertionError):
            if expect_failure:
                status = TestStatus.XFAILED
            else:
                status = TestStatus.FAILED
            # Use first failed assertion's message if no raised error
            if not error and failed_assertions:
                msg = failed_assertions[0].error_message or failed_assertions[0].expression_repr
                error = AssertionError(msg)
            self._record_span(span, "failed", duration, error)

        # Success
        elif expect_failure:
            if item.xfail_strict:
                status = TestStatus.FAILED
                error = AssertionError(item.xfail_reason or "xfail test passed")
                self._record_span(span, "failed", duration, error)
            else:
                status = TestStatus.XPASSED
                self._record_span(span, "passed", duration)
        else:
            status = TestStatus.PASSED
            self._record_span(span, "passed", duration)

        return TestResult(
            status=status,
            duration_ms=duration,
            error=error,
            assertion_results=assertion_results.copy(),
        )

    def _record_span(
        self, span: Any, status: str, duration: float, error: Exception | None = None
    ) -> None:
        """Record span attributes."""
        if not span:
            return
        span.set_attribute("test.status", status)
        span.set_attribute("test.duration_ms", duration)
        if error:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
