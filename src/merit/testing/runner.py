"""Test runner for executing discovered tests."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from merit.context import (
    merit_run_scope,
    metric_results_collector,
)
from merit.predicates import (
    close_predicate_api_client,
    create_predicate_api_client,
)
from merit.testing.discovery import collect
from merit.testing.environment import capture_environment
from merit.testing.executor import TestExecutor
from merit.testing.models import (
    MeritRun,
    TestExecution,
    TestItem,
    TestResult,
    TestStatus,
)
from merit.testing.resources import ResourceResolver, Scope, get_registry
from merit.tracing import clear_traces, init_tracing


if TYPE_CHECKING:
    from merit.metrics.base import MetricResult
    from merit.reports.base import Reporter


class Runner:
    """Executes discovered tests with resource injection.

    Examples:
        # Sequential execution (default)
        runner = Runner()
        result = await runner.run(path="tests/")

        # Concurrent execution with 5 workers
        runner = Runner(concurrency=5)
        result = await runner.run(path="tests/")

        # Unlimited concurrency (capped at 10)
        runner = Runner(concurrency=0)
        result = await runner.run(path="tests/")

        # Custom reporters
        from merit.reports import ConsoleReporter, JsonReporter
        runner = Runner(reporters=[ConsoleReporter(), JsonReporter("results.json")])
        result = await runner.run(path="tests/")
    """

    DEFAULT_MAX_CONCURRENCY = 10

    def __init__(
        self,
        *,
        reporters: list[Reporter] | None = None,
        maxfail: int | None = None,
        fail_fast: bool = False,
        verbosity: int = 0,
        concurrency: int = 1,
        timeout: float | None = None,
        enable_tracing: bool = False,
        trace_output: Path | str | None = None,
    ) -> None:
        from merit.reports import ConsoleReporter

        self.reporters: list[Reporter] = (
            reporters if reporters is not None else [ConsoleReporter(verbosity=verbosity)]
        )
        self.maxfail = maxfail if maxfail and maxfail > 0 else None
        self.fail_fast = fail_fast
        self.verbosity = verbosity
        self.timeout = timeout
        self.concurrency = concurrency if concurrency > 0 else self.DEFAULT_MAX_CONCURRENCY
        self.enable_tracing = enable_tracing
        self.trace_output = Path(trace_output) if trace_output else Path("traces.jsonl")
        self._executor = TestExecutor(enable_tracing=enable_tracing)

    async def _notify_no_tests_found(self) -> None:
        await asyncio.gather(*[r.on_no_tests_found() for r in self.reporters])

    async def _notify_collection_complete(self, items: list[TestItem]) -> None:
        await asyncio.gather(*[r.on_collection_complete(items) for r in self.reporters])

    async def _notify_test_complete(self, execution: TestExecution) -> None:
        await asyncio.gather(*[r.on_test_complete(execution) for r in self.reporters])

    async def _notify_run_complete(self, merit_run: MeritRun) -> None:
        await asyncio.gather(*[r.on_run_complete(merit_run) for r in self.reporters])

    async def _notify_run_stopped_early(self, failure_count: int) -> None:
        await asyncio.gather(*[r.on_run_stopped_early(failure_count) for r in self.reporters])

    async def _notify_tracing_enabled(self, output_path: Path) -> None:
        await asyncio.gather(*[r.on_tracing_enabled(output_path) for r in self.reporters])

    async def run(self, items: list[TestItem] | None = None, path: str | None = None) -> MeritRun:
        """Run tests and return results.

        Args:
            items: Pre-collected test items, or None to discover.
            path: Path to discover tests from if items not provided.

        Returns:
            MeritRun with environment, results, and test executions.
        """
        environment = capture_environment()
        merit_run = MeritRun(environment=environment)

        create_predicate_api_client()

        if self.enable_tracing:
            init_tracing(output_path=self.trace_output)
            clear_traces()

        if items is None:
            items = collect(path)

        if not items:
            await self._notify_no_tests_found()
            merit_run.end_time = datetime.now(UTC)
            return merit_run

        if self.fail_fast:
            for item in items:
                item.fail_fast = True

        await self._notify_collection_complete(items)

        resolver = ResourceResolver(get_registry())
        metric_results: list[MetricResult] = []

        start = time.perf_counter()
        with merit_run_scope(merit_run), metric_results_collector(metric_results):
            if self.concurrency == 1:
                await self._run_sequential(items, resolver, merit_run)
            else:
                await self._run_concurrent(items, resolver, merit_run)

            await resolver.teardown()

        await close_predicate_api_client()

        merit_run.result.total_duration_ms = (time.perf_counter() - start) * 1000
        merit_run.result.metric_results = metric_results.copy()
        merit_run.end_time = datetime.now(UTC)

        await self._notify_run_complete(merit_run)

        if self.enable_tracing:
            await self._notify_tracing_enabled(self.trace_output)

        return merit_run

    async def _run_sequential(
        self, items: list[TestItem], resolver: ResourceResolver, merit_run: MeritRun
    ) -> None:
        """Run tests sequentially."""
        failures = 0
        for item in items:
            ctx = self._executor.create_test_context(item)
            result = await self._executor.run_test(item, resolver, ctx)
            execution = TestExecution(context=ctx, result=result)
            await self._notify_test_complete(execution)
            await resolver.teardown_scope(Scope.CASE)

            merit_run.result.executions.append(execution)

            if result.status.is_failure:
                failures += 1
                if self.maxfail and failures >= self.maxfail:
                    merit_run.result.stopped_early = True
                    await self._notify_run_stopped_early(self.maxfail)
                    break

    async def _run_concurrent(
        self, items: list[TestItem], resolver: ResourceResolver, merit_run: MeritRun
    ) -> None:
        """Run tests concurrently with semaphore control."""
        semaphore = asyncio.Semaphore(self.concurrency)
        lock = asyncio.Lock()
        failures = 0
        stop_flag = False

        async def run_one(idx: int, item: TestItem) -> tuple[int, TestExecution | None]:
            nonlocal failures, stop_flag
            if stop_flag:
                return (idx, None)
            async with semaphore:
                if stop_flag:
                    return (idx, None)
                child_resolver = resolver.fork_for_case()
                ctx = self._executor.create_test_context(item)
                result: TestResult
                t_start = time.perf_counter()
                try:
                    if self.timeout:
                        result = await asyncio.wait_for(
                            self._executor.run_test(item, child_resolver, ctx),
                            timeout=self.timeout,
                        )
                    else:
                        result = await self._executor.run_test(item, child_resolver, ctx)
                except TimeoutError:
                    duration = (time.perf_counter() - t_start) * 1000
                    result = TestResult(
                        status=TestStatus.ERROR,
                        duration_ms=duration,
                        error=TimeoutError(f"Test timed out after {self.timeout}s"),
                    )
                except Exception as e:
                    duration = (time.perf_counter() - t_start) * 1000
                    result = TestResult(status=TestStatus.ERROR, duration_ms=duration, error=e)
                await child_resolver.teardown_scope(Scope.CASE)

                execution = TestExecution(context=ctx, result=result)

                if result.status.is_failure:
                    async with lock:
                        failures += 1
                        if self.maxfail and failures >= self.maxfail:
                            stop_flag = True
                            merit_run.result.stopped_early = True
                return (idx, execution)

        indexed_results = await asyncio.gather(
            *[run_one(i, item) for i, item in enumerate(items)], return_exceptions=True
        )

        sorted_results = sorted(
            (r for r in indexed_results if isinstance(r, tuple) and r[1] is not None),
            key=lambda x: x[0],
        )

        for _, execution in sorted_results:
            if execution is not None:
                merit_run.result.executions.append(execution)
                await self._notify_test_complete(execution)

        if merit_run.result.stopped_early and self.maxfail:
            await self._notify_run_stopped_early(self.maxfail)


def run(path: str | None = None) -> MeritRun:
    """Run tests synchronously (convenience wrapper).

    Args:
        path: Path to discover tests from.

    Returns:
        MeritRun with all test outcomes.
    """
    return asyncio.run(Runner().run(path=path))
