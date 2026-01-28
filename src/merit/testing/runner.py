"""Test runner for executing discovered tests."""

from __future__ import annotations

import asyncio
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from merit.context import (
    metric_results_collector,
    runner_scope,
)
from merit.context.output_capture import sys_output_capture
from merit.metrics_.base import MetricResult
from merit.predicates import (
    close_predicate_api_client,
    create_predicate_api_client,
)
from merit.reports import ConsoleReporter, Reporter
from merit.resources import ResourceResolver, get_registry
from merit.storage import SQLiteStore
from merit.testing.discovery import collect
from merit.testing.environment import capture_environment
from merit.testing.execution import DefaultTestFactory, ResultBuilder, TestTracer
from merit.testing.models import (
    MeritRun,
    MeritTestDefinition,
    TestExecution,
    TestResult,
    TestStatus,
)
from merit.tracing import clear_traces, init_tracing


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
        from merit.reports import ConsoleReporter
        runner = Runner(reporters=[ConsoleReporter()])
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
        capture_output: bool = True,
        save_to_db: bool = True,
        db_path: Path | str | None = None,
    ) -> None:
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
        self.capture_output = capture_output
        self.save_to_db = save_to_db
        self.db_path = Path(db_path) if db_path else None

        self._tracer = TestTracer(enabled=enable_tracing)
        self._result_builder = ResultBuilder()
        self._factory = DefaultTestFactory(
            tracer=self._tracer,
            result_builder=self._result_builder,
        )

        # Used in single.py test execution through run_context
        self.semaphore: asyncio.Semaphore | None = None
        self.stop_flag: bool = False

        self.merit_run: MeritRun | None = None

    async def _notify_no_tests_found(self) -> None:
        await asyncio.gather(*[r.on_no_tests_found() for r in self.reporters])

    async def _notify_collection_complete(self, items: list[MeritTestDefinition]) -> None:
        await asyncio.gather(*[r.on_collection_complete(items) for r in self.reporters])

    async def _notify_test_complete(self, execution: TestExecution) -> None:
        await asyncio.gather(*[r.on_test_complete(execution) for r in self.reporters])

    async def _notify_run_complete(self, merit_run: MeritRun) -> None:
        await asyncio.gather(*[r.on_run_complete(merit_run) for r in self.reporters])

    async def _notify_run_stopped_early(self, failure_count: int) -> None:
        await asyncio.gather(*[r.on_run_stopped_early(failure_count) for r in self.reporters])

    async def _notify_tracing_enabled(self, output_path: Path) -> None:
        await asyncio.gather(*[r.on_tracing_enabled(output_path) for r in self.reporters])

    async def run(
        self, items: list[MeritTestDefinition] | None = None, path: str | None = None
    ) -> MeritRun:
        """Run tests and return results.

        Args:
            items: Pre-collected test items, or None to discover.
            path: Path to discover tests from if items not provided.

        Returns:
            MeritRun with environment, results, and test executions.
        """
        environment = capture_environment()
        self.merit_run = MeritRun(environment=environment)

        create_predicate_api_client()

        if self.enable_tracing:
            init_tracing(output_path=self.trace_output)
            clear_traces()

        if items is None:
            items = collect(path)

        if not items:
            await self._notify_no_tests_found()
            self.merit_run.end_time = datetime.now(UTC)
            return self.merit_run

        if self.fail_fast:
            for item in items:
                item.fail_fast = True

        metric_results: list[MetricResult] = []
        start = time.perf_counter()

        with (
            runner_scope(self),
            sys_output_capture(swallow=self.capture_output),
            metric_results_collector(metric_results),
        ):
            await self._notify_collection_complete(items)

            resolver = ResourceResolver(get_registry())

            self.semaphore = asyncio.Semaphore(self.concurrency)
            self.stop_flag = False

            execution = self._execute_run(
                items=items,
                resolver=resolver,
                merit_run=self.merit_run,
            )

            run_task = asyncio.create_task(execution)

            try:
                if self.timeout:
                    await asyncio.wait_for(run_task, timeout=self.timeout)
                else:
                    await run_task
            except TimeoutError:
                self.merit_run.result.stopped_early = True
                self.stop_flag = True

        await close_predicate_api_client()

        self.merit_run.result.total_duration_ms = (time.perf_counter() - start) * 1000
        self.merit_run.result.metric_results = metric_results.copy()
        self.merit_run.end_time = datetime.now(UTC)

        await self._notify_run_complete(self.merit_run)

        if self.enable_tracing:
            await self._notify_tracing_enabled(self.trace_output)

        if self.save_to_db:
            try:
                SQLiteStore(self.db_path).save_run(self.merit_run)
            except Exception as e:
                warnings.warn(f"Failed to persist run to database: {e}", RuntimeWarning)

        return self.merit_run

    async def _execute_run(
        self,
        *,
        items: list[MeritTestDefinition],
        resolver: ResourceResolver,
        merit_run: MeritRun,
    ) -> None:
        """Execute the test run with the given items and resolver."""
        try:
            if self.concurrency == 1:
                await self._run_sequential(items, resolver, merit_run)
            else:
                await self._run_concurrent(items, resolver, merit_run)
        finally:
            await resolver.teardown()

    async def _execute_item(
        self, item: MeritTestDefinition, resolver: ResourceResolver
    ) -> TestExecution:
        """Execute a single test with error handling."""
        test = self._factory.build(item)
        t_start = time.perf_counter()

        try:
            execution = await test.execute(resolver)
        except Exception as e:
            duration = (time.perf_counter() - t_start) * 1000
            return TestExecution(
                definition=item,
                result=TestResult(status=TestStatus.ERROR, duration_ms=duration, error=e),
                execution_id=uuid4(),
            )

        return execution

    async def _run_sequential(
        self, items: list[MeritTestDefinition], resolver: ResourceResolver, merit_run: MeritRun
    ) -> None:
        """Run tests sequentially."""
        failures = 0

        for item in items:
            if self.stop_flag:
                break
            execution = await self._execute_item(item, resolver)
            await self._notify_test_complete(execution)

            merit_run.result.executions.append(execution)

            if execution.result.status.is_failure:
                failures += 1
                if self.maxfail and failures >= self.maxfail:
                    merit_run.result.stopped_early = True
                    self.stop_flag = True
                    await self._notify_run_stopped_early(self.maxfail)
                    break

    async def _run_concurrent(
        self, items: list[MeritTestDefinition], resolver: ResourceResolver, merit_run: MeritRun
    ) -> None:
        """Run tests concurrently."""
        lock = asyncio.Lock()
        failures = 0
        results: list[TestExecution | None] = [None] * len(items)

        async def run_one(idx: int, item: MeritTestDefinition) -> None:
            nonlocal failures

            if self.stop_flag:
                return

            execution = await self._execute_item(item, resolver)

            if execution.result.status.is_failure:
                async with lock:
                    failures += 1
                    if self.maxfail and failures >= self.maxfail:
                        self.stop_flag = True
                        merit_run.result.stopped_early = True

            results[idx] = execution

        await asyncio.gather(
            *[run_one(i, item) for i, item in enumerate(items)], return_exceptions=True
        )

        for execution in results:
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
