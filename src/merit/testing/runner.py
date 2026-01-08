"""Test runner for executing discovered tests."""

from __future__ import annotations

import asyncio
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from opentelemetry.trace import StatusCode

from merit.assertions.base import AssertionResult
from merit.context import (
    ResolverContext,
    TestContext,
    merit_run_scope,
    resolver_context_scope,
    test_context_scope,
)
from merit.predicates import (
    close_predicate_api_client,
    create_predicate_api_client,
)
from merit.testing.discovery import TestItem, collect
from merit.testing.resources import ResourceResolver, Scope, get_registry
from merit.tracing import clear_traces, get_tracer, init_tracing
from merit.version import __version__

if TYPE_CHECKING:
    from merit.context import TestContext
    from merit.reports.base import Reporter


@dataclass
class RunEnvironment:
    """Metadata about the environment where tests were executed."""

    # Git info
    commit_hash: str | None = None
    branch: str | None = None
    dirty: bool | None = None

    # System info
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    hostname: str = field(default_factory=socket.gethostname)
    working_directory: str = field(default_factory=os.getcwd)
    merit_version: str = __version__

    # Environment variables (filtered)
    env_vars: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "commit_hash": self.commit_hash,
            "branch": self.branch,
            "dirty": self.dirty,
            "python_version": self.python_version,
            "platform": self.platform,
            "hostname": self.hostname,
            "working_directory": self.working_directory,
            "merit_version": self.merit_version,
            "env_vars": self.env_vars,
        }


def _get_git_info() -> tuple[str | None, str | None, bool | None]:
    """Capture git metadata if available."""
    try:
        # Check if inside git repo
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            check=True,
            capture_output=True,
            timeout=1,
        )

        # Get hash
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        ).stdout.strip()

        # Get branch
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        ).stdout.strip()

        # Check dirty status
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        ).stdout.strip()
        dirty = bool(status)

        return commit, branch, dirty

    except (subprocess.SubprocessError, FileNotFoundError):
        return None, None, None


def _filter_env_vars() -> dict[str, str]:
    """Capture and mask relevant environment variables."""
    # TODO: detect ci_provider from env vars

    allowlist = {
        "MODEL_VENDOR",
        "INFERENCE_VENDOR",
        "CLOUD_ML_REGION",
        "GOOGLE_CLOUD_PROJECT",
        "AWS_REGION",
    }

    captured = {}
    for key, value in os.environ.items():
        if key in allowlist:
            captured[key] = value

    # Explicitly check for known keys to capture masked versions
    sensitive_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]

    for key in sensitive_keys:
        if key in os.environ:
            val = os.environ[key]
            if len(val) > 4:
                captured[key] = f"***{val[-4:]}"
            else:
                captured[key] = "***"

    return captured


def capture_environment() -> RunEnvironment:
    """Capture current environment metadata."""
    # TODO: capture frozen package versions for full reproducibility

    commit, branch, dirty = _get_git_info()

    return RunEnvironment(
        commit_hash=commit,
        branch=branch,
        dirty=dirty,
        env_vars=_filter_env_vars(),
    )


class TestStatus(Enum):
    """Test execution status."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    XFAILED = "xfailed"
    XPASSED = "xpassed"


@dataclass
class TestResult:
    """Result of a single test execution."""

    status: TestStatus
    duration_ms: float
    error: Exception | None = None
    assertion_results: list[AssertionResult] = field(default_factory=list)
    repeat_runs: list["TestResult"] | None = None


@dataclass
class TestExecution:
    """Complete record of a test execution, combining context and result.

    Encapsulates both the test context (inputs/setup) and the result (outcome)
    as a single execution record.
    """

    context: TestContext
    result: TestResult

    @property
    def item(self) -> TestItem:
        """The test item that was executed."""
        return self.context.item

    @property
    def status(self) -> TestStatus:
        """Convenience access to result status."""
        return self.result.status

    @property
    def duration_ms(self) -> float:
        """Convenience access to result duration."""
        return self.result.duration_ms


@dataclass
class RunResult:
    """Result of a complete test run."""

    executions: list[TestExecution] = field(default_factory=list)
    total_duration_ms: float = 0
    stopped_early: bool = False

    @property
    def passed(self) -> int:
        """Count of passed tests."""
        return sum(1 for e in self.executions if e.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        """Count of failed tests."""
        return sum(1 for e in self.executions if e.status == TestStatus.FAILED)

    @property
    def errors(self) -> int:
        """Count of errored tests."""
        return sum(1 for e in self.executions if e.status == TestStatus.ERROR)

    @property
    def skipped(self) -> int:
        """Count of skipped tests."""
        return sum(1 for e in self.executions if e.status == TestStatus.SKIPPED)

    @property
    def xfailed(self) -> int:
        """Count of expected failures."""
        return sum(1 for e in self.executions if e.status == TestStatus.XFAILED)

    @property
    def xpassed(self) -> int:
        """Count of unexpected passes for xfail tests."""
        return sum(1 for e in self.executions if e.status == TestStatus.XPASSED)

    @property
    def total(self) -> int:
        """Total test count."""
        return len(self.executions)


@dataclass
class MeritRun:
    """Complete record of a test run, combining environment and results.

    This is created at the top level of a merit test run and encapsulates
    all information about the run, including environment metadata and
    test executions with their contexts.

    Access result data via merit_run.result.* (e.g., merit_run.result.passed).
    """

    run_id: UUID = field(default_factory=uuid4)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None

    environment: RunEnvironment = field(default_factory=RunEnvironment)
    result: RunResult = field(default_factory=RunResult)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete run to a dictionary."""
        return {
            "run_id": str(self.run_id),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "environment": self.environment.to_dict(),
            "total_duration_ms": self.result.total_duration_ms,
            "stopped_early": self.result.stopped_early,
            "passed": self.result.passed,
            "failed": self.result.failed,
            "errors": self.result.errors,
            "skipped": self.result.skipped,
            "total": self.result.total,
        }


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
        self.timeout = timeout  # Per-test timeout in seconds
        # 0 = unlimited (capped at DEFAULT_MAX_CONCURRENCY), 1 = sequential, >1 = concurrent
        self.concurrency = concurrency if concurrency > 0 else self.DEFAULT_MAX_CONCURRENCY
        self.enable_tracing = enable_tracing
        self.trace_output = Path(trace_output) if trace_output else Path("traces.jsonl")

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

        # Initialize predicate client
        create_predicate_api_client()

        # Initialize tracing if enabled
        if self.enable_tracing:
            init_tracing(output_path=self.trace_output)
            clear_traces()

        if items is None:
            items = collect(path)

        if not items:
            await self._notify_no_tests_found()
            merit_run.end_time = datetime.now(UTC)
            return merit_run

        await self._notify_collection_complete(items)

        start = time.perf_counter()

        resolver = ResourceResolver(get_registry())

        with merit_run_scope(merit_run):
            if self.concurrency == 1:
                await self._run_sequential(items, resolver, merit_run)
            else:
                await self._run_concurrent(items, resolver, merit_run)

            # Teardown all resources
            await resolver.teardown()

        await close_predicate_api_client()

        merit_run.result.total_duration_ms = (time.perf_counter() - start) * 1000
        merit_run.end_time = datetime.now(UTC)

        await self._notify_run_complete(merit_run)

        # Tracing is streamed; surface the path for the user
        if self.enable_tracing:
            await self._notify_tracing_enabled(self.trace_output)

        return merit_run

    async def _run_sequential(
        self, items: list[TestItem], resolver: ResourceResolver, merit_run: MeritRun
    ) -> None:
        """Run tests sequentially."""
        failures = 0
        for item in items:
            ctx = self._create_test_context(item)
            with test_context_scope(ctx):
                result = await self._run_test(item, resolver, ctx)
                result.assertion_results = ctx.assertion_results.copy()
                execution = TestExecution(context=ctx, result=result)
                await self._notify_test_complete(execution)
                await resolver.teardown_scope(Scope.CASE)

            merit_run.result.executions.append(execution)

            if result.status in {TestStatus.FAILED, TestStatus.ERROR}:
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
                start = time.perf_counter()
                ctx = self._create_test_context(item)
                result: TestResult
                with test_context_scope(ctx):
                    try:
                        if self.timeout:
                            result = await asyncio.wait_for(
                                self._run_test(item, child_resolver, ctx),
                                timeout=self.timeout,
                            )
                        else:
                            result = await self._run_test(item, child_resolver, ctx)
                    except TimeoutError:
                        duration = (time.perf_counter() - start) * 1000
                        result = TestResult(
                            status=TestStatus.ERROR,
                            duration_ms=duration,
                            error=TimeoutError(f"Test timed out after {self.timeout}s"),
                        )
                    except Exception as e:
                        duration = (time.perf_counter() - start) * 1000
                        result = TestResult(
                            status=TestStatus.ERROR, duration_ms=duration, error=e
                        )
                    finally:
                        result.assertion_results = ctx.assertion_results.copy()
                        await child_resolver.teardown_scope(Scope.CASE)

                execution = TestExecution(context=ctx, result=result)

                if result.status in {TestStatus.FAILED, TestStatus.ERROR}:
                    async with lock:
                        failures += 1
                        if self.maxfail and failures >= self.maxfail:
                            stop_flag = True
                            merit_run.result.stopped_early = True
                return (idx, execution)

        indexed_results = await asyncio.gather(
            *[run_one(i, item) for i, item in enumerate(items)], return_exceptions=True
        )

        # Sort by original index for deterministic output order
        sorted_results = sorted(
            (r for r in indexed_results if isinstance(r, tuple) and r[1] is not None),
            key=lambda x: x[0],
        )

        for _, execution in sorted_results:
            merit_run.result.executions.append(execution)
            await self._notify_test_complete(execution)

        if merit_run.result.stopped_early:
            await self._notify_run_stopped_early(self.maxfail)

    def _create_test_context(self, item: TestItem) -> TestContext:
        """Create a TestContext from a TestItem."""
        return TestContext(item=item)

    async def _run_repeated_test(
        self, item: TestItem, resolver: ResourceResolver, ctx: TestContext
    ) -> TestResult:
        """Execute a test multiple times and aggregate results."""
        start = time.perf_counter()
        repeat_runs: list[TestResult] = []
        min_passes = (
            item.repeat_min_passes if item.repeat_min_passes is not None else item.repeat_count
        )

        # Create a non-repeating item for individual runs
        single_item = TestItem(
            name=item.name,
            fn=item.fn,
            module_path=item.module_path,
            is_async=item.is_async,
            params=item.params,
            class_name=item.class_name,
            param_values=item.param_values,
            id_suffix=item.id_suffix,
            tags=item.tags,
            skip_reason=item.skip_reason,
            xfail_reason=item.xfail_reason,
            xfail_strict=item.xfail_strict,
            repeat_count=1,
            repeat_min_passes=None,
        )

        for _ in range(item.repeat_count):
            result = await self._run_single_test(single_item, resolver, ctx)
            repeat_runs.append(result)

        duration = (time.perf_counter() - start) * 1000
        passed_count = sum(1 for r in repeat_runs if r.status == TestStatus.PASSED)

        if passed_count >= min_passes:
            status = TestStatus.PASSED
        else:
            status = TestStatus.FAILED

        return TestResult(
            status=status,
            duration_ms=duration,
            repeat_runs=repeat_runs,
        )

    async def _run_test(
        self, item: TestItem, resolver: ResourceResolver, ctx: TestContext
    ) -> TestResult:
        """Execute a single test with resource injection."""
        # Handle repeated tests
        if item.repeat_count > 1:
            return await self._run_repeated_test(item, resolver, ctx)

        return await self._run_single_test(item, resolver, ctx)

    async def _run_single_test(
        self, item: TestItem, resolver: ResourceResolver, ctx: TestContext
    ) -> TestResult:
        """Execute a single test run (no repeat handling)."""
        start = time.perf_counter()

        if item.skip_reason:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                status=TestStatus.SKIPPED,
                duration_ms=duration,
                error=AssertionError(item.skip_reason),
            )

        expect_failure = item.xfail_reason is not None

        kwargs = {}
        if item.param_values:
            kwargs.update(item.param_values)

        resolver_ctx = ResolverContext(
            consumer_name=item.name,
        )

        with resolver_context_scope(resolver_ctx):
            for param in item.params:
                if param in kwargs:
                    continue
                try:
                    kwargs[param] = await resolver.resolve(param)
                except ValueError:
                    # Unknown resource - might be a non-resource param
                    pass

        # For class methods, instantiate the class
        instance = None
        if item.class_name:
            # Get the class from the function's globals
            cls = item.fn.__globals__.get(item.class_name)
            if cls:
                instance = cls()

        test_result = await self._execute_test_body(item, instance, kwargs, start, expect_failure)
        return test_result

    async def _execute_test_body(
        self,
        item: TestItem,
        instance: Any,
        kwargs: dict[str, Any],
        start: float,
        expect_failure: bool,
    ) -> TestResult:
        """Execute the test body, optionally wrapped in a trace span."""
        tracer = get_tracer() if self.enable_tracing else None
        span_context = tracer.start_as_current_span(f"test.{item.full_name}") if tracer else None

        try:
            if span_context:
                span = span_context.__enter__()
                span.set_attribute("test.name", item.name)
                span.set_attribute("test.module", str(item.module_path))
                if item.tags:
                    span.set_attribute("test.tags", list(item.tags))

            if instance:
                if item.is_async:
                    await item.fn(instance, **kwargs)
                else:
                    item.fn(instance, **kwargs)
            elif item.is_async:
                await item.fn(**kwargs)
            else:
                item.fn(**kwargs)

            duration = (time.perf_counter() - start) * 1000

            if span_context:
                span.set_attribute("test.status", "passed")
                span.set_attribute("test.duration_ms", duration)

            if expect_failure:
                if item.xfail_strict:
                    err = AssertionError(item.xfail_reason or "xfail marked test passed")
                    return TestResult(
                        status=TestStatus.FAILED, duration_ms=duration, error=err
                    )
                return TestResult(status=TestStatus.XPASSED, duration_ms=duration)
            return TestResult(status=TestStatus.PASSED, duration_ms=duration)

        except AssertionError as e:
            duration = (time.perf_counter() - start) * 1000

            if span_context:
                span.set_attribute("test.status", "failed")
                span.set_attribute("test.duration_ms", duration)
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)

            if expect_failure:
                err = AssertionError(item.xfail_reason) if item.xfail_reason else e
                return TestResult(
                    status=TestStatus.XFAILED, duration_ms=duration, error=err
                )
            return TestResult(status=TestStatus.FAILED, duration_ms=duration, error=e)

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000

            if span_context:
                span.set_attribute("test.status", "error")
                span.set_attribute("test.duration_ms", duration)
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)

            if expect_failure:
                err = AssertionError(item.xfail_reason) if item.xfail_reason else e
                return TestResult(
                    status=TestStatus.XFAILED, duration_ms=duration, error=err
                )
            return TestResult(status=TestStatus.ERROR, duration_ms=duration, error=e)

        finally:
            if span_context:
                span_context.__exit__(None, None, None)


def run(path: str | None = None) -> RunResult:
    """Run tests synchronously (convenience wrapper).

    Args:
        path: Path to discover tests from.

    Returns:
        RunResult with all test outcomes.
    """
    return asyncio.run(Runner().run(path=path))
