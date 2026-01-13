"""Tests for merit.testing.runner module."""

import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from merit.context import TestContext
from merit.testing.environment import _filter_env_vars, capture_environment
from merit.testing.models import (
    RunEnvironment,
    RunResult,
    TestExecution,
    TestItem,
    TestResult,
    TestStatus,
)
from merit.testing.resources import clear_registry, resource
from merit.testing.runner import Runner


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the global registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


def make_item(
    fn,
    name: str | None = None,
    is_async: bool = False,
    params: list[str] | None = None,
    skip_reason: str | None = None,
    xfail_reason: str | None = None,
    xfail_strict: bool = False,
    id_suffix: str | None = None,
) -> TestItem:
    """Helper to create TestItem for testing."""
    return TestItem(
        fn=fn,
        name=name or fn.__name__,
        module_path=Path("test_module.py"),
        is_async=is_async,
        params=params or [],
        class_name=None,
        modifiers=[],
        tags=set(),
        skip_reason=skip_reason,
        xfail_reason=xfail_reason,
        xfail_strict=xfail_strict,
        id_suffix=id_suffix,
    )


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_counts_passed(self):
        result = RunResult()
        items = [make_item(lambda: None) for _ in range(3)]
        result.executions = [
            TestExecution(
                context=TestContext(item=items[0]),
                result=TestResult(status=TestStatus.PASSED, duration_ms=1),
            ),
            TestExecution(
                context=TestContext(item=items[1]),
                result=TestResult(status=TestStatus.PASSED, duration_ms=1),
            ),
            TestExecution(
                context=TestContext(item=items[2]),
                result=TestResult(status=TestStatus.FAILED, duration_ms=1),
            ),
        ]
        assert result.passed == 2
        assert result.failed == 1
        assert result.total == 3

    def test_counts_all_statuses(self):
        result = RunResult()
        items = [make_item(lambda: None) for _ in range(6)]
        result.executions = [
            TestExecution(
                context=TestContext(item=items[0]),
                result=TestResult(status=TestStatus.PASSED, duration_ms=1),
            ),
            TestExecution(
                context=TestContext(item=items[1]),
                result=TestResult(status=TestStatus.FAILED, duration_ms=1),
            ),
            TestExecution(
                context=TestContext(item=items[2]),
                result=TestResult(status=TestStatus.ERROR, duration_ms=1),
            ),
            TestExecution(
                context=TestContext(item=items[3]),
                result=TestResult(status=TestStatus.SKIPPED, duration_ms=1),
            ),
            TestExecution(
                context=TestContext(item=items[4]),
                result=TestResult(status=TestStatus.XFAILED, duration_ms=1),
            ),
            TestExecution(
                context=TestContext(item=items[5]),
                result=TestResult(status=TestStatus.XPASSED, duration_ms=1),
            ),
        ]
        assert result.passed == 1
        assert result.failed == 1
        assert result.errors == 1
        assert result.skipped == 1
        assert result.xfailed == 1
        assert result.xpassed == 1


class TestEnvironmentCapture:
    """Tests for environment capture functionality."""

    def test_filter_env_vars_masks_keys(self):
        """Test that sensitive keys are masked."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-1234567890abcdef",
                "ANTHROPIC_API_KEY": "sk-ant-1234567890abcdef",
                "MODEL_VENDOR": "openai",
                "RANDOM_VAR": "should_not_be_captured",
            },
            clear=True,
        ):
            captured = _filter_env_vars()

            assert captured["MODEL_VENDOR"] == "openai"
            assert "RANDOM_VAR" not in captured
            assert captured["OPENAI_API_KEY"] == "***cdef"
            assert captured["ANTHROPIC_API_KEY"] == "***cdef"

    def test_filter_env_vars_short_keys(self):
        """Test masking of short keys."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "123"}, clear=True):
            captured = _filter_env_vars()
            assert captured["OPENAI_API_KEY"] == "***"

    def test_capture_environment_structure(self):
        """Test that capture_environment returns a valid RunEnvironment."""
        env = capture_environment()
        assert isinstance(env, RunEnvironment)
        assert env.python_version is not None
        assert env.platform is not None
        assert env.merit_version is not None


class TestRunner:
    """Tests for Runner class."""

    @pytest.mark.asyncio
    async def test_runs_passing_test(self):
        def passing_test():
            assert True

        item = make_item(passing_test)
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.passed == 1
        assert result.result.failed == 0

    @pytest.mark.asyncio
    async def test_runs_failing_test(self):
        def failing_test():
            assert False, "expected failure"

        item = make_item(failing_test)
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.failed == 1
        assert "expected failure" in str(result.result.executions[0].result.error)

    @pytest.mark.asyncio
    async def test_runs_async_test(self):
        async def async_test():
            await asyncio.sleep(0.001)
            assert True

        item = make_item(async_test, is_async=True)
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.passed == 1

    @pytest.mark.asyncio
    async def test_handles_exception_as_error(self):
        def error_test():
            raise RuntimeError("something broke")

        item = make_item(error_test)
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.errors == 1
        assert isinstance(result.result.executions[0].result.error, RuntimeError)

    @pytest.mark.asyncio
    async def test_skipped_test(self):
        def skipped_test():
            pass

        item = make_item(skipped_test, skip_reason="not ready")
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.skipped == 1

    @pytest.mark.asyncio
    async def test_xfail_test_fails_as_expected(self):
        def xfail_test():
            assert False

        item = make_item(xfail_test, xfail_reason="known bug")
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.xfailed == 1

    @pytest.mark.asyncio
    async def test_xfail_test_passes_unexpectedly(self):
        def xpass_test():
            pass

        item = make_item(xpass_test, xfail_reason="expected to fail")
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.xpassed == 1

    @pytest.mark.asyncio
    async def test_xfail_strict_fails_on_pass(self):
        def strict_xfail_test():
            pass

        item = make_item(strict_xfail_test, xfail_reason="must fail", xfail_strict=True)
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.failed == 1


class TestResourceInjection:
    """Tests for resource injection in runner."""

    @pytest.mark.asyncio
    async def test_injects_resource(self):
        @resource
        def injected():
            return "injected_value"

        captured = []

        def test_with_resource(injected):
            captured.append(injected)

        item = make_item(test_with_resource, params=["injected"])
        runner = Runner(reporters=[])
        await runner.run(items=[item])

        assert captured == ["injected_value"]

    @pytest.mark.asyncio
    async def test_trace_context_requires_trace_flag(self):
        def test_needs_trace(trace_context):
            pass

        item = make_item(test_needs_trace, params=["trace_context"])
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.errors == 1
        assert "--trace" in str(result.result.executions[0].result.error)

    @pytest.mark.asyncio
    async def test_ignores_unknown_params(self):
        def test_unknown(unknown_param):
            pass

        item = make_item(test_unknown, params=["unknown_param"])
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        # Should error because unknown_param is not provided
        assert result.result.errors == 1


class TestMaxfail:
    """Tests for maxfail functionality."""

    @pytest.mark.asyncio
    async def test_stops_after_maxfail(self):
        fail_count = 0

        def failing():
            nonlocal fail_count
            fail_count += 1
            assert False

        items = [make_item(failing, name=f"fail_{i}") for i in range(5)]
        runner = Runner(reporters=[], maxfail=2)
        result = await runner.run(items=items)

        assert result.result.failed == 2
        assert result.result.stopped_early
        assert fail_count == 2

    @pytest.mark.asyncio
    async def test_maxfail_counts_errors_too(self):
        def error_test():
            raise RuntimeError

        items = [make_item(error_test, name=f"err_{i}") for i in range(5)]
        runner = Runner(reporters=[], maxfail=1)
        result = await runner.run(items=items)

        assert result.result.errors == 1
        assert result.result.stopped_early


class TestConcurrency:
    """Tests for concurrent test execution."""

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        start_times = []

        async def slow_test():
            start_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)

        items = [make_item(slow_test, name=f"slow_{i}", is_async=True) for i in range(3)]
        runner = Runner(reporters=[], concurrency=3)
        result = await runner.run(items=items)

        assert result.result.passed == 3
        # All should start within a small window (concurrent)
        assert max(start_times) - min(start_times) < 0.05

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        execution_order = []

        async def ordered_test(idx):
            execution_order.append(idx)
            await asyncio.sleep(0.01)

        items = []
        for i in range(3):

            async def test_fn(i=i):
                execution_order.append(i)
                await asyncio.sleep(0.01)

            items.append(make_item(test_fn, name=f"test_{i}", is_async=True))

        runner = Runner(reporters=[], concurrency=1)
        result = await runner.run(items=items)

        assert result.result.passed == 3
        # Should execute in order
        assert execution_order == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_concurrency_zero_caps_at_default(self):
        runner = Runner(reporters=[], concurrency=0)
        assert runner.concurrency == Runner.DEFAULT_MAX_CONCURRENCY

    @pytest.mark.asyncio
    async def test_concurrent_maxfail(self):
        fail_count = 0

        async def failing():
            nonlocal fail_count
            fail_count += 1
            await asyncio.sleep(0.01)
            assert False

        items = [make_item(failing, name=f"fail_{i}", is_async=True) for i in range(10)]
        runner = Runner(reporters=[], concurrency=5, maxfail=2)
        result = await runner.run(items=items)

        assert result.result.stopped_early
        # May have more than 2 due to concurrent execution, but should stop
        assert result.result.failed >= 2


class TestTimeout:
    """Tests for per-test timeout."""

    @pytest.mark.asyncio
    async def test_timeout_triggers_error(self):
        async def slow_test():
            await asyncio.sleep(10)

        item = make_item(slow_test, is_async=True)
        runner = Runner(reporters=[], concurrency=2, timeout=0.1)
        result = await runner.run(items=[item])

        assert result.result.errors == 1
        assert "timed out" in str(result.result.executions[0].result.error).lower()

    @pytest.mark.asyncio
    async def test_no_timeout_by_default(self):
        async def quick_test():
            await asyncio.sleep(0.01)

        item = make_item(quick_test, is_async=True)
        runner = Runner(reporters=[], concurrency=2)
        # timeout is None by default
        assert runner.timeout is None

        result = await runner.run(items=[item])
        assert result.result.passed == 1


class TestResultOrdering:
    """Tests for result ordering in concurrent execution."""

    @pytest.mark.asyncio
    async def test_results_ordered_by_discovery(self):
        async def varying_speed(delay):
            await asyncio.sleep(delay)

        items = []
        delays = [0.05, 0.01, 0.03]  # Different completion order
        for i, delay in enumerate(delays):

            async def test_fn(d=delay):
                await asyncio.sleep(d)

            items.append(make_item(test_fn, name=f"test_{i}", is_async=True))

        runner = Runner(reporters=[], concurrency=3)
        result = await runner.run(items=items)

        # Results should be in discovery order, not completion order
        names = [r.item.name for r in result.result.executions]
        assert names == ["test_0", "test_1", "test_2"]


class TestResourceTeardown:
    """Tests for resource teardown during test runs."""

    @pytest.mark.asyncio
    async def test_case_resources_torn_down_between_tests(self):
        teardown_count = 0

        @resource(scope="case")
        def case_res():
            yield "value"
            nonlocal teardown_count
            teardown_count += 1

        def test_with_case(case_res):
            assert case_res == "value"

        items = [
            make_item(test_with_case, name="test_1", params=["case_res"]),
            make_item(test_with_case, name="test_2", params=["case_res"]),
        ]

        runner = Runner(reporters=[])
        result = await runner.run(items=items)

        assert result.result.passed == 2
        assert teardown_count == 2

    @pytest.mark.asyncio
    async def test_suite_resources_shared(self):
        create_count = 0

        @resource(scope="suite")
        def suite_res():
            nonlocal create_count
            create_count += 1
            return f"suite_{create_count}"

        captured = []

        def test_suite(suite_res):
            captured.append(suite_res)

        items = [
            make_item(test_suite, name="test_1", params=["suite_res"]),
            make_item(test_suite, name="test_2", params=["suite_res"]),
        ]

        runner = Runner(reporters=[])
        await runner.run(items=items)

        assert create_count == 1
        assert captured == ["suite_1", "suite_1"]
