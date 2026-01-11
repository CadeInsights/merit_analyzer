import asyncio
from pathlib import Path

import pytest

from merit.testing import Runner, repeat
from merit.testing.models import RepeatModifier, TestItem


def test_repeat_decorator_records_metadata():
    @repeat(5, min_passes=3)
    def sample():
        pass

    modifiers = getattr(sample, "__merit_modifiers__", [])
    assert len(modifiers) == 1
    assert isinstance(modifiers[0], RepeatModifier)
    assert modifiers[0].count == 5
    assert modifiers[0].min_passes == 3


def test_repeat_decorator_validation():
    with pytest.raises(ValueError, match="repeat count must be >= 1"):

        @repeat(0)
        def sample1():
            pass

    with pytest.raises(ValueError, match="min_passes must be >= 1"):

        @repeat(5, min_passes=0)
        def sample2():
            pass

    with pytest.raises(ValueError, match="min_passes .* cannot exceed count"):

        @repeat(3, min_passes=5)
        def sample3():
            pass


def test_runner_handles_repeat_all_pass():
    runner = Runner(reporters=[])

    call_count = 0

    def merit_always_pass():
        nonlocal call_count
        call_count += 1

    repeat_item = TestItem(
        name="merit_always_pass",
        fn=merit_always_pass,
        module_path=Path("sample.py"),
        is_async=False,
        params=[],
        modifiers=[RepeatModifier(count=5, min_passes=5)],
        tags={"repeat"},
    )

    run_result = asyncio.run(runner.run(items=[repeat_item]))

    assert run_result.result.passed == 1
    assert call_count == 5
    assert run_result.result.executions[0].result.sub_runs is not None
    assert len(run_result.result.executions[0].result.sub_runs) == 5
    assert all(r.status.value == "passed" for r in run_result.result.executions[0].result.sub_runs)


def test_runner_handles_repeat_partial_pass():
    runner = Runner(reporters=[])

    call_count = 0

    def merit_flaky():
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            return
        raise AssertionError("flake")

    repeat_item = TestItem(
        name="merit_flaky",
        fn=merit_flaky,
        module_path=Path("sample.py"),
        is_async=False,
        params=[],
        modifiers=[RepeatModifier(count=5, min_passes=3)],
        tags={"repeat"},
    )

    run_result = asyncio.run(runner.run(items=[repeat_item]))

    assert run_result.result.passed == 1
    assert call_count == 5
    assert run_result.result.executions[0].result.sub_runs is not None
    assert len(run_result.result.executions[0].result.sub_runs) == 5
    passed = sum(
        1 for r in run_result.result.executions[0].result.sub_runs if r.status.value == "passed"
    )
    assert passed == 3


def test_runner_handles_repeat_insufficient_passes():
    runner = Runner(reporters=[])

    call_count = 0

    def merit_mostly_fail():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return
        raise AssertionError("fail")

    repeat_item = TestItem(
        name="merit_mostly_fail",
        fn=merit_mostly_fail,
        module_path=Path("sample.py"),
        is_async=False,
        params=[],
        modifiers=[RepeatModifier(count=5, min_passes=3)],
        tags={"repeat"},
    )

    run_result = asyncio.run(runner.run(items=[repeat_item]))

    assert run_result.result.failed == 1
    assert call_count == 5
    assert run_result.result.executions[0].result.sub_runs is not None
    assert len(run_result.result.executions[0].result.sub_runs) == 5
    passed = sum(
        1 for r in run_result.result.executions[0].result.sub_runs if r.status.value == "passed"
    )
    assert passed == 2
