import asyncio
import io
from pathlib import Path

from rich.console import Console

from merit.testing import Runner
from merit.testing.discovery import TestItem
from merit.testing.tags import get_tag_data, tag


def test_tag_decorator_records_metadata():
    @tag("slow", "llm")
    @tag.skip(reason="network down")
    @tag.xfail(reason="flaky", strict=True)
    def sample():
        pass

    data = get_tag_data(sample)
    assert data.tags == {"slow", "llm", "skip", "xfail"}
    assert data.skip_reason == "network down"
    assert data.xfail_reason == "flaky"
    assert data.xfail_strict is True


def test_runner_handles_skip_and_xfail():
    console = Console(file=io.StringIO())
    runner = Runner(console=console)

    def merit_skip():
        raise AssertionError("should not run")

    skip_item = TestItem(
        name="merit_skip",
        fn=merit_skip,
        module_path=Path("sample.py"),
        is_async=False,
        params=[],
        skip_reason="skip me",
        tags={"skip"},
    )

    def merit_xfail():
        raise AssertionError("boom")

    xfail_item = TestItem(
        name="merit_xfail",
        fn=merit_xfail,
        module_path=Path("sample.py"),
        is_async=False,
        params=[],
        xfail_reason="known bug",
        tags={"xfail"},
    )

    run_result = asyncio.run(runner.run(items=[skip_item, xfail_item]))

    assert run_result.skipped == 1
    assert run_result.xfailed == 1
    assert run_result.passed == 0


def test_repeat_decorator_records_metadata():
    @tag.repeat(5, min_passes=3)
    def sample():
        pass

    from merit.testing.tags import get_repeat_data
    data = get_repeat_data(sample)
    assert data is not None
    assert data.count == 5
    assert data.min_passes == 3

    tag_data = get_tag_data(sample)
    assert "repeat" in tag_data.tags


def test_repeat_decorator_validation():
    import pytest

    with pytest.raises(ValueError, match="repeat count must be >= 1"):
        @tag.repeat(0)
        def sample1():
            pass

    with pytest.raises(ValueError, match="min_passes must be >= 1"):
        @tag.repeat(5, min_passes=0)
        def sample2():
            pass

    with pytest.raises(ValueError, match="min_passes .* cannot exceed count"):
        @tag.repeat(3, min_passes=5)
        def sample3():
            pass


def test_runner_handles_repeat_all_pass():
    console = Console(file=io.StringIO())
    runner = Runner(console=console)

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
        repeat_count=5,
        repeat_min_passes=5,
        tags={"repeat"},
    )

    run_result = asyncio.run(runner.run(items=[repeat_item]))

    assert run_result.passed == 1
    assert call_count == 5
    assert run_result.results[0].repeat_runs is not None
    assert len(run_result.results[0].repeat_runs) == 5
    assert all(r.status.value == "passed" for r in run_result.results[0].repeat_runs)


def test_runner_handles_repeat_partial_pass():
    console = Console(file=io.StringIO())
    runner = Runner(console=console)

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
        repeat_count=5,
        repeat_min_passes=3,
        tags={"repeat"},
    )

    run_result = asyncio.run(runner.run(items=[repeat_item]))

    assert run_result.passed == 1
    assert call_count == 5
    assert run_result.results[0].repeat_runs is not None
    assert len(run_result.results[0].repeat_runs) == 5
    passed = sum(1 for r in run_result.results[0].repeat_runs if r.status.value == "passed")
    assert passed == 3


def test_runner_handles_repeat_insufficient_passes():
    console = Console(file=io.StringIO())
    runner = Runner(console=console)

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
        repeat_count=5,
        repeat_min_passes=3,
        tags={"repeat"},
    )

    run_result = asyncio.run(runner.run(items=[repeat_item]))

    assert run_result.failed == 1
    assert call_count == 5
    assert run_result.results[0].repeat_runs is not None
    assert len(run_result.results[0].repeat_runs) == 5
    passed = sum(1 for r in run_result.results[0].repeat_runs if r.status.value == "passed")
    assert passed == 2
