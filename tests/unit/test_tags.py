import asyncio
from pathlib import Path

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
    runner = Runner(reporters=[])

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

    assert run_result.result.skipped == 1
    assert run_result.result.xfailed == 1
    assert run_result.result.passed == 0
