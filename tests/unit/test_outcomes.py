"""Tests for imperative outcomes (skip, fail, xfail)."""

from pathlib import Path

import pytest

from merit.testing import TestStatus, fail, skip, xfail
from merit.testing.models import MeritTestDefinition
from merit.testing.outcomes import SkipTest, XFailTest
from merit.testing.runner import Runner


def make_item(fn, name: str | None = None, is_async: bool = False) -> MeritTestDefinition:
    return MeritTestDefinition(
        fn=fn,
        name=name or fn.__name__,
        module_path=Path("test_module.py"),
        is_async=is_async,
        params=[],
        class_name=None,
        modifiers=[],
        tags=set(),
    )


class TestImperativeSkip:
    @pytest.mark.asyncio
    async def test_skip_marks_test_as_skipped(self):
        def skipping_test():
            skip("not ready")

        item = make_item(skipping_test)
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.skipped == 1
        assert result.result.executions[0].result.status == TestStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_skip_with_reason(self):
        def skipping_test():
            skip("missing dependency")

        item = make_item(skipping_test)
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.skipped == 1
        assert "missing dependency" in str(result.result.executions[0].result.error)

    @pytest.mark.asyncio
    async def test_skip_without_reason(self):
        def skipping_test():
            skip()

        item = make_item(skipping_test)
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.skipped == 1
        assert isinstance(result.result.executions[0].result.error, SkipTest)

    @pytest.mark.asyncio
    async def test_skip_stops_execution(self):
        executed = []

        def skipping_test():
            executed.append("before")
            skip("stopping")
            executed.append("after")

        item = make_item(skipping_test)
        runner = Runner(reporters=[])
        await runner.run(items=[item])

        assert executed == ["before"]


class TestImperativeFail:
    @pytest.mark.asyncio
    async def test_fail_marks_test_as_failed(self):
        def failing_test():
            fail("explicit failure")

        item = make_item(failing_test)
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.failed == 1
        assert result.result.executions[0].result.status == TestStatus.FAILED

    @pytest.mark.asyncio
    async def test_fail_with_reason(self):
        def failing_test():
            fail("something went wrong")

        item = make_item(failing_test)
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.failed == 1
        assert "something went wrong" in str(result.result.executions[0].result.error)

    @pytest.mark.asyncio
    async def test_fail_without_reason(self):
        def failing_test():
            fail()

        item = make_item(failing_test)
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.failed == 1

    @pytest.mark.asyncio
    async def test_fail_stops_execution(self):
        executed = []

        def failing_test():
            executed.append("before")
            fail("stopping")
            executed.append("after")

        item = make_item(failing_test)
        runner = Runner(reporters=[])
        await runner.run(items=[item])

        assert executed == ["before"]


class TestImperativeXFail:
    @pytest.mark.asyncio
    async def test_xfail_marks_test_as_xfailed(self):
        def xfailing_test():
            xfail("known bug")

        item = make_item(xfailing_test)
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.xfailed == 1
        assert result.result.executions[0].result.status == TestStatus.XFAILED

    @pytest.mark.asyncio
    async def test_xfail_with_reason(self):
        def xfailing_test():
            xfail("issue #123")

        item = make_item(xfailing_test)
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.xfailed == 1
        assert "issue #123" in str(result.result.executions[0].result.error)

    @pytest.mark.asyncio
    async def test_xfail_without_reason(self):
        def xfailing_test():
            xfail()

        item = make_item(xfailing_test)
        runner = Runner(reporters=[])
        result = await runner.run(items=[item])

        assert result.result.xfailed == 1
        assert isinstance(result.result.executions[0].result.error, XFailTest)

    @pytest.mark.asyncio
    async def test_xfail_stops_execution(self):
        executed = []

        def xfailing_test():
            executed.append("before")
            xfail("stopping")
            executed.append("after")

        item = make_item(xfailing_test)
        runner = Runner(reporters=[])
        await runner.run(items=[item])

        assert executed == ["before"]
