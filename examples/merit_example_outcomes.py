"""Example merit tests demonstrating imperative outcomes (skip, fail, xfail)."""

import os

import merit


def merit_skip_when_env_missing():
    """Skip test if required environment variable is not set."""
    if "API_KEY" not in os.environ:
        merit.skip("API_KEY not configured")

    assert os.environ["API_KEY"] is not None


def merit_skip_unconditionally():
    """Skip a test unconditionally."""
    merit.skip("not implemented yet")


def merit_fail_on_invalid_state():
    """Explicitly fail when detecting invalid state."""
    data = {"status": "error"}

    if data["status"] == "error":
        merit.fail("received error status from API")

    assert data["status"] == "ok"


def merit_xfail_known_bug():
    """Mark test as expected failure for a known bug."""
    merit.xfail("issue #42: division by zero not handled")

    result = 1 / 0  # This will raise ZeroDivisionError
    assert result == 0


def merit_conditional_xfail():
    """Conditionally mark as expected failure."""
    import sys

    if sys.version_info < (3, 12):
        merit.xfail("feature requires Python 3.12+")

    assert True


async def merit_async_skip():
    """Skip works in async tests too."""
    merit.skip("async feature not ready")


class MeritOutcomeTests:
    """Outcomes work in class methods."""

    def merit_skip_in_class(self):
        merit.skip("class test skipped")

    def merit_fail_in_class(self):
        merit.fail("class test failed")

    def merit_xfail_in_class(self):
        merit.xfail("class test xfailed")
