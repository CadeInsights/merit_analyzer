"""Test outcome control flow."""

from typing import NoReturn


class SkipTest(BaseException):
    """Skip the current test."""

    def __init__(self, reason: str = "") -> None:
        self.reason = reason
        super().__init__(reason)


class FailTest(BaseException):
    """Explicitly fail the current test."""

    def __init__(self, reason: str = "") -> None:
        self.reason = reason
        super().__init__(reason)


class XFailTest(BaseException):
    """Mark the current test as expected to fail."""

    def __init__(self, reason: str = "") -> None:
        self.reason = reason
        super().__init__(reason)


def skip(reason: str = "") -> NoReturn:
    """Skip the current test."""
    raise SkipTest(reason)


def fail(reason: str = "") -> NoReturn:
    """Explicitly fail the current test."""
    raise FailTest(reason)


def xfail(reason: str = "") -> NoReturn:
    """Mark the current test as expected to fail and stop execution."""
    raise XFailTest(reason)
