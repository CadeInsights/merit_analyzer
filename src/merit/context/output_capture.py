"""Sys-level stdout/stderr capture for tests.

Captures Python-level output (print, sys.stdout.write). Does NOT capture
fd-level output (subprocesses, C extensions writing to fd 1/2).
"""

from __future__ import annotations

import io
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TextIO


@dataclass
class OutputBuffer:
    """Captured stdout/stderr for a single test."""

    stdout: io.StringIO = field(default_factory=io.StringIO)
    stderr: io.StringIO = field(default_factory=io.StringIO)
    _disabled: bool = field(default=False, repr=False)

    def readouterr(self) -> tuple[str, str]:
        """Read and clear captured output."""
        out = self.stdout.getvalue()
        err = self.stderr.getvalue()
        self.stdout.seek(0)
        self.stdout.truncate()
        self.stderr.seek(0)
        self.stderr.truncate()
        return out, err

    @contextmanager
    def disabled(self) -> Iterator[None]:
        """Temporarily disable capture, allowing output to pass through to real stdout/stderr."""
        self._disabled = True
        try:
            yield
        finally:
            self._disabled = False


class _SysStreamDispatcher(io.TextIOBase):
    """Replacement for sys.stdout/stderr that dispatches to per-test buffer."""

    def __init__(
        self,
        original: TextIO,
        buffer_var: ContextVar[OutputBuffer | None],
        is_stdout: bool,
        swallow: bool,
    ) -> None:
        self._original = original
        self._buffer_var = buffer_var
        self._is_stdout = is_stdout
        self._swallow = swallow

    def write(self, s: str) -> int:
        buf = self._buffer_var.get()
        if buf is not None and not buf._disabled:
            target = buf.stdout if self._is_stdout else buf.stderr
            target.write(s)
            if self._swallow:
                return len(s)
        self._original.write(s)
        return len(s)

    def flush(self) -> None:
        # Always flush the original stream because `write()` may have
        # forwarded data to `_original` even when `_swallow` is True
        try:
            self._original.flush()
        except Exception:
            # Best-effort flush; ignore errors from underlying stream
            pass

        # If we're swallowing and a buffer is active, also flush the
        # buffer destination so buffered output is emitted promptly.
        buf = self._buffer_var.get()
        if self._swallow and buf is not None and not buf._disabled:
            target = buf.stdout if self._is_stdout else buf.stderr
            try:
                target.flush()
            except Exception:
                pass

    @property
    def encoding(self) -> str | None:
        return getattr(self._original, "encoding", "utf-8")

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return self._original.isatty()


class SysOutputCapture:
    """Manages sys-level output capture."""

    def __init__(self, swallow: bool = True) -> None:
        self._swallow = swallow
        self._buffer: ContextVar[OutputBuffer | None] = ContextVar("output_buffer", default=None)
        self._original_stdout: TextIO | None = None
        self._original_stderr: TextIO | None = None

    @property
    def original_stdout(self) -> TextIO | None:
        """Access the original stdout stream (bypasses capture)."""
        return self._original_stdout

    @property
    def original_stderr(self) -> TextIO | None:
        """Access the original stderr stream (bypasses capture)."""
        return self._original_stderr

    def install(self) -> None:
        """Replace sys.stdout/stderr with dispatching streams."""
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = _SysStreamDispatcher(
            self._original_stdout, self._buffer, is_stdout=True, swallow=self._swallow
        )
        sys.stderr = _SysStreamDispatcher(
            self._original_stderr, self._buffer, is_stdout=False, swallow=self._swallow
        )

    def uninstall(self) -> None:
        """Restore original sys.stdout/stderr."""
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr

    @contextmanager
    def capture(self) -> Iterator[OutputBuffer]:
        """Activate per-test capture. Yields buffer with captured output."""
        buf = OutputBuffer()
        token: Token[OutputBuffer | None] = self._buffer.set(buf)
        try:
            yield buf
        finally:
            self._buffer.reset(token)


_current_capture: ContextVar[SysOutputCapture | None] = ContextVar("current_capture", default=None)


def get_current_capture() -> SysOutputCapture | None:
    """Get the active capture instance."""
    return _current_capture.get()


@contextmanager
def sys_output_capture(swallow: bool = True) -> Iterator[SysOutputCapture]:
    """Context manager for sys-level output capture.

    Args:
        swallow: If True, output is captured only. If False, output is
                 captured AND passed through to real stdout/stderr.
    """
    capture = SysOutputCapture(swallow=swallow)
    capture.install()
    token: Token[SysOutputCapture | None] = _current_capture.set(capture)
    try:
        yield capture
    finally:
        _current_capture.reset(token)
        capture.uninstall()
