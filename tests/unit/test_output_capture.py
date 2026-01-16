"""Tests for sys-level output capture."""

import sys

from merit.context.output_capture import (
    OutputBuffer,
    SysOutputCapture,
    get_current_capture,
    sys_output_capture,
)


class TestOutputBuffer:
    def test_readouterr_returns_captured_output(self):
        buf = OutputBuffer()
        buf.stdout.write("hello")
        buf.stderr.write("error")

        out, err = buf.readouterr()

        assert out == "hello"
        assert err == "error"

    def test_readouterr_clears_buffer(self):
        buf = OutputBuffer()
        buf.stdout.write("hello")
        buf.readouterr()

        out, err = buf.readouterr()

        assert out == ""
        assert err == ""

    def test_disabled_context_manager(self):
        buf = OutputBuffer()

        assert not buf._disabled
        with buf.disabled():
            assert buf._disabled
        assert not buf._disabled


class TestSysOutputCapture:
    def test_install_replaces_sys_streams(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        capture = SysOutputCapture(swallow=True)

        capture.install()

        assert sys.stdout is not original_stdout
        assert sys.stderr is not original_stderr

        capture.uninstall()

        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr

    def test_capture_context_manager_provides_buffer(self):
        capture = SysOutputCapture(swallow=True)
        capture.install()

        with capture.capture() as buf:
            print("hello")
            sys.stderr.write("error\n")

        capture.uninstall()

        out, err = buf.readouterr()
        assert out == "hello\n"
        assert err == "error\n"

    def test_swallow_true_hides_output_when_buffer_active(self, capsys):
        capture = SysOutputCapture(swallow=True)
        capture.install()

        with capture.capture():
            print("hidden")

        capture.uninstall()

        # Output should not appear in real stdout when buffer is active
        real_out, _ = capsys.readouterr()
        assert "hidden" not in real_out

    def test_swallow_true_passes_through_when_no_buffer(self, capsys):
        capture = SysOutputCapture(swallow=True)
        capture.install()

        # No buffer active - output should pass through
        print("visible")

        capture.uninstall()

        real_out, _ = capsys.readouterr()
        assert "visible" in real_out

    def test_disabled_bypasses_capture(self, capsys):
        capture = SysOutputCapture(swallow=True)
        capture.install()

        with capture.capture() as buf:
            print("captured")
            with buf.disabled():
                print("bypassed")
            print("captured again")

        capture.uninstall()

        out, _ = buf.readouterr()
        real_out, _ = capsys.readouterr()

        assert "captured" in out
        assert "captured again" in out
        assert "bypassed" not in out
        assert "bypassed" in real_out

    def test_swallow_false_shows_output(self, capsys):
        capture = SysOutputCapture(swallow=False)
        capture.install()

        with capture.capture() as buf:
            print("visible")

        capture.uninstall()

        # Output should appear in both buffer and real stdout
        out, _ = buf.readouterr()
        assert out == "visible\n"
        real_out, _ = capsys.readouterr()
        assert real_out == "visible\n"


class TestSysOutputCaptureContextManager:
    def test_sets_current_capture(self):
        assert get_current_capture() is None

        with sys_output_capture(swallow=True) as capture:
            assert get_current_capture() is capture

        assert get_current_capture() is None

    def test_nested_captures_isolate_output(self):
        with sys_output_capture(swallow=True) as outer_capture:
            with outer_capture.capture() as buf1:
                print("outer")

            with outer_capture.capture() as buf2:
                print("inner")

        out1, _ = buf1.readouterr()
        out2, _ = buf2.readouterr()

        assert out1 == "outer\n"
        assert out2 == "inner\n"
