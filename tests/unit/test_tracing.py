"""Tests for merit.tracing module."""

import json

import pytest

from merit.tracing import clear_traces, get_tracer, init_tracing, trace_step


@pytest.fixture(scope="module")
def trace_output_path(tmp_path_factory):
    """Initialize tracing once for all tests in this module."""
    tmp_dir = tmp_path_factory.mktemp("traces")
    output_path = tmp_dir / "test_traces.jsonl"
    init_tracing(output_path=str(output_path))
    return output_path


@pytest.fixture(autouse=True)
def clear_traces_each():
    """Clear traces before and after each test."""
    clear_traces()
    yield
    clear_traces()


@pytest.mark.usefixtures("trace_output_path")
class TestInitTracing:
    """Tests for init_tracing function."""

    def test_init_tracing_sets_up_provider(self):
        tracer = get_tracer()
        assert tracer is not None

    def test_init_tracing_idempotent(self):
        init_tracing()  # Should not raise
        tracer = get_tracer()
        assert tracer is not None


@pytest.mark.usefixtures("trace_output_path")
class TestTraceStep:
    """Tests for trace_step context manager."""

    def test_trace_step_creates_span(self, trace_output_path):
        with trace_step("test_step"):
            pass

        assert trace_output_path.exists()

        lines = trace_output_path.read_text().strip().split("\n")
        assert len(lines) == 1

        span = json.loads(lines[0])
        assert span["name"] == "test_step"

    def test_trace_step_with_attributes(self, trace_output_path):
        with trace_step("step_with_attrs", {"key": "value", "count": 42}):
            pass

        lines = trace_output_path.read_text().strip().split("\n")
        span = json.loads(lines[0])

        attrs = span["attributes"]
        assert attrs.get("key") == "value"
        assert attrs.get("count") == 42

    def test_nested_trace_steps(self, trace_output_path):
        with trace_step("outer"), trace_step("inner"):
            pass

        lines = trace_output_path.read_text().strip().split("\n")
        assert len(lines) == 2
