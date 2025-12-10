"""Tests for merit.tracing module."""

import json

import pytest

from merit.tracing import (
    clear_traces,
    export_traces,
    get_tracer,
    init_tracing,
    trace_step,
)


@pytest.fixture(scope="module", autouse=True)
def setup_tracing_once():
    """Initialize tracing once for all tests in this module."""
    init_tracing(output_path="test_traces.jsonl")


@pytest.fixture(autouse=True)
def clear_traces_each():
    """Clear traces before and after each test."""
    clear_traces()
    yield
    clear_traces()


class TestInitTracing:
    """Tests for init_tracing function."""

    def test_init_tracing_sets_up_provider(self):
        tracer = get_tracer()
        assert tracer is not None

    def test_init_tracing_idempotent(self):
        init_tracing()  # Should not raise
        tracer = get_tracer()
        assert tracer is not None


class TestTraceStep:
    """Tests for trace_step context manager."""

    def test_trace_step_creates_span(self, tmp_path):
        # Re-init with tmp path for this test
        from merit.tracing import _exporter

        if _exporter:
            _exporter.output_path = tmp_path / "traces.jsonl"
            _exporter.output_path.write_text("")

        with trace_step("test_step"):
            pass

        output_file = tmp_path / "traces.jsonl"
        assert output_file.exists()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 1

        span = json.loads(lines[0])
        assert span["name"] == "test_step"

    def test_trace_step_with_attributes(self, tmp_path):
        from merit.tracing import _exporter

        if _exporter:
            _exporter.output_path = tmp_path / "traces_attrs.jsonl"
            _exporter.output_path.write_text("")

        with trace_step("step_with_attrs", {"key": "value", "count": 42}):
            pass

        output_file = tmp_path / "traces_attrs.jsonl"
        lines = output_file.read_text().strip().split("\n")
        span = json.loads(lines[0])

        attrs = span["attributes"]
        assert attrs.get("key") == "value"
        assert attrs.get("count") == 42

    def test_nested_trace_steps(self, tmp_path):
        from merit.tracing import _exporter

        if _exporter:
            _exporter.output_path = tmp_path / "traces_nested.jsonl"
            _exporter.output_path.write_text("")

        with trace_step("outer"), trace_step("inner"):
            pass

        output_file = tmp_path / "traces_nested.jsonl"
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2


class TestExportTraces:
    """Tests for export_traces function."""

    def test_export_is_noop(self, tmp_path):
        # export_traces is now a no-op since we stream
        count = export_traces(tmp_path / "unused.json")
        assert count == 0
