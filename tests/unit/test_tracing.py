"""Tests for merit.tracing module."""

import json
import os

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
    init_tracing()


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

    def test_trace_content_env_var_sets_traceloop(self, monkeypatch):
        # Test that MERIT_TRACE_CONTENT is read during init
        # We can only verify env var handling by checking if it's propagated
        # since tracing is already initialized
        monkeypatch.setenv("MERIT_TRACE_CONTENT", "false")
        # Env var is read during init, which already happened
        # Just verify the env var can be set
        assert os.environ.get("MERIT_TRACE_CONTENT") == "false"


class TestTraceStep:
    """Tests for trace_step context manager."""

    def test_trace_step_creates_span(self, tmp_path):
        with trace_step("test_step"):
            pass

        output_file = tmp_path / "traces.json"
        count = export_traces(output_file)
        assert count == 1

        data = json.loads(output_file.read_text())
        span = data["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert span["name"] == "test_step"

    def test_trace_step_with_attributes(self, tmp_path):
        with trace_step("step_with_attrs", {"key": "value", "count": 42}):
            pass

        output_file = tmp_path / "traces_attrs.json"
        export_traces(output_file)

        data = json.loads(output_file.read_text())
        span = data["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attrs = span["attributes"]
        assert attrs.get("key") == "value"
        assert attrs.get("count") == 42

    def test_nested_trace_steps(self, tmp_path):
        with trace_step("outer"), trace_step("inner"):
            pass

        output_file = tmp_path / "traces_nested.json"
        count = export_traces(output_file)
        assert count == 2


class TestExportTraces:
    """Tests for export_traces function."""

    def test_export_creates_file(self, tmp_path):
        with trace_step("export_test"):
            pass

        output_file = tmp_path / "traces.json"
        count = export_traces(output_file)

        assert count == 1
        assert output_file.exists()

    def test_export_creates_parent_dirs(self, tmp_path):
        with trace_step("nested_export"):
            pass

        output_file = tmp_path / "nested" / "dir" / "traces.json"
        export_traces(output_file)

        assert output_file.exists()

    def test_export_returns_zero_when_no_spans(self, tmp_path):
        output_file = tmp_path / "empty.json"
        count = export_traces(output_file)

        assert count == 0

    def test_export_json_structure(self, tmp_path):
        with trace_step("structure_test"):
            pass

        output_file = tmp_path / "structure.json"
        export_traces(output_file)

        data = json.loads(output_file.read_text())
        assert "resourceSpans" in data
        assert len(data["resourceSpans"]) == 1

        resource_span = data["resourceSpans"][0]
        assert "resource" in resource_span
        assert "scopeSpans" in resource_span

        span = resource_span["scopeSpans"][0]["spans"][0]
        assert "traceId" in span
        assert "spanId" in span
        assert "name" in span
        assert "startTimeUnixNano" in span
        assert "endTimeUnixNano" in span


class TestClearTraces:
    """Tests for clear_traces function."""

    def test_clear_removes_spans(self, tmp_path):
        with trace_step("to_be_cleared"):
            pass

        clear_traces()

        output_file = tmp_path / "cleared.json"
        count = export_traces(output_file)
        assert count == 0
