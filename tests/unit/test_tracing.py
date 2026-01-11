"""Tests for merit.tracing module."""

import json

import pytest
from opentelemetry import trace

from merit.testing.resources import ResourceResolver, get_registry
from merit.tracing import (
    TraceContext,
    clear_traces,
    get_tracer,
    init_tracing,
    set_trace_output_path,
    trace_step,
)


@pytest.fixture(scope="module")
def trace_output_path(tmp_path_factory):
    """Initialize tracing once for all tests in this module."""
    tmp_dir = tmp_path_factory.mktemp("traces")
    output_path = tmp_dir / "test_traces.jsonl"
    set_trace_output_path(output_path=str(output_path))
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


@pytest.mark.usefixtures("trace_output_path")
class TestTraceContext:
    """Tests for TraceContext resource and object."""

    @pytest.mark.asyncio
    async def test_trace_context_injection(self):
        """Test that trace_context resolves correctly within an active span."""
        tracer = get_tracer()

        # Start a span to simulate a running test
        with tracer.start_as_current_span("test_root_span") as span:
            resolver = ResourceResolver(get_registry())

            # Resolve the trace_context resource
            # Note: The resource is a generator, resolver handles the lifecycle
            ctx = await resolver.resolve("trace_context")

            assert isinstance(ctx, TraceContext)
            assert ctx.is_enabled is True

            # Verify IDs match the current span
            assert ctx.trace_id == format(span.get_span_context().trace_id, "032x")
            assert ctx.span_id == format(span.get_span_context().span_id, "016x")

            # Create a child span
            with trace_step("child_step"):
                pass

            # Verify child span capture
            child_spans = ctx.get_child_spans()
            assert len(child_spans) >= 1
            assert any(s.name == "child_step" for s in child_spans)

            # Test custom attributes
            ctx.set_attribute("my.custom.attr", "value")

            # Teardown the resolver (important for generators)
            await resolver.teardown()

    @pytest.mark.asyncio
    async def test_trace_context_lifecycle(self):
        """Test that trace_context clears spans on teardown."""
        tracer = get_tracer()

        with tracer.start_as_current_span("lifecycle_test") as span:
            resolver = ResourceResolver(get_registry())
            ctx = await resolver.resolve("trace_context")
            trace_id = ctx.trace_id

            with trace_step("step_1"): pass

            assert len(ctx.get_child_spans()) > 0

            # Trigger teardown
            # This should invoke trace_context generator teardown which calls collector.clear(trace_id)
            await resolver.teardown()

            # Verify spans are cleared from collector for this trace
            from merit.tracing import get_span_collector
            collector = get_span_collector()
            assert len(collector.get_spans(trace_id)) == 0

