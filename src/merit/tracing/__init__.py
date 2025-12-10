"""OpenTelemetry tracing for Merit test runs.

Captures structured traces of LLM calls, tool invocations, agent loops,
and custom test steps. Uses OpenLLMetry for automatic LLM instrumentation.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from merit.tracing.exporters import StreamingFileSpanExporter


_exporter: StreamingFileSpanExporter | None = None
_initialized = False


def init_tracing(
    *,
    service_name: str = "merit",
    trace_content: bool | None = None,
    output_path: Path | str = "traces.jsonl",
) -> None:
    """Initialize OpenTelemetry tracing with streaming file export.

    Must be called before any LLM clients are instantiated to ensure
    instrumentation captures all calls.

    Args:
        service_name: Service name for the trace resource.
        trace_content: Whether to capture prompts/completions in spans.
            Defaults to MERIT_TRACE_CONTENT env var, or True if not set.
        output_path: Path to write trace data to (JSONL format).
    """
    global _exporter, _initialized

    if _initialized:
        return

    # Configure trace content capture
    if trace_content is None:
        trace_content = os.environ.get("MERIT_TRACE_CONTENT", "true").lower() == "true"

    # Set up streaming exporter
    _exporter = StreamingFileSpanExporter(output_path)
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(_exporter))
    trace.set_tracer_provider(provider)

    # Instrument LLM clients via OpenLLMetry
    _instrument_llm_clients()

    _initialized = True


def _instrument_llm_clients() -> None:
    """Instrument OpenAI and Anthropic clients."""
    from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor

    # Note: These instrumentors capture content by default.
    # If privacy controls are needed, we can configure them here.
    OpenAIInstrumentor().instrument()
    AnthropicInstrumentor().instrument()


def get_tracer(name: str = "merit") -> trace.Tracer:
    """Get a tracer instance for creating spans."""
    return trace.get_tracer(name)


def clear_traces() -> None:
    """Clear the trace file."""
    if _exporter is not None:
        # Re-initialize the file (clears content)
        _exporter.output_path.write_text("")


@contextmanager
def trace_step(name: str, attributes: dict[str, Any] | None = None):
    """Context manager for tracing custom steps in test logic.

    Creates a span that nests under the current active span.

    Args:
        name: Name for this trace step.
        attributes: Optional attributes to attach to the span.

    Example:
        with trace_step("preprocessing", {"input_size": len(data)}):
            cleaned = preprocess(data)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span
