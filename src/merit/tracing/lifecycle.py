"""Lifecycle management for OpenTelemetry tracing in Merit.

This module sets up the tracer provider, streaming exporter, instrumentations,
and exposes helper functions for getting a tracer, clearing traces, and
convenience context manager `trace_step`.
"""

# Copied implementation from previous `core.py`. Keep this file focused on
# lifecycle-related responsibilities (initialization, exporter, provider,
# instrumentation, and tracing helpers).

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from merit.tracing.exporters import StreamingFileSpanExporter
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor


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


def set_trace_output_path(output_path: Path | str) -> None:
    """Set the output path for the current exporter.

    Useful for testing to redirect traces to a temporary file.
    """
    global _exporter
    if _exporter is None:
        # If not initialized, initialize it
        init_tracing(output_path=output_path)
    else:
        _exporter.output_path = Path(output_path)
        _exporter.output_path.parent.mkdir(parents=True, exist_ok=True)
        _exporter.output_path.write_text("")


def _instrument_llm_clients() -> None:
    """Instrument OpenAI and Anthropic clients."""
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
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span
