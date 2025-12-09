"""OpenTelemetry tracing for Merit test runs.

Captures structured traces of LLM calls, tool invocations, agent loops,
and custom test steps. Uses OpenLLMetry for automatic LLM instrumentation.
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


_exporter: InMemorySpanExporter | None = None
_initialized = False


def init_tracing(*, service_name: str = "merit", trace_content: bool | None = None) -> None:
    """Initialize OpenTelemetry tracing with in-memory export.

    Must be called before any LLM clients are instantiated to ensure
    instrumentation captures all calls.

    Args:
        service_name: Service name for the trace resource.
        trace_content: Whether to capture prompts/completions in spans.
            Defaults to MERIT_TRACE_CONTENT env var, or True if not set.
    """
    global _exporter, _initialized

    if _initialized:
        return

    # Configure trace content capture
    if trace_content is None:
        trace_content = os.environ.get("MERIT_TRACE_CONTENT", "true").lower() == "true"

    os.environ["TRACELOOP_TRACE_CONTENT"] = str(trace_content).lower()

    # Set up in-memory exporter
    _exporter = InMemorySpanExporter()
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

    OpenAIInstrumentor().instrument()
    AnthropicInstrumentor().instrument()


def get_tracer(name: str = "merit") -> trace.Tracer:
    """Get a tracer instance for creating spans."""
    return trace.get_tracer(name)


def export_traces(path: Path | str) -> int:
    """Export collected spans to a JSON file.

    Args:
        path: Output file path for the trace data.

    Returns:
        Number of spans exported.
    """
    if _exporter is None:
        return 0

    spans = _exporter.get_finished_spans()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    trace_data = {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs_to_dict(span.resource.attributes)},
                "scopeSpans": [
                    {
                        "scope": {"name": span.instrumentation_scope.name if span.instrumentation_scope else ""},
                        "spans": [_span_to_dict(span)],
                    }
                ],
            }
            for span in spans
        ]
    }

    path.write_text(json.dumps(trace_data, indent=2, default=str))
    return len(spans)


def clear_traces() -> None:
    """Clear all collected spans from the in-memory exporter."""
    if _exporter is not None:
        _exporter.clear()


def _span_to_dict(span: Any) -> dict[str, Any]:
    """Convert a ReadableSpan to a JSON-serializable dict."""
    return {
        "traceId": format(span.context.trace_id, "032x"),
        "spanId": format(span.context.span_id, "016x"),
        "parentSpanId": format(span.parent.span_id, "016x") if span.parent else None,
        "name": span.name,
        "kind": span.kind.name if span.kind else "INTERNAL",
        "startTimeUnixNano": span.start_time,
        "endTimeUnixNano": span.end_time,
        "attributes": _attrs_to_dict(span.attributes or {}),
        "status": {"code": span.status.status_code.name if span.status else "UNSET"},
        "events": [
            {
                "name": e.name,
                "timeUnixNano": e.timestamp,
                "attributes": _attrs_to_dict(e.attributes or {}),
            }
            for e in (span.events or [])
        ],
    }


def _attrs_to_dict(attrs: Any) -> dict[str, Any]:
    """Convert span attributes to a plain dict."""
    if hasattr(attrs, "items"):
        return {k: v for k, v in attrs.items()}
    return dict(attrs) if attrs else {}


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
