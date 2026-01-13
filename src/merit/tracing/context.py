"""Trace context for accessing span data during test execution."""

from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import Span

from merit.tracing.lifecycle import InMemorySpanCollector


@dataclass
class TraceContext:
    """Provides access to trace data for the current test.

    Injected via dependency injection when a test declares `trace_context`
    as a parameter. Allows querying child spans, LLM calls, and setting
    custom attributes on the test span.
    """

    _trace_id: str
    _span: Span
    _collector: InMemorySpanCollector | None

    @classmethod
    def from_current(cls, collector: InMemorySpanCollector | None) -> "TraceContext":
        """Create TraceContext from the currently active span."""
        span = trace.get_current_span()
        trace_id = format(span.get_span_context().trace_id, "032x")
        return cls(_trace_id=trace_id, _span=span, _collector=collector)

    @property
    def trace_id(self) -> str:
        """The trace ID for this test's span."""
        return self._trace_id

    @property
    def span_id(self) -> str:
        """The span ID for this test's span."""
        return format(self._span.get_span_context().span_id, "016x")

    @property
    def is_enabled(self) -> bool:
        """Whether tracing is enabled."""
        return self._collector is not None

    def get_child_spans(self) -> list[ReadableSpan]:
        """Get all spans created during this test's execution.

        Returns spans that share this test's trace_id, including
        SUT calls, LLM calls, and custom trace steps.
        """
        if not self._collector:
            return []
        return self._collector.get_spans(self._trace_id)

    def get_llm_calls(self) -> list[ReadableSpan]:
        """Get spans from LLM API calls (OpenAI, Anthropic).

        Filters child spans by known LLM instrumentation prefixes.
        """
        return [
            s for s in self.get_child_spans()
            if s.name.startswith(("openai.", "anthropic.", "gen_ai."))
        ]

    def get_sut_spans(self, name: str | None = None) -> list[ReadableSpan]:
        """Get spans from @merit.sut decorated functions.

        Args:
            name: Optional SUT name to filter by (case-sensitive).
        """
        spans = [
            s for s in self.get_child_spans()
            if s.attributes and s.attributes.get("merit.sut")
        ]
        if name:
            spans = [s for s in spans if s.attributes.get("merit.sut.name") == name]
        return spans

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a custom attribute on the test span."""
        self._span.set_attribute(key, value)
