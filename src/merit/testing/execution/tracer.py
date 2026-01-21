"""Test tracer - handles tracing for test execution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from opentelemetry.trace import Span, StatusCode

from merit.tracing import get_tracer


if TYPE_CHECKING:
    from merit.testing.models import MeritTestDefinition, TestResult


@dataclass
class TestTracer:
    """Handles tracing spans for test execution."""

    enabled: bool = False

    @contextmanager
    def span(self, definition: MeritTestDefinition) -> Iterator[Span | None]:
        """Context manager for optional tracing."""
        if not self.enabled:
            yield None
            return

        tracer = get_tracer()
        if not tracer:
            yield None
            return

        with tracer.start_as_current_span(f"test.{definition.full_name}") as span:
            span.set_attribute("test.name", definition.name)
            span.set_attribute("test.module", str(definition.module_path))
            if definition.tags:
                span.set_attribute("test.tags", list(definition.tags))
            if definition.id_suffix:
                span.set_attribute("test.id_suffix", definition.id_suffix)
                if self._is_uuid(definition.id_suffix):
                    span.set_attribute("test.case_id", definition.id_suffix)
            yield span

    def _is_uuid(self, value: str) -> bool:
        try:
            UUID(value)
            return True
        except ValueError:
            return False

    def get_trace_id(self, span: Span | None) -> str | None:
        """Extract trace_id from span."""
        if not span:
            return None
        ctx = span.get_span_context()
        return format(ctx.trace_id, "032x") if ctx.trace_id else None

    def record(self, span: Span | None, result: TestResult) -> None:
        """Record span attributes from test result."""
        if not span:
            return
        span.set_attribute("test.status", result.status.value)
        span.set_attribute("test.duration_ms", result.duration_ms)
        if result.error:
            span.set_status(StatusCode.ERROR, str(result.error))
            span.record_exception(result.error)
