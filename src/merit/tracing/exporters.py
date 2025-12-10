"""Streaming file exporter for OpenTelemetry spans.

Writes spans to a JSONL file as they are finished, avoiding memory buildup.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class StreamingFileSpanExporter(SpanExporter):
    """Exports spans to a file in JSONL format as they are received."""

    def __init__(self, output_path: Path | str) -> None:
        self.output_path = Path(output_path)
        # Ensure directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear file if it exists
        self.output_path.write_text("")

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export a batch of spans to the file."""
        try:
            with self.output_path.open("a", encoding="utf-8") as f:
                for span in spans:
                    data = self._span_to_dict(span)
                    f.write(json.dumps(data, default=str) + "\n")
            return SpanExportResult.SUCCESS
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error exporting spans to file: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Cleanup resources."""

    def _span_to_dict(self, span: ReadableSpan) -> dict[str, Any]:
        """Convert a ReadableSpan to a JSON-serializable dict."""
        return {
            "traceId": format(span.context.trace_id, "032x"),
            "spanId": format(span.context.span_id, "016x"),
            "parentSpanId": format(span.parent.span_id, "016x") if span.parent else None,
            "name": span.name,
            "kind": span.kind.name if span.kind else "INTERNAL",
            "startTimeUnixNano": span.start_time,
            "endTimeUnixNano": span.end_time,
            "attributes": self._attrs_to_dict(span.attributes or {}),
            "status": {"code": span.status.status_code.name if span.status else "UNSET"},
            "events": [
                {
                    "name": e.name,
                    "timeUnixNano": e.timestamp,
                    "attributes": self._attrs_to_dict(e.attributes or {}),
                }
                for e in (span.events or [])
            ],
            "resource": {"attributes": self._attrs_to_dict(span.resource.attributes)},
            "scope": {"name": span.instrumentation_scope.name if span.instrumentation_scope else ""},
        }

    def _attrs_to_dict(self, attrs: Any) -> dict[str, Any]:
        """Convert span attributes to a plain dict."""
        if hasattr(attrs, "items"):
            return {k: v for k, v in attrs.items()}
        return dict(attrs) if attrs else {}
