"""Streaming file exporter for OpenTelemetry spans.

Writes spans to a JSONL file as they are finished, avoiding memory buildup.
"""

from collections.abc import Sequence
from pathlib import Path

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
                    data = span.to_json(indent=None)
                    f.write(data + "\n")
            return SpanExportResult.SUCCESS
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error exporting spans to file: {e}")
            return SpanExportResult.FAILURE
