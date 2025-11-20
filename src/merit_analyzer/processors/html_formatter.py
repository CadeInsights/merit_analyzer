"""HTML formatter for analysis results."""

import json
import re
import uuid
from pathlib import Path
from typing import Any

from ..types import TestCaseGroup


def format_analysis_results_html(results: list[TestCaseGroup], path: str, source_csv: str | None = None) -> str:
    """Format analysis results into HTML using the embedded template.

    Args:
        results: List of analysis results from CodeAnalyzer
        path: Path where to save the HTML file
        source_csv: Optional path to the source CSV file

    Returns:
        Path to the saved HTML file
    """
    template_path = Path(__file__).parent / "templates" / "report-template.html"

    with open(template_path, encoding="utf-8") as f:
        template = f.read()

    report_data = _convert_to_json_format(results, source_csv)
    json_str = json.dumps(report_data, indent=2)

    # Replace the JSON data in the script tag
    # Use a function-based replacement to avoid regex escape issues with JSON content
    def replace_json(match):
        return f'<script type="application/json" id="report-data">\n{json_str}\n</script>'

    pattern = r'<script type="application/json" id="report-data">.*?</script>'
    html_content = re.sub(pattern, replace_json, template, flags=re.DOTALL)

    # Replace hardcoded source file name (always replace to remove template data)
    source_file_pattern = r'<span id="sourceFile">.*?</span>'
    source_file_replacement = f'<span id="sourceFile">{source_csv or "N/A"}</span>'
    html_content = re.sub(source_file_pattern, source_file_replacement, html_content)

    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return path


def _convert_to_json_format(results: list[TestCaseGroup], source_csv: str | None = None) -> dict[str, Any]:
    """Convert TestCaseGroup list to the JSON format expected by the HTML template."""
    report_data: dict[str, Any] = {"reportId": str(uuid.uuid4()), "clusters": []}

    if source_csv:
        report_data["generatedFrom"] = source_csv

    for idx, group in enumerate(results, 1):
        cluster = {
            "id": str(idx),
            "name": group.metadata.name,
            "problematicBehavior": group.metadata.description,
            "involvedComponents": [],
            "traceback": [],
            "recommendedFixes": [],
            "relatedFailedTests": [],
        }

        if group.error_analysis:
            analysis = group.error_analysis

            cluster["involvedComponents"] = [
                {"name": comp.name, "file": comp.path, "type": f"ComponentType.{comp.type.name}"}
                for comp in analysis.involved_code_components
            ]

            cluster["traceback"] = [
                {"frame": f"Frame #{frame.index}", "description": frame.summary} for frame in analysis.traceback
            ]

            cluster["recommendedFixes"] = [
                {
                    "type": rec.type,
                    "title": rec.title,
                    "description": rec.description,
                    "file": rec.file,
                    "lines": rec.line_number,
                }
                for rec in analysis.recommendations
            ]

        cluster["relatedFailedTests"] = [_convert_test_case_to_dict(case) for case in group.test_cases]

        report_data["clusters"].append(cluster)

    return report_data


def _convert_test_case_to_dict(case: Any) -> dict[str, Any]:
    """Convert TestCase to dictionary format."""
    from dataclasses import asdict

    case_dict = asdict(case)

    if case.assertions_result:
        case_dict["assertions_result"] = {
            "passed": case.assertions_result.passed,
            "errors": case.assertions_result.errors or [],
        }

    return case_dict
