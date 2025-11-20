"""Markdown formatter for analysis results."""

from typing import List
from dataclasses import asdict

from ..types import ErrorAnalysis, TestCaseGroup


def format_analysis_results(results: List[TestCaseGroup], path: str) -> str:
    """
    Format analysis results into markdown.

    Structure: error group name > problematic behavior > problematic code > relevant test results

    Args:
        results: List of analysis results from CodeAnalyzer

    Returns:
        Markdown formatted string
    """
    markdown_sections = []

    # Title
    markdown_sections.append("# Test Failure Analysis Report\n")
    markdown_sections.append(f"**Total Error Groups:** {len(results)}\n")
    markdown_sections.append("---\n")

    # Each error group
    for idx, result in enumerate(results, 1):
        section = _format_error_group(idx, result)
        markdown_sections.append(section)

    report = "\n".join(markdown_sections)

    with open(path, "w") as f:
        f.write(report)

    return report


def _format_error_group(idx: int, result: TestCaseGroup) -> str:
    """Format a single error group section."""
    lines = []

    # Header
    lines.append(f"## {idx}. {result.metadata.name}\n")

    # Problematic Behavior
    lines.append("### Problematic Behavior\n")
    lines.append(f"{result.metadata.description}\n")

    if result.error_analysis is not None:
        analysis = result.error_analysis

        # Components
        lines.append("### Involved Components\n")

        for component in analysis.involved_code_components:
            lines.append(f"```\n{component.name} | {component.path} | {component.type}\n```\n")

        # Traceback
        lines.append("### Traceback\n")

        for frame in analysis.traceback:
            lines.append(f"```\nFrame #{frame.index} | Frame: {frame.summary} \n```\n")

        # Recommendations
        lines.append("### Recommended Fixes\n")
        for rec_idx, rec in enumerate(analysis.recommendations, 1):
            lines.append(f"#### {rec_idx}. {rec.type} | {rec.title}\n")
            lines.append(f"\nDescription: {rec.description}\n")
            lines.append(f"\nFile: {rec.file}\n")
            lines.append(f"\nLines: {rec.line_number}\n")
            lines.append("")  # Empty line between recommendations

    # Relevant Test Results
    lines.append("### Related Failed Tests\n")
    for case in result.test_cases:
        lines.append(f"- {str(asdict(case))}\n")

    lines.append("---\n")

    return "\n".join(lines)
