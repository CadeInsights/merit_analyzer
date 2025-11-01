"""Analysis report data models."""

from pydantic import BaseModel, Field  # type: ignore
from typing import List, Dict, Any, Optional
from datetime import datetime
from .test_result import TestResult
from .pattern import Pattern
from .recommendation import Recommendation


class ReportSummary(BaseModel):
    """High-level summary statistics."""

    total_tests: int
    passed: int
    failed: int
    error: int = 0
    skipped: int = 0
    pass_rate: float = Field(default=0.0)
    patterns_found: int
    analysis_timestamp: str
    analysis_duration_seconds: Optional[float] = None
    project_path: Optional[str] = None
    frameworks_detected: List[str] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        if self.total_tests > 0:
            self.pass_rate = self.passed / self.total_tests

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_tests == 0:
            return 0.0
        return self.failed / self.total_tests

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_tests == 0:
            return 0.0
        return self.error / self.total_tests


class PatternSummary(BaseModel):
    """Summary of a failure pattern."""

    name: str
    description: Optional[str] = None
    failure_count: int
    failure_rate: float
    example_tests: List[TestResult]
    recommendations: List[Recommendation]
    root_cause: Optional[str] = None
    affected_components: Optional[List[str]] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    keywords: List[str] = Field(default_factory=list)

    @property
    def has_recommendations(self) -> bool:
        """Check if pattern has recommendations."""
        return len(self.recommendations) > 0

    @property
    def high_priority_recommendations(self) -> List[Recommendation]:
        """Get high priority recommendations."""
        return [r for r in self.recommendations if r.priority.value == "high"]

    def get_recommendations_by_type(self, rec_type: str) -> List[Recommendation]:
        """Get recommendations by type."""
        return [r for r in self.recommendations if r.type.value == rec_type]


class AnalysisReport(BaseModel):
    """Complete analysis report."""

    summary: ReportSummary
    patterns: Dict[str, PatternSummary]
    action_plan: List[str]
    architecture: Dict[str, Any]
    recommendations: List[Recommendation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    executive_summary: Optional[Dict[str, Any]] = Field(default=None, description="Executive summary with top fixes")
    consolidated_recommendations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Consolidated recommendations with impact scores")

    @property
    def total_recommendations(self) -> int:
        """Total number of recommendations."""
        return len(self.recommendations)

    @property
    def high_priority_count(self) -> int:
        """Number of high priority recommendations."""
        return sum(1 for r in self.recommendations if r.priority.value == "high")

    def get_recommendations_by_priority(self) -> Dict[str, List[Recommendation]]:
        """Group recommendations by priority."""
        grouped = {"high": [], "medium": [], "low": []}
        for rec in self.recommendations:
            grouped[rec.priority.value].append(rec)
        return grouped

    def get_recommendations_by_type(self) -> Dict[str, List[Recommendation]]:
        """Group recommendations by type."""
        grouped = {}
        for rec in self.recommendations:
            rec_type = rec.type.value
            if rec_type not in grouped:
                grouped[rec_type] = []
            grouped[rec_type].append(rec)
        return grouped

    def display(self) -> None:
        """Pretty print the report using Rich."""
        from rich.console import Console  # type: ignore
        from rich.table import Table  # type: ignore
        from rich.panel import Panel  # type: ignore
        from rich.text import Text  # type: ignore

        console = Console()

        # Summary panel
        summary_text = Text()
        summary_text.append(f"Tests: {self.summary.total_tests} total, ", style="white")
        summary_text.append(f"{self.summary.passed} passed", style="green")
        summary_text.append(", ", style="white")
        summary_text.append(f"{self.summary.failed} failed", style="red")
        summary_text.append(f" ({self.summary.pass_rate:.1%} pass rate)\n", style="white")
        summary_text.append(f"Patterns Found: {self.summary.patterns_found}\n", style="white")
        summary_text.append(f"Recommendations: {self.total_recommendations}", style="blue")

        console.print(Panel.fit(summary_text, title="Merit Analyzer Report", border_style="blue"))

        # Patterns table
        if self.patterns:
            console.print("\n[bold]Failure Patterns:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Pattern", style="cyan", width=30)
            table.add_column("Failures", justify="right", style="red")
            table.add_column("Rate", justify="right", style="yellow")
            table.add_column("Recommendations", justify="right", style="green")

            for pattern_name, pattern in self.patterns.items():
                table.add_row(
                    pattern_name,
                    str(pattern.failure_count),
                    f"{pattern.failure_rate:.1%}",
                    str(len(pattern.recommendations)),
                )

            console.print(table)

        # Action plan
        if self.action_plan:
            console.print("\n[bold green]Prioritized Action Plan:[/bold green]")
            for i, action in enumerate(self.action_plan, 1):
                console.print(f"  {i}. {action}")

        # Timestamp
        console.print(f"\n[dim]Report generated: {self.summary.analysis_timestamp}[/dim]")

    def to_markdown(self) -> str:
        """Export report as Markdown."""
        md = f"""# Merit Analyzer Report

## Summary

- **Total Tests**: {self.summary.total_tests}
- **Passed**: {self.summary.passed}
- **Failed**: {self.summary.failed}
- **Error**: {self.summary.error}
- **Skipped**: {self.summary.skipped}
- **Pass Rate**: {self.summary.pass_rate:.1%}
- **Patterns Found**: {self.summary.patterns_found}
- **Recommendations**: {self.total_recommendations}
- **Analysis Date**: {self.summary.analysis_timestamp}

"""

        if self.summary.frameworks_detected:
            md += f"**Detected Frameworks**: {', '.join(self.summary.frameworks_detected)}\n\n"

        if self.summary.analysis_duration_seconds:
            md += f"**Analysis Duration**: {self.summary.analysis_duration_seconds:.1f} seconds\n\n"
        
        # Add executive summary if available
        if self.executive_summary:
            from ..recommendations.executive_summary import ExecutiveSummaryGenerator
            generator = ExecutiveSummaryGenerator()
            md += generator.format_executive_summary_markdown(self.executive_summary)
            md += "\n---\n\n"

        # Patterns section
        if self.patterns:
            md += "## Failure Patterns\n\n"
            for pattern_name, pattern in self.patterns.items():
                md += f"""### {pattern_name}

- **Failure Count**: {pattern.failure_count}
- **Failure Rate**: {pattern.failure_rate:.1%}
- **Recommendations**: {len(pattern.recommendations)}

"""

                if pattern.description:
                    md += f"**Description**: {pattern.description}\n\n"

                if pattern.root_cause:
                    md += f"**Root Cause**: {pattern.root_cause}\n\n"

                if pattern.affected_components:
                    md += f"**Affected Components**: {', '.join(pattern.affected_components)}\n\n"

                # Recommendations for this pattern
                if pattern.recommendations:
                    md += "#### Recommendations\n\n"
                    for i, rec in enumerate(pattern.recommendations, 1):
                        md += rec.to_markdown()
                        if i < len(pattern.recommendations):
                            md += "\n---\n\n"

        # Overall recommendations
        if self.recommendations:
            md += "## All Recommendations\n\n"
            for i, rec in enumerate(self.recommendations, 1):
                md += f"### {i}. {rec.title}\n\n"
                md += rec.to_markdown()
                if i < len(self.recommendations):
                    md += "\n---\n\n"

        # Action plan
        if self.action_plan:
            md += "## Action Plan\n\n"
            for action in self.action_plan:
                md += f"- {action}\n"

        return md

    def to_json(self) -> str:
        """Export report as JSON."""
        import json
        return json.dumps(self.model_dump(exclude_none=True), indent=2, default=str)

    def save(self, filepath: str, format: str = "json") -> None:
        """Save report to file."""
        if format.lower() == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.to_json())
        elif format.lower() == "markdown":
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.to_markdown())
        else:
            raise ValueError(f"Unsupported format: {format}")
