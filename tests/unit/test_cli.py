import argparse

import pytest
from rich.console import Console

from merit_analyzer.interface.cli import AnalyzeCommand, CLIApplication
from merit_analyzer.types import (
    AssertionsResult,
    ErrorAnalysis,
    GroupMetadata,
    TestCase,
    TestCaseGroup,
    TestCaseValues,
)
from merit_analyzer.types.code import CodeComponent, ComponentType
from merit_analyzer.types.error import ErrorSolution, FrameInfo


def test_cli_routes_to_analyze(monkeypatch):
    captured = {}

    class FakeAnalyze:
        def __init__(self, console, args):
            captured["console"] = console
            captured["args"] = args

        def run(self):
            captured["ran"] = True

    from merit_analyzer.interface import cli

    monkeypatch.setattr(cli, "AnalyzeCommand", FakeAnalyze)

    app = CLIApplication()
    exit_code = app.run(["analyze", "cases.csv"])

    assert exit_code == 0
    assert captured["ran"] is True
    assert captured["args"].csv_path == "cases.csv"


@pytest.mark.asyncio
async def test_analyze_command_executes_pipeline(monkeypatch):
    from merit_analyzer.interface import cli

    case = TestCase(
        case_data=TestCaseValues(case_input="input", reference_value="expected"),
        output_for_assertions="actual",
        assertions_result=AssertionsResult(False, []),
    )

    async def fake_generate(self):
        self.assertions_result.errors = ["enriched"]

    monkeypatch.setattr(TestCase, "generate_error_data", fake_generate, raising=False)

    recorded = {}

    def fake_parse(path):
        recorded["parsed"] = path
        return [case]

    async def fake_cluster(cases):
        recorded["clustered"] = cases
        return [
            TestCaseGroup(
                metadata=GroupMetadata(name="GROUP_ALPHA", description="desc"),
                test_cases=cases,
            )
        ]

    analysis = ErrorAnalysis(
        involved_code_components=[CodeComponent(name="component", path="src/app.py", type=ComponentType.FILE)],
        traceback=[FrameInfo(index=0, summary="Frame summary")],
        recommendations=[
            ErrorSolution(
                type="code",
                title="Fix bug",
                description="Do something",
                file="src/app.py",
                line_number="10-12",
            )
        ],
    )

    class FakeAnalyzer:
        def __init__(self):
            self.calls = 0

        async def run(self, group):
            self.calls += 1
            return analysis

    fake_analyzer = FakeAnalyzer()

    def fake_formatter(groups, path):
        recorded["report_path"] = path
        recorded["groups"] = groups
        return "report"

    monkeypatch.setattr(cli, "parse_test_cases_from_csv", fake_parse)
    monkeypatch.setattr(cli, "cluster_failures", fake_cluster)
    monkeypatch.setattr(cli, "ErrorAnalyzer", lambda: fake_analyzer)
    monkeypatch.setattr(cli, "format_analysis_results", fake_formatter)

    args = argparse.Namespace(
        csv_path="cases.csv",
        report_path="report.md",
        model_vendor=None,
        inference_vendor=None,
    )

    command = AnalyzeCommand(Console(record=True), args)
    await command.execute()

    assert recorded["parsed"] == "cases.csv"
    assert recorded["clustered"] == [case]
    assert fake_analyzer.calls == 1
    assert recorded["groups"][0].error_analysis == analysis
    assert recorded["report_path"] == "report.md"
    assert case.assertions_result.errors == ["enriched"]  # type: ignore
