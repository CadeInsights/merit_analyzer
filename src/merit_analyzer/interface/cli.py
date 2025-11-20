from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from ..engines.error_analyzer.driver import ErrorAnalyzer
from ..processors.clustering import cluster_failures
from ..processors.html_formatter import format_analysis_results_html
from ..processors.parse_test_cases import parse_test_cases_from_csv


class CLIApplication:
    """Top-level command router."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.parser = argparse.ArgumentParser(
            prog="merit-analyzer",
            description="Run error analysis on AI test results.",
        )
        subparsers = self.parser.add_subparsers(dest="command", required=True)
        analyze = subparsers.add_parser(
            "analyze",
            help="Cluster failures and generate an HTML report.",
        )
        analyze.add_argument(
            "csv_path",
            help="CSV export containing case_input, reference_value, output_for_assertions, passed, error_message columns.",
        )
        analyze.add_argument(
            "--report",
            dest="report_path",
            default="merit_report.html",
            help="Where to write the HTML report (default: merit_report.html)",
        )
        analyze.add_argument(
            "--model-vendor",
            dest="model_vendor",
            help="MODEL_VENDOR env override (e.g., openai, anthropic).",
        )
        analyze.add_argument(
            "--inference-vendor",
            dest="inference_vendor",
            help="INFERENCE_VENDOR env override (e.g., openai, aws, gcp).",
        )

    def run(self, argv: Sequence[str] | None = None) -> int:
        load_dotenv(Path.cwd() / ".env")
        args = self.parser.parse_args(argv)
        AnalyzeCommand(self.console, args).run()
        return 0


class AnalyzeCommand:
    """Pipeline driver for `merit-analyzer analyze`."""

    def __init__(self, console: Console, args: argparse.Namespace) -> None:
        self.console = console
        self.csv_path = Path(args.csv_path).expanduser()
        self.report_path = Path(args.report_path).expanduser()
        self.model_vendor = args.model_vendor
        self.inference_vendor = args.inference_vendor
        self.analyzer = ErrorAnalyzer()

    def run(self) -> None:
        if self.model_vendor:
            os.environ["MODEL_VENDOR"] = self.model_vendor
        if self.inference_vendor:
            os.environ["INFERENCE_VENDOR"] = self.inference_vendor
        asyncio.run(self.execute())

    async def execute(self) -> None:
        self.console.print("[cyan]Parsing test cases...", end="")
        test_cases = parse_test_cases_from_csv(str(self.csv_path))
        self.console.print(" [green]✓[/green]")

        failed_cases = [case for case in test_cases if not case.assertions_result or not case.assertions_result.passed]
        if not failed_cases:
            format_analysis_results_html([], str(self.report_path), str(self.csv_path))
            report_url = self.report_path.resolve().as_uri()
            self.console.print(
                f"No failing tests found. Blank report generated at [link={report_url}]{report_url}[/link]",
                style="bold green",
            )
            return

        self.console.print("[cyan]Generating error descriptions...", end="")
        for case in failed_cases:
            needs_errors = not case.assertions_result or not case.assertions_result.errors
            if needs_errors:
                await case.generate_error_data()
        self.console.print(" [green]✓[/green]")

        self.console.print("[cyan]Clustering failures...", end="")
        groups = await cluster_failures(failed_cases)
        self.console.print(" [green]✓[/green]")

        self.console.print("[cyan]Running deep analysis per cluster...", style="bold")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("[cyan]Analyzing clusters...", total=len(groups))
            for group in groups:
                group.error_analysis = await self.analyzer.run(group)
                progress.advance(task)

        format_analysis_results_html(groups, str(self.report_path), str(self.csv_path))
        report_url = self.report_path.resolve().as_uri()
        self.console.print(f"Report saved to [link={report_url}]{report_url}[/link]", style="bold green")
