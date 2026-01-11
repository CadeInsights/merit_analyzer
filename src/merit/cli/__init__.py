"""CLI module for Merit test runner."""

from __future__ import annotations

import argparse
import asyncio
import shlex
import sys
from collections.abc import Callable, Sequence

from rich.console import Console

from merit.config import MeritConfig, load_config
from merit.testing.discovery import TestItem, collect
from merit.testing.runner import Runner


def main() -> None:
    """Entry point for merit CLI."""
    config = load_config()
    parser = _build_parser()
    argv = [*config.addopts, *sys.argv[1:]] if config.addopts else sys.argv[1:]
    args = parser.parse_args(argv)

    if args.command != "test":
        parser.print_help()
        raise SystemExit(0)

    exit_code = asyncio.run(_run_tests(args, config))
    raise SystemExit(exit_code)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="merit", description="Merit AI Testing Framework")
    subparsers = parser.add_subparsers(dest="command")

    test_parser = subparsers.add_parser("test", help="Run merit tests")
    test_parser.add_argument("paths", nargs="*", help="Test files or directories")
    test_parser.add_argument("-k", "--keyword", help="Filter tests by keyword expression")
    test_parser.add_argument(
        "-t", "--tag", dest="include_tags", action="append", help="Run tests with given tag"
    )
    test_parser.add_argument(
        "--skip-tag",
        dest="exclude_tags",
        action="append",
        help="Skip tests that match this tag",
    )
    test_parser.add_argument("--maxfail", type=int, help="Stop after this many failures")
    test_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop test after the first failed assertion",
    )
    test_parser.add_argument(
        "--concurrency",
        type=int,
        help="Number of concurrent tests (default: 1, 0 for unlimited up to 10)",
    )
    test_parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable OpenTelemetry tracing of tests and SUT calls",
    )
    test_parser.add_argument(
        "--trace-output",
        type=str,
        default="traces.jsonl",
        help="Output path for trace data (default: traces.jsonl)",
    )
    test_parser.add_argument("-q", "--quiet", action="count", default=0, help="Reduce CLI output")
    test_parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase CLI output"
    )

    return parser


def _resolve_paths(args: argparse.Namespace, config: MeritConfig) -> list[str]:
    if args.paths:
        return args.paths
    return config.test_paths


def _resolve_tags(args: argparse.Namespace, config: MeritConfig) -> tuple[list[str], list[str]]:
    include = list(config.include_tags)
    exclude = list(config.exclude_tags)
    if args.include_tags:
        include.extend(args.include_tags)
    if args.exclude_tags:
        exclude.extend(args.exclude_tags)
    return include, exclude


def _resolve_keyword(args: argparse.Namespace, config: MeritConfig) -> str | None:
    return args.keyword or config.keyword


def _resolve_maxfail(args: argparse.Namespace, config: MeritConfig) -> int | None:
    if args.maxfail is not None:
        return args.maxfail if args.maxfail > 0 else None
    return config.maxfail


def _resolve_verbosity(args: argparse.Namespace, config: MeritConfig) -> int:
    return config.verbosity + args.verbose - args.quiet


def _resolve_concurrency(args: argparse.Namespace, config: MeritConfig) -> int:
    if args.concurrency is not None:
        return max(0, args.concurrency)
    return config.concurrency


def _collect_items(paths: Sequence[str]) -> list[TestItem]:
    items: list[TestItem] = []
    for path in paths:
        items.extend(collect(path))
    return items


def _filter_items(
    items: list[TestItem],
    include_tags: Sequence[str],
    exclude_tags: Sequence[str],
    keyword: str | None,
) -> list[TestItem]:
    filtered = items

    if include_tags:
        include = set(include_tags)
        filtered = [item for item in filtered if item.tags & include]

    if exclude_tags:
        exclude = set(exclude_tags)
        filtered = [item for item in filtered if not (item.tags & exclude)]

    if keyword:
        matcher = KeywordMatcher(keyword)
        filtered = [item for item in filtered if matcher.match(item.full_name)]

    return filtered


async def _run_tests(args: argparse.Namespace, config: MeritConfig) -> int:
    paths = _resolve_paths(args, config)
    include_tags, exclude_tags = _resolve_tags(args, config)
    keyword = _resolve_keyword(args, config)
    maxfail = _resolve_maxfail(args, config)
    verbosity = _resolve_verbosity(args, config)
    concurrency = _resolve_concurrency(args, config)

    items = _collect_items(paths)
    try:
        items = _filter_items(items, include_tags, exclude_tags, keyword)
    except ValueError as exc:
        Console().print(f"[red]{exc}[/red]")
        return 2

    runner = Runner(
        maxfail=maxfail,
        verbosity=verbosity,
        concurrency=concurrency,
        enable_tracing=args.trace,
        trace_output=args.trace_output,
        fail_fast=args.fail_fast,
    )
    merit_run = await runner.run(items=items)

    return 0 if merit_run.result.failed == 0 and merit_run.result.errors == 0 else 1


class KeywordMatcher:
    """Evaluate pytest-style -k expressions."""

    def __init__(self, expression: str) -> None:
        self.tokens = shlex.split(expression)
        self.index = 0
        self.func = self._parse_or()
        if self._peek() is not None:
            msg = "Invalid keyword expression"
            raise ValueError(msg)

    def match(self, text: str) -> bool:
        return self.func(text)

    def _parse_or(self) -> Callable[[str], bool]:
        left = self._parse_and()
        while self._peek_word("or"):
            self._advance()
            right = self._parse_and()
            prev = left
            left = lambda text, prev=prev, right=right: prev(text) or right(text)
        return left

    def _parse_and(self) -> Callable[[str], bool]:
        left = self._parse_not()
        while self._peek_word("and"):
            self._advance()
            right = self._parse_not()
            prev = left
            left = lambda text, prev=prev, right=right: prev(text) and right(text)
        return left

    def _parse_not(self) -> Callable[[str], bool]:
        if self._peek_word("not"):
            self._advance()
            operand = self._parse_not()
            return lambda text, operand=operand: not operand(text)
        return self._parse_term()

    def _parse_term(self) -> Callable[[str], bool]:
        token = self._peek()
        if token is None:
            msg = "Unexpected end of keyword expression"
            raise ValueError(msg)
        if token == "(":
            self._advance()
            expr = self._parse_or()
            if not self._peek_word(")"):
                msg = "Unmatched '(' in keyword expression"
                raise ValueError(msg)
            self._advance()
            return expr
        if token == ")":
            msg = "Unexpected ')' in keyword expression"
            raise ValueError(msg)
        self._advance()
        literal = token
        return lambda text, literal=literal: literal in text

    def _peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _peek_word(self, word: str) -> bool:
        token = self._peek()
        return token is not None and token.lower() == word

    def _advance(self) -> None:
        self.index += 1


__all__ = ["KeywordMatcher", "main"]
