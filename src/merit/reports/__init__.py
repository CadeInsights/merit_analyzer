"""Reporting module for merit test output."""

from merit.reports.base import Reporter
from merit.reports.console import ConsoleReporter
from merit.reports.registry import (
    _builtin_registry,
    _reporter_registry,
    get_reporter_registry,
    reporter,
    resolve_reporter,
    resolve_reporters,
)


_reporter_registry["ConsoleReporter"] = ConsoleReporter
_builtin_registry["ConsoleReporter"] = ConsoleReporter

__all__ = [
    "ConsoleReporter",
    "Reporter",
    "get_reporter_registry",
    "reporter",
    "resolve_reporter",
    "resolve_reporters",
]
