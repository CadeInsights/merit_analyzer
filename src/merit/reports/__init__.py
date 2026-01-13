"""Reporting module for merit test output."""

from merit.reports.base import Reporter
from merit.reports.console import ConsoleReporter


__all__ = ["ConsoleReporter", "Reporter"]
