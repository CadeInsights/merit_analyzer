"""Shared types for the Merit testing framework."""

from enum import Enum


class Scope(Enum):
    """Resource/metric lifecycle scope."""

    CASE = "case"  # Fresh instance per test
    SUITE = "suite"  # Shared across tests in same file
    SESSION = "session"  # Shared across entire test run
