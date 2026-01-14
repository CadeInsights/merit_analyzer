"""Test discovery."""

from merit.testing.discovery.collector import collect
from merit.testing.discovery.loader import MeritModuleLoader
from merit.testing.models import MeritTestDefinition


# Backwards compatibility alias
TestItem = MeritTestDefinition

__all__ = ["MeritModuleLoader", "MeritTestDefinition", "TestItem", "collect"]
