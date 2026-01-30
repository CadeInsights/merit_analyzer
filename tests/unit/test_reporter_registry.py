"""Tests for merit.reports.registry module."""

from pathlib import Path

import pytest

from merit.reports.base import Reporter
from merit.reports.registry import (
    clear_reporter_registry,
    get_reporter_registry,
    register_builtin,
    reporter,
    resolve_reporter,
    resolve_reporters,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the reporter registry before and after each test."""
    clear_reporter_registry()
    yield
    clear_reporter_registry()


class DummyReporter(Reporter):
    """Minimal reporter for testing."""

    def __init__(self, verbosity: int = 0):
        self.verbosity = verbosity

    async def on_no_tests_found(self) -> None:
        pass

    async def on_collection_complete(self, items) -> None:
        pass

    async def on_test_complete(self, execution) -> None:
        pass

    async def on_run_complete(self, merit_run) -> None:
        pass

    async def on_run_stopped_early(self, failure_count: int) -> None:
        pass

    async def on_tracing_enabled(self, output_path: Path) -> None:
        pass


class TestReporterDecorator:
    """Tests for the @reporter decorator."""

    def test_registers_class(self):
        @reporter
        class MyReporter(DummyReporter):
            pass

        registry = get_reporter_registry()
        assert "MyReporter" in registry
        assert registry["MyReporter"] is MyReporter

    def test_registers_with_custom_name(self):
        @reporter(name="custom")
        class MyReporter(DummyReporter):
            pass

        registry = get_reporter_registry()
        assert "custom" in registry
        assert "MyReporter" not in registry
        assert registry["custom"] is MyReporter

    def test_disabled_registration(self):
        @reporter(enabled=False)
        class DisabledReporter(DummyReporter):
            pass

        registry = get_reporter_registry()
        assert "DisabledReporter" not in registry

    def test_returns_original_class(self):
        @reporter
        class MyReporter(DummyReporter):
            pass

        assert MyReporter.__name__ == "MyReporter"
        instance = MyReporter()
        assert isinstance(instance, DummyReporter)


class TestResolveReporter:
    """Tests for resolve_reporter function."""

    def test_resolve_from_registry(self):
        @reporter
        class TestReporter(DummyReporter):
            pass

        instance = resolve_reporter("TestReporter")
        assert isinstance(instance, TestReporter)

    def test_resolve_with_kwargs(self):
        @reporter
        class ConfigurableReporter(DummyReporter):
            def __init__(self, verbosity: int = 0):
                self.verbosity = verbosity

        instance = resolve_reporter("ConfigurableReporter", verbosity=2)
        assert instance.verbosity == 2

    def test_resolve_import_string_colon(self):
        instance = resolve_reporter("merit.reports.console:ConsoleReporter")
        from merit.reports.console import ConsoleReporter

        assert isinstance(instance, ConsoleReporter)

    def test_resolve_import_string_dot(self):
        instance = resolve_reporter("merit.reports.console.ConsoleReporter")
        from merit.reports.console import ConsoleReporter

        assert isinstance(instance, ConsoleReporter)

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown reporter"):
            resolve_reporter("NonExistent")

    def test_resolve_invalid_import_raises(self):
        with pytest.raises((ModuleNotFoundError, ValueError)):
            resolve_reporter("nonexistent.module:Reporter")


class TestResolveReporters:
    """Tests for resolve_reporters function."""

    def test_resolve_multiple(self):
        @reporter
        class ReporterA(DummyReporter):
            pass

        @reporter
        class ReporterB(DummyReporter):
            pass

        instances = resolve_reporters(["ReporterA", "ReporterB"])
        assert len(instances) == 2
        assert isinstance(instances[0], ReporterA)
        assert isinstance(instances[1], ReporterB)

    def test_resolve_with_options(self):
        @reporter
        class OptionsReporter(DummyReporter):
            def __init__(self, verbosity: int = 0):
                self.verbosity = verbosity

        instances = resolve_reporters(
            ["OptionsReporter"],
            options={"OptionsReporter": {"verbosity": 3}},
        )
        assert instances[0].verbosity == 3

    def test_resolve_empty_list(self):
        instances = resolve_reporters([])
        assert instances == []


class TestBuiltinRegistration:
    """Tests for register_builtin decorator."""

    def test_builtin_persists_after_clear(self):
        @register_builtin
        class BuiltinReporter(DummyReporter):
            pass

        assert "BuiltinReporter" in get_reporter_registry()
        clear_reporter_registry()
        assert "BuiltinReporter" in get_reporter_registry()

    def test_console_reporter_is_builtin(self):
        clear_reporter_registry()
        assert "ConsoleReporter" in get_reporter_registry()
