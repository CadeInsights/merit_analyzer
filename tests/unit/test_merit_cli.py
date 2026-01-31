from argparse import Namespace
from pathlib import Path

from merit.cli import KeywordMatcher, _filter_items, _resolve_reporters
from merit.config import DEFAULT_CONFIG, MeritConfig
from merit.reports import ConsoleReporter
from merit.testing.discovery import TestItem


def dummy() -> None:  # Helper for TestItem.fn
    return None


def make_item(name: str, tags: set[str], id_suffix: str | None = None) -> TestItem:
    return TestItem(
        name=name,
        fn=dummy,
        module_path=Path("module.py"),
        is_async=False,
        params=[],
        tags=tags,
        id_suffix=id_suffix,
    )


def test_keyword_matcher_supports_boolean_logic():
    matcher = KeywordMatcher("foo and not bar")
    assert matcher.match("foo_case")
    assert not matcher.match("bar_case")
    assert not matcher.match("other")


def test_filter_items_applies_tag_logic():
    items = [
        make_item("merit_fast", {"fast", "smoke"}),
        make_item("merit_slow", {"slow"}),
    ]

    filtered = _filter_items(items, include_tags=["smoke"], exclude_tags=[], keyword=None)
    assert [item.name for item in filtered] == ["merit_fast"]

    filtered = _filter_items(items, include_tags=[], exclude_tags=["slow"], keyword=None)
    assert [item.name for item in filtered] == ["merit_fast"]

    filtered = _filter_items(items, include_tags=[], exclude_tags=[], keyword="slow")
    assert [item.name for item in filtered] == ["merit_slow"]


class TestResolveReporters:
    """Tests for _resolve_reporters function."""

    def _make_args(self, **kwargs) -> Namespace:
        defaults = {"reporters": None}
        defaults.update(kwargs)
        return Namespace(**defaults)

    def _make_config(self, **kwargs) -> MeritConfig:
        return MeritConfig(
            test_paths=list(DEFAULT_CONFIG.test_paths),
            include_tags=list(DEFAULT_CONFIG.include_tags),
            exclude_tags=list(DEFAULT_CONFIG.exclude_tags),
            keyword=DEFAULT_CONFIG.keyword,
            maxfail=DEFAULT_CONFIG.maxfail,
            verbosity=DEFAULT_CONFIG.verbosity,
            addopts=list(DEFAULT_CONFIG.addopts),
            concurrency=DEFAULT_CONFIG.concurrency,
            timeout=DEFAULT_CONFIG.timeout,
            db_path=DEFAULT_CONFIG.db_path,
            save_to_db=DEFAULT_CONFIG.save_to_db,
            reporters=kwargs.get("reporters", []),
            reporter_options=kwargs.get("reporter_options", {}),
        )

    def test_default_console_reporter(self):
        args = self._make_args()
        config = self._make_config()
        reporters = _resolve_reporters(args, config, verbosity=0)
        assert len(reporters) == 1
        assert isinstance(reporters[0], ConsoleReporter)

    def test_default_reporter_always_added(self):
        args = self._make_args()
        config = self._make_config(reporters=[])
        reporters = _resolve_reporters(args, config, verbosity=0)
        assert len(reporters) == 1
        assert isinstance(reporters[0], ConsoleReporter)

    def test_cli_reporter_flag(self):
        args = self._make_args(reporters=["ConsoleReporter"])
        config = self._make_config()
        reporters = _resolve_reporters(args, config, verbosity=0)
        assert len(reporters) == 1
        assert isinstance(reporters[0], ConsoleReporter)

    def test_config_reporters(self):
        args = self._make_args()
        config = self._make_config(reporters=["ConsoleReporter"])
        reporters = _resolve_reporters(args, config, verbosity=0)
        assert len(reporters) == 1
        assert isinstance(reporters[0], ConsoleReporter)

    def test_cli_overrides_config(self):
        args = self._make_args(reporters=["ConsoleReporter"])
        config = self._make_config(reporters=["merit.reports.console:ConsoleReporter"])
        reporters = _resolve_reporters(args, config, verbosity=0)
        assert len(reporters) == 1

    def test_verbosity_passed_to_console_reporter(self):
        args = self._make_args()
        config = self._make_config()
        reporters = _resolve_reporters(args, config, verbosity=2)
        assert reporters[0].verbosity == 2
