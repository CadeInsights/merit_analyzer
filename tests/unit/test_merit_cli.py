from pathlib import Path

from merit.cli import KeywordMatcher, _filter_items
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
