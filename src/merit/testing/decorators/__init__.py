"""Test decorators."""

from merit.testing.decorators.parametrize import parametrize
from merit.testing.decorators.repeat import repeat
from merit.testing.decorators.tags import TagData, get_tag_data, merge_tag_data, tag


__all__ = [
    "TagData",
    "get_tag_data",
    "merge_tag_data",
    "parametrize",
    "repeat",
    "tag",
]
