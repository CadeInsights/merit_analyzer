"""Configuration loading for the Merit CLI."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MeritConfig:
    """Resolved configuration values for running tests."""

    test_paths: list[str]
    include_tags: list[str]
    exclude_tags: list[str]
    keyword: str | None
    maxfail: int | None
    verbosity: int
    addopts: list[str]
    concurrency: int  # 1=sequential, 0=unlimited, max default=10


DEFAULT_CONFIG = MeritConfig(
    test_paths=["."],
    include_tags=[],
    exclude_tags=[],
    keyword=None,
    maxfail=None,
    verbosity=0,
    addopts=[],
    concurrency=1,
)


def load_config(start_path: str | Path | None = None) -> MeritConfig:
    """Load configuration from pyproject.toml or merit.toml."""
    base = Path(start_path or Path.cwd()).resolve()
    config = MeritConfig(
        test_paths=list(DEFAULT_CONFIG.test_paths),
        include_tags=list(DEFAULT_CONFIG.include_tags),
        exclude_tags=list(DEFAULT_CONFIG.exclude_tags),
        keyword=DEFAULT_CONFIG.keyword,
        maxfail=DEFAULT_CONFIG.maxfail,
        verbosity=DEFAULT_CONFIG.verbosity,
        addopts=list(DEFAULT_CONFIG.addopts),
        concurrency=DEFAULT_CONFIG.concurrency,
    )

    pyproject = _find_file(base, "pyproject.toml")
    if pyproject:
        data = _load_toml(pyproject)
        section = data.get("tool", {}).get("merit")
        if isinstance(section, dict):
            _apply_section(config, section)

    merit_toml = _find_file(base, "merit.toml")
    if merit_toml:
        data = _load_toml(merit_toml)
        _apply_section(config, data)

    if not config.test_paths:
        config.test_paths = list(DEFAULT_CONFIG.test_paths)

    return config


def _find_file(start: Path, filename: str) -> Path | None:
    """Search upwards from start for filename."""
    current = start
    while True:
        candidate = current / filename
        if candidate.exists():
            return candidate
        if current.parent == current:
            return None
        current = current.parent


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fp:
        return tomllib.load(fp)


def _apply_section(config: MeritConfig, section: dict[str, Any]) -> None:
    """Apply a single config section to the resolved config."""
    mapping = {
        "test-paths": "test_paths",
        "test_paths": "test_paths",
        "include-tags": "include_tags",
        "include_tags": "include_tags",
        "exclude-tags": "exclude_tags",
        "exclude_tags": "exclude_tags",
        "keyword": "keyword",
        "maxfail": "maxfail",
        "verbosity": "verbosity",
        "addopts": "addopts",
        "concurrency": "concurrency",
    }

    for key, value in section.items():
        attr = mapping.get(key)
        if attr is None:
            continue
        if attr in {"test_paths", "include_tags", "exclude_tags", "addopts"}:
            if isinstance(value, list):
                setattr(config, attr, [str(v) for v in value])
        elif attr == "verbosity":
            if isinstance(value, int):
                config.verbosity = value
        elif attr == "maxfail":
            if isinstance(value, int) and value > 0:
                config.maxfail = value
        elif attr == "keyword":
            if isinstance(value, str):
                config.keyword = value
        elif attr == "concurrency":
            if isinstance(value, int) and value >= 0:
                config.concurrency = value


__all__ = ["DEFAULT_CONFIG", "MeritConfig", "load_config"]
