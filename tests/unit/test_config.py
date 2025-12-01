from pathlib import Path

from merit.config import load_config


def test_load_config_prefers_merit_toml(tmp_path: Path):
    project = tmp_path
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        """
[tool.merit]
test-paths = ["tests"]
include-tags = ["slow"]
maxfail = 1
verbosity = 1
addopts = ["-q"]
""".strip()
    )
    merit_file = project / "merit.toml"
    merit_file.write_text(
        """
test-paths = ["examples"]
include-tags = ["smoke"]
exclude-tags = ["slow"]
keyword = "chatbot"
""".strip()
    )

    config = load_config(project)

    assert config.test_paths == ["examples"]
    assert config.include_tags == ["smoke"]
    assert config.exclude_tags == ["slow"]
    assert config.keyword == "chatbot"
    assert config.maxfail == 1
    assert config.verbosity == 1
    assert config.addopts == ["-q"]


def test_load_config_defaults_when_missing(tmp_path: Path):
    project = tmp_path
    config = load_config(project)
    assert config.test_paths == ["."]
    assert config.include_tags == []
    assert config.exclude_tags == []
    assert config.keyword is None
    assert config.maxfail is None
    assert config.verbosity == 0
    assert config.addopts == []
