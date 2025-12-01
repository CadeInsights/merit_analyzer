import asyncio
import io
from pathlib import Path

from rich.console import Console

from merit.testing import Runner
from merit.testing.discovery import TestItem
from merit.testing.parametrize import get_parameter_sets, parametrize
from merit.testing.resources import clear_registry, resource


def test_get_parameter_sets_combines_multiple_decorators():
    @parametrize("value", [1, 2], ids=["one", "two"])
    @parametrize("flag", [True, False])
    def sample(value, flag):
        return value, flag

    sets = get_parameter_sets(sample)
    assert len(sets) == 4
    # Ensure ids compose in declaration order
    ids = sorted(s.id_suffix for s in sets)
    assert ids == ["flag=False-one", "flag=False-two", "flag=True-one", "flag=True-two"]


def test_runner_applies_parameter_values():
    recorded = {}

    def merit_sample(param_a, resource_b):
        recorded["param_a"] = param_a
        recorded["resource_b"] = resource_b

    @resource
    def resource_b():
        return "from_resource"

    console = Console(file=io.StringIO())
    runner = Runner(console=console)

    item = TestItem(
        name="merit_sample",
        fn=merit_sample,
        module_path=Path("sample.py"),
        is_async=False,
        params=["param_a", "resource_b"],
        param_values={"param_a": "from_param"},
        id_suffix="param_a=from_param",
    )

    try:
        run_result = asyncio.run(runner.run(items=[item]))
    finally:
        clear_registry()

    assert recorded["param_a"] == "from_param"
    assert recorded["resource_b"] == "from_resource"
    assert run_result.passed == 1
