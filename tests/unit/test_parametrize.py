import asyncio
from pathlib import Path

from merit.testing import Runner
from merit.testing.models import ParameterSet, ParametrizeModifier, TestItem
from merit.testing.parametrize import parametrize
from merit.testing.resources import clear_registry, resource


def test_parametrize_decorator_records_modifier():
    @parametrize("value", [1, 2], ids=["one", "two"])
    def sample(value):
        return value

    modifiers = getattr(sample, "__merit_modifiers__", [])
    assert len(modifiers) == 1
    assert isinstance(modifiers[0], ParametrizeModifier)
    assert len(modifiers[0].parameter_sets) == 2
    assert modifiers[0].parameter_sets[0].id_suffix == "one"
    assert modifiers[0].parameter_sets[1].id_suffix == "two"


def test_parametrize_stacking_creates_multiple_modifiers():
    @parametrize("value", [1, 2])
    @parametrize("flag", [True, False])
    def sample(value, flag):
        return value, flag

    modifiers = getattr(sample, "__merit_modifiers__", [])
    # Each decorator adds one modifier
    assert len(modifiers) == 2
    # First applied (inner): flag
    assert modifiers[0].parameter_sets[0].values == {"flag": True}
    # Second applied (outer): value
    assert modifiers[1].parameter_sets[0].values == {"value": 1}


def test_runner_applies_parameter_values():
    recorded = {}

    def merit_sample(param_a, resource_b):
        recorded["param_a"] = param_a
        recorded["resource_b"] = resource_b

    @resource
    def resource_b():
        return "from_resource"

    runner = Runner(reporters=[])

    # Create a parametrize modifier with one parameter set
    modifier = ParametrizeModifier(
        parameter_sets=(
            ParameterSet(values={"param_a": "from_param"}, id_suffix="param_a=from_param"),
        )
    )

    item = TestItem(
        name="merit_sample",
        fn=merit_sample,
        module_path=Path("sample.py"),
        is_async=False,
        params=["param_a", "resource_b"],
        modifiers=[modifier],
    )

    try:
        run_result = asyncio.run(runner.run(items=[item]))
    finally:
        clear_registry()

    # The result should have sub_runs for parametrize
    assert run_result.result.executions[0].result.sub_runs is not None
    assert len(run_result.result.executions[0].result.sub_runs) == 1
    assert recorded["param_a"] == "from_param"
    assert recorded["resource_b"] == "from_resource"
    assert run_result.result.passed == 1


def test_runner_runs_all_parameter_sets():
    results = []

    def merit_collect(x):
        results.append(x)

    runner = Runner(reporters=[])

    modifier = ParametrizeModifier(
        parameter_sets=(
            ParameterSet(values={"x": 1}, id_suffix="x=1"),
            ParameterSet(values={"x": 2}, id_suffix="x=2"),
            ParameterSet(values={"x": 3}, id_suffix="x=3"),
        )
    )

    item = TestItem(
        name="merit_collect",
        fn=merit_collect,
        module_path=Path("sample.py"),
        is_async=False,
        params=["x"],
        modifiers=[modifier],
    )

    run_result = asyncio.run(runner.run(items=[item]))

    assert results == [1, 2, 3]
    assert run_result.result.passed == 1
    assert run_result.result.executions[0].result.sub_runs is not None
    assert len(run_result.result.executions[0].result.sub_runs) == 3
