import sys
from uuid import uuid4

import pytest

from merit.assertions.base import AssertionResult
from merit.context import (
    TestContext,
    assertions_collector,
    metric_results_collector,
    metrics as metrics_scope_ctx,
    test_context_scope as context_scope_ctx,
)
from merit.metrics_.base import Metric, MetricResult
from merit.resources import ResourceResolver, clear_registry
from merit.testing.discovery import collect


def test_rewritten_assert_collects_predicate_results(tmp_path):
    mod_name = f"merit_{uuid4().hex}"
    mod_path = tmp_path / f"{mod_name}.py"
    mod_path.write_text(
        """
from merit.metrics_.base import Metric
from merit.predicates.base import predicate

@predicate
def equals(actual, reference):
    return actual == reference

def merit_sample():
    m = Metric(name="m")
    m.add_record([1, 2, 3])
    assert equals(1, 1) and (m.len == 3)
""".lstrip()
    )

    try:
        [item] = collect(mod_path)
        assertion_results: list[AssertionResult] = []
        ctx = TestContext(item=item)
        with context_scope_ctx(ctx), assertions_collector(assertion_results):
            item.fn()

        assert len(assertion_results) == 1
        ar = assertion_results[0]
        assert ar.passed is True
        assert "equals(1, 1)" in ar.expression_repr.expr

        assert len(ar.predicate_results) == 1
        assert ar.predicate_results[0].predicate_metadata.predicate_name == "equals"

    finally:
        sys.modules.pop(mod_name, None)


def test_rewritten_assert_failure_sets_error_message_and_raises(tmp_path):
    mod_name = f"merit_{uuid4().hex}"
    mod_path = tmp_path / f"{mod_name}.py"
    mod_path.write_text(
        """
from merit.predicates.base import predicate

@predicate
def equals(actual, reference):
    return actual == reference

def merit_fail():
    assert equals(1, 2), "nope"
""".lstrip()
    )

    try:
        [item] = collect(mod_path)
        assertion_results: list[AssertionResult] = []
        ctx = TestContext(item=item)
        with context_scope_ctx(ctx), assertions_collector(assertion_results):
            item.fn()

        assert len(assertion_results) == 1
        ar = assertion_results[0]
        assert ar.passed is False
        assert ar.error_message == "nope"
        assert "equals(1, 2)" in ar.expression_repr.expr
    finally:
        sys.modules.pop(mod_name, None)


def test_rewritten_multiple_asserts_record_multiple_metric_values(tmp_path):
    mod_name = f"merit_{uuid4().hex}"
    mod_path = tmp_path / f"{mod_name}.py"
    mod_path.write_text(
        """
from merit.predicates.base import predicate

@predicate
def equals(actual, reference):
    return actual == reference

def merit_metric_capture_multi():
    assert equals(1, 1)
    assert equals(1, 2), "nope"
""".lstrip()
    )

    try:
        [item] = collect(mod_path)
        assertion_results: list[AssertionResult] = []
        ctx = TestContext(item=item)
        m = Metric(name="assert_outcomes")
        with (
            context_scope_ctx(ctx),
            metrics_scope_ctx([m]),
            assertions_collector(assertion_results),
        ):
            item.fn()

        assert m.raw_values == [True, False]
        assert m.metadata.collected_from_merits == {"merit_metric_capture_multi"}
        assert len(assertion_results) == 2
        assert assertion_results[0].passed is True
        assert assertion_results[1].passed is False
    finally:
        sys.modules.pop(mod_name, None)


@pytest.mark.asyncio
async def test_rewritten_asserts_inside_metric_functions_are_collected(tmp_path):
    clear_registry()
    mod_name = f"merit_{uuid4().hex}"
    mod_path = tmp_path / f"{mod_name}.py"
    mod_path.write_text(
        """
import merit
from merit.metrics_.base import Metric

@merit.metric
def my_metric():
    m = Metric(name="m")
    yield m
    assert False, "nope"

def merit_dummy():
    pass
""".lstrip()
    )

    try:
        [item] = collect(mod_path)
        resolver = ResourceResolver()
        metric_results: list[MetricResult] = []
        with metric_results_collector(metric_results):
            await resolver.resolve("my_metric")
            await resolver.teardown()

        assert len(metric_results) == 1
        [metric_result] = metric_results
        assert len(metric_result.assertion_results) == 1
        ar = metric_result.assertion_results[0]
        assert ar.passed is False
        assert ar.error_message == "nope"
    finally:
        clear_registry()
        sys.modules.pop(mod_name, None)


def test_assertion_repr_major_cases(tmp_path):
    mod_name = f"merit_{uuid4().hex}"
    mod_path = tmp_path / f"{mod_name}.py"
    mod_path.write_text(
        """
def merit_repr_cases():
    t1 = 5
    t2 = 10
    print(t1)
    print(t2)
    assert t1
    print("hello")
    print("world")
    class Obj:
        def __init__(self):
            self.attr = 7
    obj = Obj()
    assert obj.attr
    arr = [1, 2, 3]
    assert arr[0]
    def func(x):
        return x + 1
    assert func(4)
    xs = [0, 1, 2]
    assert [x > 0 for x in xs]
""".lstrip()
    )

    try:
        [item] = collect(mod_path)
        assertion_results: list[AssertionResult] = []
        with assertions_collector(assertion_results):
            item.fn()

        assert len(assertion_results) == 5
        results = {ar.expression_repr.expr: ar.expression_repr for ar in assertion_results}

        t1_repr = results["assert t1"]
        assert t1_repr.lines_above == "\n    print(t1)\n    print(t2)"
        assert t1_repr.lines_below == '\n    print("hello")\n    print("world")'
        assert t1_repr.resolved_args == {"t1": "5"}

        assert results["assert obj.attr"].resolved_args == {"obj.attr": "7"}
        assert results["assert arr[0]"].resolved_args == {"arr[0]": "1"}
        assert results["assert func(4)"].resolved_args == {"func(4)": "5"}
        assert results["assert [x > 0 for x in xs]"].resolved_args == {
            "[x > 0 for x in xs]": "[False, True, True]"
        }
    finally:
        sys.modules.pop(mod_name, None)


def test_assertion_repr_compare_cases(tmp_path):
    mod_name = f"merit_{uuid4().hex}"
    mod_path = tmp_path / f"{mod_name}.py"
    mod_path.write_text(
        """
def merit_compare_cases():
    t1 = 5
    t2 = 10
    assert t1 < t2
    assert t1 + t2 > 0
    a = 1
    b = 2
    c = 3
    assert a < b < c
""".lstrip()
    )

    try:
        [item] = collect(mod_path)
        assertion_results: list[AssertionResult] = []
        with assertions_collector(assertion_results):
            item.fn()

        assert len(assertion_results) == 3
        results = {ar.expression_repr.expr: ar.expression_repr for ar in assertion_results}

        assert results["assert t1 < t2"].resolved_args == {"t1": "5", "t2": "10"}
        assert results["assert t1 + t2 > 0"].resolved_args == {"t1": "5", "t2": "10"}
        assert results["assert a < b < c"].resolved_args == {"a": "1", "b": "2", "c": "3"}
    finally:
        sys.modules.pop(mod_name, None)


def test_assertion_repr_complex_compare_and_boolop(tmp_path):
    mod_name = f"merit_{uuid4().hex}"
    mod_path = tmp_path / f"{mod_name}.py"
    mod_path.write_text(
        """
def merit_complex_compare_cases():
    a = 5
    b = [5, 6, 7]
    def c():
        return 20
    assert a > 1 and a in b and sum(b) < c()
""".lstrip()
    )

    try:
        [item] = collect(mod_path)
        assertion_results: list[AssertionResult] = []
        with assertions_collector(assertion_results):
            item.fn()

        assert len(assertion_results) == 1
        [ar] = assertion_results
        assert ar.expression_repr.expr == "assert a > 1 and a in b and sum(b) < c()"
        assert ar.expression_repr.resolved_args == {
            "a": "5",
            "b": "[5, 6, 7]",
            "sum(b)": "18",
            "c()": "20",
        }
    finally:
        sys.modules.pop(mod_name, None)


def test_assertion_repr_multiline_assert(tmp_path):
    mod_name = f"merit_{uuid4().hex}"
    mod_path = tmp_path / f"{mod_name}.py"
    mod_path.write_text(
        """
def merit_multiline_assert():
    def func(x):
        return x + 1
    x = 4
    print("above")
    assert func(
        x
    )
    print("below")
""".lstrip()
    )

    try:
        [item] = collect(mod_path)
        assertion_results: list[AssertionResult] = []
        with assertions_collector(assertion_results):
            item.fn()

        assert len(assertion_results) == 1
        [ar] = assertion_results

        assert ar.expression_repr.expr == "assert func(\n        x\n    )"
        assert ar.expression_repr.lines_above == '\n    x = 4\n    print("above")'
        assert ar.expression_repr.lines_below == '\n    print("below")'
        assert ar.expression_repr.resolved_args == {"func(\n        x\n    )": "5"}
    finally:
        sys.modules.pop(mod_name, None)
