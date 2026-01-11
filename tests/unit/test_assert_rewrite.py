import sys
from uuid import uuid4

import pytest

from merit.context import (
    TestContext,
    assertions_collector,
    metric_results_collector,
    metrics as metrics_scope_ctx,
    test_context_scope as context_scope_ctx,
)
from merit.metrics.base import Metric
from merit.testing.discovery import collect
from merit.testing.resources import ResourceResolver, clear_registry


def test_rewritten_assert_collects_predicate_results_and_metric_values(tmp_path):
    mod_name = f"merit_{uuid4().hex}"
    mod_path = tmp_path / f"{mod_name}.py"
    mod_path.write_text(
        """
from merit.metrics.base import Metric
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
        ctx = TestContext(item=item)
        with context_scope_ctx(ctx), assertions_collector(ctx.assertion_results):
            item.fn()

        assert len(ctx.assertion_results) == 1
        ar = ctx.assertion_results[0]
        assert ar.passed is True
        assert "equals(1, 1)" in ar.expression_repr

        assert len(ar.predicate_results) == 1
        assert ar.predicate_results[0].predicate_metadata.predicate_name == "equals"

        names = {mv.full_name for mv in ar.metric_values}
        assert "m.len" in names
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
        ctx = TestContext(item=item)
        with context_scope_ctx(ctx), assertions_collector(ctx.assertion_results):
            item.fn()

        assert len(ctx.assertion_results) == 1
        ar = ctx.assertion_results[0]
        assert ar.passed is False
        assert ar.error_message == "nope"
        assert "equals(1, 2)" in ar.expression_repr
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
        ctx = TestContext(item=item)
        m = Metric(name="assert_outcomes")
        with (
            context_scope_ctx(ctx),
            metrics_scope_ctx([m]),
            assertions_collector(ctx.assertion_results),
        ):
            item.fn()

        assert m.raw_values == [True, False]
        assert m.metadata.collected_from_merits == {"merit_metric_capture_multi"}
        assert len(ctx.assertion_results) == 2
        assert ctx.assertion_results[0].passed is True
        assert ctx.assertion_results[1].passed is False
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
from merit.metrics.base import Metric

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
        metric_results = []
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
