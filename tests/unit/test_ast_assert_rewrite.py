"""Tests for merit.lib.assert_transformer module.

This module tests AST rewriting of `assert` statements. The transformer converts
assertions to record AssertionResult objects instead of raising AssertionError,
while capturing any PredicateResult or MetricResult values evaluated.
"""

import ast
import pytest

from merit.lib.assert_transformer import (
    AssertExprRewriter,
    AssertRewriteTransformer,
    build_injected_globals,
)
# Import dunder-prefixed helpers with non-mangled aliases to avoid Python's
# name mangling when used inside test classes
from merit.lib import assert_transformer as _at
record_assertion_result = _at.__merit_record_assertion_result
capture_predicate = _at.__merit_capture_predicate
capture_metric_attr = _at.__merit_capture_metric_attr

from merit.assertions.result import AssertionResult
from merit.predicates.base import PredicateResult, PredicateMetadata
from merit.metrics.base import Metric
from merit.metrics.result import MetricResult
from merit.context.context import (
    TEST_CONTEXT,
    METRICS_CONTEXT,
    TestContext,
    errors_to_metrics,
)
# Alias to avoid pytest collecting this as a test
from merit.context.context import test_context_scope as ctx_scope


# ---------------------------------------------------------------------------
# Tests for helper functions: record_assertion_result
# ---------------------------------------------------------------------------

class TestRecordAssertionResult:
    """Tests for record_assertion_result function."""

    def test_records_to_test_context(self):
        """When TEST_CONTEXT is set, assertion results are recorded there."""
        ctx = TestContext(test_item_name="test_func")
        ar = AssertionResult(expression="x == 1", passed=True)

        with ctx_scope(ctx):
            record_assertion_result(ar)

        assert ar in ctx.collected_assertion_results
        assert len(ctx.collected_assertion_results) == 1

    def test_records_to_metrics_context(self):
        """When METRICS_CONTEXT is set, assertion results are recorded to all metrics."""
        m1 = Metric(name="metric_1")
        m2 = Metric(name="metric_2")
        ar = AssertionResult(expression="y > 0", passed=False)

        with errors_to_metrics([m1, m2]):
            record_assertion_result(ar)

        # Metric stores AssertionResult.passed as a bool record
        assert m1.len == 1
        assert m2.len == 1
        assert m1.raw_values == [False]
        assert m2.raw_values == [False]

    def test_records_to_both_contexts(self):
        """When both contexts are set, assertion results go to both."""
        ctx = TestContext(test_item_name="dual_test")
        m = Metric(name="combined_metric")
        ar = AssertionResult(expression="z != None", passed=True)

        with ctx_scope(ctx):
            with errors_to_metrics([m]):
                record_assertion_result(ar)

        assert ar in ctx.collected_assertion_results
        assert m.raw_values == [True]

    def test_no_context_does_not_fail(self):
        """When no context is set, function completes without error."""
        ar = AssertionResult(expression="a == b", passed=False)
        # Should not raise
        record_assertion_result(ar)


# ---------------------------------------------------------------------------
# Tests for helper functions: capture_predicate
# ---------------------------------------------------------------------------

class TestCapturePredicate:
    """Tests for capture_predicate function."""

    def test_captures_predicate_result(self):
        """PredicateResult values are appended to sink."""
        sink: list[PredicateResult] = []
        pr = PredicateResult(
            predicate_metadata=PredicateMetadata(actual="a", reference="b"),
            value=True,
        )

        result = capture_predicate(pr, sink)

        assert result is pr  # Returns original value
        assert pr in sink
        assert len(sink) == 1

    def test_non_predicate_not_captured(self):
        """Non-PredicateResult values are not captured but still returned."""
        sink: list[PredicateResult] = []

        # Various non-PredicateResult values
        assert capture_predicate(True, sink) is True
        assert capture_predicate(42, sink) == 42
        assert capture_predicate("string", sink) == "string"
        assert capture_predicate(None, sink) is None

        assert len(sink) == 0

    def test_preserves_expression_semantics(self):
        """Function returns original value unchanged for chaining."""
        sink: list[PredicateResult] = []
        obj = {"key": "value"}

        result = capture_predicate(obj, sink)

        assert result is obj
        assert len(sink) == 0


# ---------------------------------------------------------------------------
# Tests for helper functions: capture_metric_attr
# ---------------------------------------------------------------------------

class TestCaptureMetricAttr:
    """Tests for capture_metric_attr function."""

    def test_captures_numeric_metric_attribute(self):
        """Numeric attributes on Metric instances are captured."""
        sink: list[MetricResult] = []
        m = Metric(name="test_metric")
        m.add_record([10, 20, 30])

        result = capture_metric_attr(m, "mean", sink)

        assert result == 20.0  # mean of [10, 20, 30]
        assert len(sink) == 1
        assert sink[0].metric_full_name == "test_metric.mean"
        assert sink[0].metric_value == 20.0

    def test_captures_various_metric_attributes(self):
        """Various numeric-like attributes are captured."""
        sink: list[MetricResult] = []
        m = Metric(name="stats")
        m.add_record([1, 2, 3, 4, 5])

        capture_metric_attr(m, "len", sink)
        capture_metric_attr(m, "sum", sink)
        capture_metric_attr(m, "min", sink)
        capture_metric_attr(m, "max", sink)

        assert len(sink) == 4
        names = [r.metric_full_name for r in sink]
        assert "stats.len" in names
        assert "stats.sum" in names
        assert "stats.min" in names
        assert "stats.max" in names

    def test_non_numeric_attribute_not_captured(self):
        """Non-numeric attributes are returned but not captured."""
        sink: list[MetricResult] = []
        m = Metric(name="test")
        m.add_record([1, 2, 3])

        class WeirdMetric(Metric):
            @property
            def weird(self) -> list[str]:
                return ["not", "numeric"]

        weird = WeirdMetric(name="weird")
        result = capture_metric_attr(weird, "weird", sink)

        assert result == ["not", "numeric"]
        assert len(sink) == 0  # list contains non-scalars, so we don't capture it

    def test_list_of_scalars_attribute_captured(self):
        """Lists/tuples of numeric-like values on Metric instances are captured."""
        sink: list[MetricResult] = []
        m = Metric(name="test")
        m.add_record([1, 2, 3])

        result = capture_metric_attr(m, "raw_values", sink)

        assert result == [1, 2, 3]
        assert len(sink) == 1
        assert sink[0].metric_full_name == "test.raw_values"
        assert sink[0].metric_value == [1.0, 2.0, 3.0]

    def test_non_metric_object_not_captured(self):
        """Attribute access on non-Metric objects is not captured."""
        sink: list[MetricResult] = []

        class Dummy:
            mean = 42.0

        obj = Dummy()
        result = capture_metric_attr(obj, "mean", sink)

        assert result == 42.0
        assert len(sink) == 0

    def test_uses_metric_name_or_class_name(self):
        """Falls back to class name when metric has no name."""
        sink: list[MetricResult] = []
        m = Metric()  # No name set
        m.add_record([5])

        capture_metric_attr(m, "sum", sink)

        assert sink[0].metric_full_name == "Metric.sum"


# ---------------------------------------------------------------------------
# Tests for AssertExprRewriter
# ---------------------------------------------------------------------------

class TestAssertExprRewriter:
    """Tests for AST expression rewriting."""

    def test_rewrite_simple_expression_unchanged(self):
        """Simple expressions without calls/attributes pass through."""
        rewriter = AssertExprRewriter(
            predicate_sink_name="__preds",
            metric_sink_name="__metrics",
        )
        # Simple name reference
        tree = ast.parse("x", mode="eval")
        original = ast.dump(tree.body)
        rewritten = rewriter.rewrite(tree.body)
        # Names should remain unchanged
        assert isinstance(rewritten, ast.Name)
        assert rewritten.id == "x"

    def test_rewrite_call_wraps_with_capture(self):
        """Function calls are wrapped with __merit_capture_predicate."""
        rewriter = AssertExprRewriter(
            predicate_sink_name="__preds",
            metric_sink_name="__metrics",
        )
        tree = ast.parse("some_predicate(a, b)", mode="eval")
        rewritten = rewriter.rewrite(tree.body)

        # Should be: __merit_capture_predicate(some_predicate(a, b), __preds)
        assert isinstance(rewritten, ast.Call)
        assert isinstance(rewritten.func, ast.Name)
        assert rewritten.func.id == "__merit_capture_predicate"
        assert len(rewritten.args) == 2
        # Second arg is the sink
        sink_arg = rewritten.args[1]
        assert isinstance(sink_arg, ast.Name)
        assert sink_arg.id == "__preds"

    def test_rewrite_attribute_wraps_with_metric_capture(self):
        """Attribute access is wrapped with __merit_capture_metric_attr."""
        rewriter = AssertExprRewriter(
            predicate_sink_name="__preds",
            metric_sink_name="__metrics",
        )
        tree = ast.parse("m.mean", mode="eval")
        rewritten = rewriter.rewrite(tree.body)

        # Should be: __merit_capture_metric_attr(m, "mean", __metrics)
        assert isinstance(rewritten, ast.Call)
        assert isinstance(rewritten.func, ast.Name)
        assert rewritten.func.id == "__merit_capture_metric_attr"
        assert len(rewritten.args) == 3
        # First arg is the object (rewritten)
        # Second arg is attribute name as constant
        assert isinstance(rewritten.args[1], ast.Constant)
        assert rewritten.args[1].value == "mean"
        # Third arg is the sink
        assert isinstance(rewritten.args[2], ast.Name)
        assert rewritten.args[2].id == "__metrics"

    def test_rewrite_await_wraps_after_await(self):
        """Await expressions wrap the result after awaiting."""
        rewriter = AssertExprRewriter(
            predicate_sink_name="__preds",
            metric_sink_name="__metrics",
        )
        # await async_pred(x)
        tree = ast.parse("await async_pred(x)", mode="eval")
        rewritten = rewriter.rewrite(tree.body)

        # Should be: __merit_capture_predicate(await async_pred(x), __preds)
        assert isinstance(rewritten, ast.Call)
        func = rewritten.func
        assert isinstance(func, ast.Name)
        assert func.id == "__merit_capture_predicate"
        # First arg should be an Await node
        assert isinstance(rewritten.args[0], ast.Await)

    def test_rewrite_nested_calls(self):
        """Nested function calls are all wrapped."""
        rewriter = AssertExprRewriter(
            predicate_sink_name="__preds",
            metric_sink_name="__metrics",
        )
        tree = ast.parse("outer(inner(x))", mode="eval")
        rewritten = rewriter.rewrite(tree.body)

        # Outer call wrapped
        assert isinstance(rewritten, ast.Call)
        outer_func = rewritten.func
        assert isinstance(outer_func, ast.Name)
        assert outer_func.id == "__merit_capture_predicate"

        # Inner call is the argument to outer, which is also wrapped
        outer_first_arg = rewritten.args[0]
        assert isinstance(outer_first_arg, ast.Call)
        inner = outer_first_arg.args[0]
        assert isinstance(inner, ast.Call)
        inner_func = inner.func
        assert isinstance(inner_func, ast.Name)
        assert inner_func.id == "__merit_capture_predicate"

    def test_rewrite_comparison_with_metric_attr(self):
        """Comparisons with metric attributes are properly rewritten."""
        rewriter = AssertExprRewriter(
            predicate_sink_name="__preds",
            metric_sink_name="__metrics",
        )
        tree = ast.parse("m.mean > 0.5", mode="eval")
        rewritten = rewriter.rewrite(tree.body)

        # Should be a Compare node with left side wrapped
        assert isinstance(rewritten, ast.Compare)
        # Left operand should be the wrapped attribute access
        left = rewritten.left
        assert isinstance(left, ast.Call)
        left_func = left.func
        assert isinstance(left_func, ast.Name)
        assert left_func.id == "__merit_capture_metric_attr"


# ---------------------------------------------------------------------------
# Tests for AssertRewriteTransformer
# ---------------------------------------------------------------------------

class TestAssertRewriteTransformer:
    """Tests for full assert statement transformation."""

    def test_simple_assert_transformed(self):
        """Simple assert becomes block that records AssertionResult."""
        source = "assert x == 1"
        tree = ast.parse(source)
        transformer = AssertRewriteTransformer(source, filename="<test>")
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        # Original assert should be replaced with multiple statements
        assert len(new_tree.body) > 1

        # Should contain AssertionResult creation
        code = ast.unparse(new_tree)
        assert "AssertionResult" in code
        assert "record_assertion_result" in code

    def test_assert_with_message_transformed(self):
        """Assert with message preserves lazy evaluation."""
        source = 'assert x > 0, "x must be positive"'
        tree = ast.parse(source)
        transformer = AssertRewriteTransformer(source, filename="<test>")
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        code = ast.unparse(new_tree)
        # Message should appear somewhere in the transformed code
        assert "x must be positive" in code

    def test_assert_preserves_source_expression(self):
        """Transformed code includes original expression text."""
        source = "assert result.is_valid()"
        tree = ast.parse(source)
        transformer = AssertRewriteTransformer(source, filename="<test>")
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        code = ast.unparse(new_tree)
        # Expression text should be preserved as string constant
        assert "result.is_valid()" in code

    def test_multiple_asserts_transformed(self):
        """Multiple asserts in module are all transformed."""
        source = """
assert a == 1
assert b == 2
assert c == 3
"""
        tree = ast.parse(source)
        transformer = AssertRewriteTransformer(source, filename="<test>")
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        code = ast.unparse(new_tree)
        # Each assert generates a record call
        assert code.count("record_assertion_result") == 3

    def test_assert_in_function_transformed(self):
        """Asserts inside functions are transformed."""
        source = """
def test_func():
    assert x == 1
"""
        tree = ast.parse(source)
        transformer = AssertRewriteTransformer(source, filename="<test>")
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        code = ast.unparse(new_tree)
        assert "record_assertion_result" in code
        assert "def test_func" in code


# ---------------------------------------------------------------------------
# End-to-end tests: compile and execute transformed code
# ---------------------------------------------------------------------------

class TestEndToEndExecution:
    """Integration tests that compile and execute transformed assertions."""

    def _transform_and_exec(self, source: str, extra_globals: dict | None = None):
        """Helper to transform source, compile, and execute."""
        tree = ast.parse(source)
        transformer = AssertRewriteTransformer(source, filename="<test>")
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        globals_dict = build_injected_globals()
        if extra_globals:
            globals_dict.update(extra_globals)

        code = compile(new_tree, "<test>", "exec")
        exec(code, globals_dict)
        return globals_dict

    def test_passing_assert_records_true(self):
        """Passing assertion records passed=True."""
        ctx = TestContext(test_item_name="test_pass")

        with ctx_scope(ctx):
            self._transform_and_exec("assert 1 == 1")

        assert len(ctx.collected_assertion_results) == 1
        ar = ctx.collected_assertion_results[0]
        assert ar.passed is True
        assert ar.expression == "1 == 1"

    def test_failing_assert_records_false_no_raise(self):
        """Failing assertion records passed=False without raising."""
        ctx = TestContext(test_item_name="test_fail")

        with ctx_scope(ctx):
            # This should NOT raise AssertionError
            self._transform_and_exec("assert 1 == 2")

        assert len(ctx.collected_assertion_results) == 1
        ar = ctx.collected_assertion_results[0]
        assert ar.passed is False
        assert ar.expression == "1 == 2"

    def test_assert_with_message_captures_on_failure(self):
        """Failure with message captures the message string."""
        ctx = TestContext(test_item_name="test_msg")

        with ctx_scope(ctx):
            self._transform_and_exec('assert False, "expected true"')

        ar = ctx.collected_assertion_results[0]
        assert ar.passed is False
        assert ar.error_message == "expected true"

    def test_assert_message_lazy_evaluated(self):
        """Message is only evaluated when assertion fails."""
        ctx = TestContext(test_item_name="test_lazy")
        
        # If message were eagerly evaluated, this would fail
        source = """
evaluated = []
def track():
    evaluated.append(1)
    return "message"
assert True, track()
"""
        with ctx_scope(ctx):
            g = self._transform_and_exec(source, {"evaluated": []})

        # track() should not have been called
        assert g["evaluated"] == []

    def test_predicate_result_captured(self):
        """PredicateResult from predicate calls is captured."""
        ctx = TestContext(test_item_name="test_pred")

        pr = PredicateResult(
            predicate_metadata=PredicateMetadata(actual="x", reference="y"),
            value=True,
        )

        def fake_predicate():
            return pr

        with ctx_scope(ctx):
            self._transform_and_exec(
                "assert fake_predicate()",
                {"fake_predicate": fake_predicate},
            )

        ar = ctx.collected_assertion_results[0]
        assert pr in ar.captured_predicate_results

    def test_metric_attribute_captured(self):
        """Metric attribute reads are captured in assertion result."""
        ctx = TestContext(test_item_name="test_metric_attr")
        m = Metric(name="accuracy")
        m.add_record([0.9, 0.95, 1.0])

        with ctx_scope(ctx):
            self._transform_and_exec(
                "assert m.mean > 0.5",
                {"m": m},
            )

        ar = ctx.collected_assertion_results[0]
        assert ar.passed is True
        assert len(ar.captured_metric_results) == 1
        assert ar.captured_metric_results[0].metric_full_name == "accuracy.mean"
        assert ar.captured_metric_results[0].metric_value == pytest.approx(0.95)

    def test_multiple_metric_attrs_captured(self):
        """Multiple metric attribute reads in one assert are all captured."""
        ctx = TestContext(test_item_name="test_multi_metric")
        m = Metric(name="scores")
        m.add_record([1, 2, 3, 4, 5])

        with ctx_scope(ctx):
            self._transform_and_exec(
                "assert m.min <= m.mean <= m.max",
                {"m": m},
            )

        ar = ctx.collected_assertion_results[0]
        assert ar.passed is True
        assert len(ar.captured_metric_results) == 3
        names = [r.metric_full_name for r in ar.captured_metric_results]
        assert "scores.min" in names
        assert "scores.mean" in names
        assert "scores.max" in names

    def test_records_to_metrics_context(self):
        """Transformed asserts record to METRICS_CONTEXT when set."""
        tracking_metric = Metric(name="all_assertions")

        with errors_to_metrics([tracking_metric]):
            self._transform_and_exec("assert True")
            self._transform_and_exec("assert False")

        assert tracking_metric.len == 2
        assert tracking_metric.raw_values == [True, False]

    def test_complex_expression_preserved(self):
        """Complex expressions have their text properly captured."""
        ctx = TestContext(test_item_name="test_complex")

        source = "assert len(items) > 0 and all(x > 0 for x in items)"
        with ctx_scope(ctx):
            self._transform_and_exec(source, {"items": [1, 2, 3]})

        ar = ctx.collected_assertion_results[0]
        # Expression text should be captured (may vary slightly due to ast.get_source_segment)
        assert "len(items)" in ar.expression or "items" in ar.expression


@pytest.mark.asyncio
class TestAsyncAssertTransformation:
    """Tests for async assert transformation."""

    async def _transform_and_exec_async(self, source: str, extra_globals: dict | None = None):
        """Helper to transform async source, compile, and execute."""
        # Wrap in async function
        wrapped_source = f"""
async def __test_async():
{chr(10).join('    ' + line for line in source.split(chr(10)))}
"""
        tree = ast.parse(wrapped_source)
        transformer = AssertRewriteTransformer(wrapped_source, filename="<test>")
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        globals_dict = build_injected_globals()
        if extra_globals:
            globals_dict.update(extra_globals)

        code = compile(new_tree, "<test>", "exec")
        exec(code, globals_dict)
        await globals_dict["__test_async"]()
        return globals_dict

    async def test_await_predicate_captured(self):
        """Awaited predicate results are captured."""
        ctx = TestContext(test_item_name="test_async_pred")

        pr = PredicateResult(
            predicate_metadata=PredicateMetadata(actual="a", reference="b"),
            value=True,
        )

        async def async_predicate():
            return pr

        with ctx_scope(ctx):
            await self._transform_and_exec_async(
                "assert await async_predicate()",
                {"async_predicate": async_predicate},
            )

        ar = ctx.collected_assertion_results[0]
        assert ar.passed is True
        assert pr in ar.captured_predicate_results


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_module_unchanged(self):
        """Empty module passes through transformer unchanged."""
        source = ""
        tree = ast.parse(source)
        transformer = AssertRewriteTransformer(source, filename="<test>")
        new_tree = transformer.visit(tree)

        assert len(new_tree.body) == 0

    def test_no_asserts_unchanged(self):
        """Module without asserts passes through unchanged."""
        source = """
x = 1
y = 2
def foo():
    return x + y
"""
        tree = ast.parse(source)
        transformer = AssertRewriteTransformer(source, filename="<test>")
        new_tree = transformer.visit(tree)

        # Structure should be similar (no assert replacements)
        assert len(new_tree.body) == 3  # x=1, y=2, def foo

    def test_assert_with_complex_comparison(self):
        """Complex comparisons are handled correctly."""
        ctx = TestContext(test_item_name="test_complex_cmp")

        tree = ast.parse("assert 1 < 2 < 3")
        transformer = AssertRewriteTransformer("assert 1 < 2 < 3", filename="<test>")
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        globals_dict = build_injected_globals()
        with ctx_scope(ctx):
            exec(compile(new_tree, "<test>", "exec"), globals_dict)

        ar = ctx.collected_assertion_results[0]
        assert ar.passed is True

    def test_assert_with_method_call(self):
        """Method calls on objects are properly transformed."""
        ctx = TestContext(test_item_name="test_method")

        class Checker:
            def is_valid(self):
                return True

        tree = ast.parse("assert obj.is_valid()")
        transformer = AssertRewriteTransformer("assert obj.is_valid()", filename="<test>")
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        globals_dict = build_injected_globals()
        globals_dict["obj"] = Checker()

        with ctx_scope(ctx):
            exec(compile(new_tree, "<test>", "exec"), globals_dict)

        ar = ctx.collected_assertion_results[0]
        assert ar.passed is True

    def test_bool_conversion_of_predicate_result(self):
        """PredicateResult is properly converted to bool for pass/fail."""
        ctx = TestContext(test_item_name="test_bool_conv")

        pr_false = PredicateResult(
            predicate_metadata=PredicateMetadata(actual="a", reference="b"),
            value=False,
        )

        def returns_false_predicate():
            return pr_false

        tree = ast.parse("assert returns_false_predicate()")
        transformer = AssertRewriteTransformer(
            "assert returns_false_predicate()", filename="<test>"
        )
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        globals_dict = build_injected_globals()
        globals_dict["returns_false_predicate"] = returns_false_predicate

        with ctx_scope(ctx):
            exec(compile(new_tree, "<test>", "exec"), globals_dict)

        ar = ctx.collected_assertion_results[0]
        # PredicateResult(value=False) should make passed=False
        assert ar.passed is False
        assert pr_false in ar.captured_predicate_results
