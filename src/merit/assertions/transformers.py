import ast


class InjectAssertionDependenciesTransformer(ast.NodeTransformer):
    """Inject assertion rewrite dependencies into a function body.

    We inject imports at the top of each transformed `merit_*` function so the
    rewritten assert statements can reference `AssertionResult` and
    `assertion_context_scope` without relying on the module loader to provide
    them.
    """

    def _inject_dependencies(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        inject_stmts: list[ast.stmt] = [
            ast.ImportFrom(
                module="merit.assertions.base",
                names=[ast.alias(name="AssertionResult", asname=None)],
                level=0,
            ),
            ast.ImportFrom(
                module="merit.context",
                names=[ast.alias(name="assertion_context_scope", asname=None)],
                level=0,
            ),
        ]

        body = list(node.body)
        insert_at = 1 if ast.get_docstring(node, clean=False) is not None else 0
        node.body = [*body[:insert_at], *inject_stmts, *body[insert_at:]]

        for stmt in inject_stmts:
            ast.copy_location(stmt, node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return self._inject_dependencies(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return self._inject_dependencies(node)


class AssertTransformer(ast.NodeTransformer):
    """Rewrite Python ``assert`` statements into Merit-aware instrumentation.

    This transformer replaces each :class:`ast.Assert` node with an equivalent
    sequence of statements that:

    - Constructs an ``AssertionResult`` capturing a string representation of the
      asserted expression (``expression_repr``).
    - Evaluates the assertion expression under ``assertion_context_scope(ar)``
      so downstream instrumentation can record context tied to that
      ``AssertionResult`` instance.
    - Stores the boolean outcome on ``ar.passed``.
    - If an ``assert`` message is present and the assertion fails, the message
      is coerced to ``str`` and stored on ``ar.error_message``.
    """

    AR_VAR_NAME = "__merit_ar"
    IS_PASSED_VAR_NAME = "__merit_passed"
    MSG_VAR_NAME = "__merit_msg"

    def __init__(self, source: str | None = None) -> None:
        self.source = source

    def visit_Assert(self, node: ast.Assert):
        # Get the source segment of the assertion expression
        segment = None
        if self.source is not None:
            segment = ast.get_source_segment(self.source, node.test)
        expr_repr = segment if isinstance(segment, str) and segment else ast.unparse(node.test)

        # Create the AssertionResult object
        ar_assign = ast.Assign(
            targets=[ast.Name(id=self.AR_VAR_NAME, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="AssertionResult", ctx=ast.Load()),
                args=[],
                keywords=[
                    ast.keyword(arg="expression_repr", value=ast.Constant(value=expr_repr)),
                ],
            ),
        )
        ast.copy_location(ar_assign, node)

        # Evaluate the assertion under the context
        eval_under_ctx = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Name(id="assertion_context_scope", ctx=ast.Load()),
                        args=[ast.Name(id=self.AR_VAR_NAME, ctx=ast.Load())],
                        keywords=[],
                    ),
                    optional_vars=None,
                )
            ],
            body=[
                ast.Assign(
                    targets=[ast.Name(id=self.IS_PASSED_VAR_NAME, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="bool", ctx=ast.Load()),
                        args=[node.test],
                        keywords=[],
                    ),
                )
            ],
        )
        ast.copy_location(eval_under_ctx, node)

        # Set the passed attribute of the AssertionResult object
        set_passed = ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=self.AR_VAR_NAME, ctx=ast.Load()),
                    attr="passed",
                    ctx=ast.Store(),
                )
            ],
            value=ast.Name(id=self.IS_PASSED_VAR_NAME, ctx=ast.Load()),
        )
        ast.copy_location(set_passed, node)

        if node.msg is None:
            return [ar_assign, eval_under_ctx, set_passed]

        # Get the error message if assertion did not pass and store it on the AssertionResult object
        fail_test = ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=self.IS_PASSED_VAR_NAME, ctx=ast.Load()))
        msg_assign = ast.Assign(
            targets=[ast.Name(id=self.MSG_VAR_NAME, ctx=ast.Store())],
            value=node.msg,
        )
        set_error_message = ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=self.AR_VAR_NAME, ctx=ast.Load()),
                    attr="error_message",
                    ctx=ast.Store(),
                )
            ],
            value=ast.Call(
                func=ast.Name(id="str", ctx=ast.Load()),
                args=[ast.Name(id=self.MSG_VAR_NAME, ctx=ast.Load())],
                keywords=[],
            ),
        )
        fail_if = ast.If(test=fail_test, body=[msg_assign, set_error_message], orelse=[])
        ast.copy_location(fail_if, node)

        return [ar_assign, eval_under_ctx, set_passed, fail_if]