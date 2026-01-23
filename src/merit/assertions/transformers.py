import ast


class InjectAssertionDependenciesTransformer(ast.NodeTransformer):
    """Inject assertion rewrite dependencies into a function body.

    We inject imports at the top of each transformed `merit_*` function so the
    rewritten assert statements can reference `AssertionResult`, `AssertionRepr`,
    `capture_var`, and `predicate_results_collector` without relying on the
    module loader to provide them.
    """

    def _inject_dependencies(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        inject_stmts: list[ast.stmt] = [
            ast.ImportFrom(
                module="merit.assertions.base",
                names=[
                    ast.alias(name="AssertionRepr", asname=None),
                    ast.alias(name="AssertionResult", asname=None),
                    ast.alias(name="capture_var", asname=None),
                ],
                level=0,
            ),
            ast.ImportFrom(
                module="merit.context",
                names=[
                    ast.alias(name="predicate_results_collector", asname=None),
                ],
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

    - Creates sink lists to collect predicate results.
    - Evaluates the assertion expression under ``predicate_results_collector``.
    - Constructs an ``AssertionResult`` with all collected data after evaluation.
    - If an ``assert`` message is present and the assertion fails, the message
      is coerced to ``str`` and stored on ``ar.error_message``.
    """

    AR_VAR_NAME = "__merit_ar"
    IS_PASSED_VAR_NAME = "__merit_passed"
    MSG_VAR_NAME = "__merit_msg"
    PREDICATE_RESULTS_VAR_NAME = "__merit_predicate_results"
    RESOLVED_ARGS_VAR_NAME = "__merit_resolved"

    def __init__(self, source: str | None = None) -> None:
        self.source = source

    def visit_Assert(self, node: ast.Assert):
        # Get the source segment of the assertion statement
        segment = None
        if self.source is not None:
            segment = ast.get_source_segment(self.source, node)
        expr_repr = (
            segment.strip()
            if isinstance(segment, str) and segment
            else f"assert {ast.unparse(node.test)}"
        )

        lines_above = ""
        lines_below = ""
        if self.source is not None and node.lineno is not None:
            source_lines = self.source.splitlines()
            start_index = max(0, node.lineno - 1)
            end_index = max(0, (node.end_lineno or node.lineno) - 1)
            above_lines = source_lines[max(0, start_index - 2) : start_index]
            below_lines = source_lines[end_index + 1 : end_index + 3]
            if above_lines:
                lines_above = "\n" + "\n".join(above_lines)
            if below_lines:
                lines_below = "\n" + "\n".join(below_lines)

        # Create empty lists for collecting predicate results
        predicate_results_assign = ast.Assign(
            targets=[ast.Name(id=self.PREDICATE_RESULTS_VAR_NAME, ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load()),
        )
        ast.copy_location(predicate_results_assign, node)

        resolved_args_assign = ast.Assign(
            targets=[ast.Name(id=self.RESOLVED_ARGS_VAR_NAME, ctx=ast.Store())],
            value=ast.Dict(keys=[], values=[]),
        )
        ast.copy_location(resolved_args_assign, node)

        def expr_name(expr: ast.expr) -> str:
            if self.source is not None:
                segment = ast.get_source_segment(self.source, expr)
                if isinstance(segment, str) and segment:
                    return segment.strip()
            return ast.unparse(expr)

        def wrap(expr: ast.expr) -> ast.expr:
            match expr:
                case ast.Name(ctx=ast.Load()) | ast.Attribute(ctx=ast.Load()) | ast.Subscript(ctx=ast.Load()):
                    label = expr_name(expr)
                case ast.Call() | ast.ListComp() | ast.SetComp() | ast.GeneratorExp() | ast.DictComp():
                    label = expr_name(expr)
                case ast.Compare():
                    expr.left = wrap(expr.left)
                    expr.comparators = [wrap(comparator) for comparator in expr.comparators]
                    return expr
                case ast.BinOp():
                    expr.left = wrap(expr.left)
                    expr.right = wrap(expr.right)
                    return expr
                case ast.BoolOp():
                    expr.values = [wrap(value) for value in expr.values]
                    return expr
                case ast.UnaryOp():
                    expr.operand = wrap(expr.operand)
                    return expr
                case ast.Tuple() | ast.List() | ast.Set():
                    expr.elts = [wrap(elt) for elt in expr.elts]
                    return expr
                case ast.Dict():
                    expr.keys = [wrap(key) if key is not None else None for key in expr.keys]
                    expr.values = [wrap(value) for value in expr.values]
                    return expr
                case _:
                    return expr

            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="capture_var", ctx=ast.Load()),
                    args=[
                        ast.Name(id=self.RESOLVED_ARGS_VAR_NAME, ctx=ast.Load()),
                        ast.Constant(value=label),
                        expr,
                    ],
                    keywords=[],
                ),
                expr,
            )

        wrapped_test = wrap(node.test)

        # Evaluate the assertion under collector contexts
        eval_under_collectors = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Name(id="predicate_results_collector", ctx=ast.Load()),
                        args=[ast.Name(id=self.PREDICATE_RESULTS_VAR_NAME, ctx=ast.Load())],
                        keywords=[],
                    ),
                    optional_vars=None,
                )],
            body=[
                ast.Assign(
                    targets=[ast.Name(id=self.IS_PASSED_VAR_NAME, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="bool", ctx=ast.Load()),
                        args=[wrapped_test],
                        keywords=[],
                    ),
                )
            ],
        )
        ast.copy_location(eval_under_collectors, node)

        # Build statements list starting with list creation and evaluation
        statements = [predicate_results_assign, resolved_args_assign, eval_under_collectors]

        # Conditionally evaluate and store error message if assertion fails
        if node.msg is not None:
            # __merit_msg = None
            # if not __merit_passed: __merit_msg = str(node.msg)
            msg_init = ast.Assign(
                targets=[ast.Name(id=self.MSG_VAR_NAME, ctx=ast.Store())],
                value=ast.Constant(value=None),
            )
            ast.copy_location(msg_init, node)
            statements.append(msg_init)

            fail_test = ast.UnaryOp(
                op=ast.Not(), operand=ast.Name(id=self.IS_PASSED_VAR_NAME, ctx=ast.Load())
            )
            msg_assign = ast.Assign(
                targets=[ast.Name(id=self.MSG_VAR_NAME, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="str", ctx=ast.Load()),
                    args=[node.msg],
                    keywords=[],
                ),
            )
            msg_if = ast.If(test=fail_test, body=[msg_assign], orelse=[])
            ast.copy_location(msg_if, node)
            statements.append(msg_if)

        # Create the AssertionResult with all collected data (after with block, passing lists directly)
        assertion_repr = ast.Call(
            func=ast.Name(id="AssertionRepr", ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(arg="expr", value=ast.Constant(value=expr_repr)),
                ast.keyword(arg="lines_above", value=ast.Constant(value=lines_above)),
                ast.keyword(arg="lines_below", value=ast.Constant(value=lines_below)),
                ast.keyword(
                    arg="resolved_args",
                    value=ast.Name(id=self.RESOLVED_ARGS_VAR_NAME, ctx=ast.Load()),
                ),
            ],
        )
        ar_keywords = [
            ast.keyword(arg="expression_repr", value=assertion_repr),
            ast.keyword(arg="passed", value=ast.Name(id=self.IS_PASSED_VAR_NAME, ctx=ast.Load())),
            ast.keyword(
                arg="predicate_results",
                value=ast.Name(id=self.PREDICATE_RESULTS_VAR_NAME, ctx=ast.Load()),
            )
        ]

        if node.msg is not None:
            ar_keywords.append(
                ast.keyword(
                    arg="error_message", value=ast.Name(id=self.MSG_VAR_NAME, ctx=ast.Load())
                )
            )

        ar_assign = ast.Assign(
            targets=[ast.Name(id=self.AR_VAR_NAME, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="AssertionResult", ctx=ast.Load()),
                args=[],
                keywords=ar_keywords,
            ),
        )
        ast.copy_location(ar_assign, node)
        statements.append(ar_assign)

        # Ensure all nested nodes have location info
        for stmt in statements:
            ast.fix_missing_locations(stmt)

        return statements

