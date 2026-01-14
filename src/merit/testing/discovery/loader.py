import ast
import importlib.abc
from pathlib import Path
from types import ModuleType
from typing import TypeVar

from merit.assertions.transformers import AssertTransformer, InjectAssertionDependenciesTransformer


TFunction = TypeVar("TFunction", ast.FunctionDef, ast.AsyncFunctionDef)


class MeritFunctionTransformer(ast.NodeTransformer):
    """Finds all functions in the module that start with `merit_` and transforms them."""

    def __init__(self, transformers: list[ast.NodeTransformer]) -> None:
        self.transformers = transformers

    def apply_transformers(self, node: TFunction) -> TFunction:
        """Apply configured transformer pipeline to a single function node."""
        for transformer in self.transformers:
            node = transformer.visit(node)
        return ast.fix_missing_locations(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name.startswith("merit_"):
            node = self.apply_transformers(node)
            return node
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if node.name.startswith("merit_"):
            node = self.apply_transformers(node)
            return node
        return self.generic_visit(node)


class MeritMetricTransformer(ast.NodeTransformer):
    """Find metric functions (decorated with `@merit.metric` / `@metric`) and transform them."""

    def __init__(self, transformers: list[ast.NodeTransformer]) -> None:
        self.transformers = transformers

    def apply_transformers(self, node: TFunction) -> TFunction:
        """Apply configured transformer pipeline to a single function node."""
        for transformer in self.transformers:
            node = transformer.visit(node)
        return ast.fix_missing_locations(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "metric":
                return self.apply_transformers(node)

            if isinstance(decorator, ast.Attribute):
                if (
                    isinstance(decorator.value, ast.Name)
                    and decorator.value.id == "merit"
                    and decorator.attr == "metric"
                ):
                    return self.apply_transformers(node)

            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Name) and func.id == "metric":
                    return self.apply_transformers(node)
                if isinstance(func, ast.Attribute):
                    if (
                        isinstance(func.value, ast.Name)
                        and func.value.id == "merit"
                        and func.attr == "metric"
                    ):
                        return self.apply_transformers(node)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "metric":
                return self.apply_transformers(node)

            if isinstance(decorator, ast.Attribute):
                if (
                    isinstance(decorator.value, ast.Name)
                    and decorator.value.id == "merit"
                    and decorator.attr == "metric"
                ):
                    return self.apply_transformers(node)

            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Name) and func.id == "metric":
                    return self.apply_transformers(node)
                if isinstance(func, ast.Attribute):
                    if (
                        isinstance(func.value, ast.Name)
                        and func.value.id == "merit"
                        and func.attr == "metric"
                    ):
                        return self.apply_transformers(node)
        return self.generic_visit(node)


class MeritModuleLoader(importlib.abc.SourceLoader):
    """Custom loader for Merit test modules with AST transformations.

    This loader participates in Python's import protocol and handles
    AST transformation and injection of Merit-specific globals during
    module execution.
    """

    def __init__(self, fullname: str, path: Path) -> None:
        """Initialize the loader.

        Args:
            fullname: The fully qualified module name.
            path: Path to the module file.
        """
        self.fullname = fullname
        self.path = path

    def get_filename(self, fullname: str) -> str:
        return str(self.path)

    def get_data(self, path: str) -> bytes:
        return Path(path).read_bytes()

    def exec_module(self, module: ModuleType) -> None:
        filename = self.get_filename(module.__name__)
        source = self.get_source(module.__name__)
        if source is None:
            msg = f"Cannot get source for module {module.__name__}"
            raise ImportError(msg)

        module_transformers = [
            MeritMetricTransformer(
                transformers=[InjectAssertionDependenciesTransformer(), AssertTransformer(source)]
            ),
            MeritFunctionTransformer(
                transformers=[InjectAssertionDependenciesTransformer(), AssertTransformer(source)]
            ),
        ]
        tree = ast.parse(source, filename=filename)
        for transformer in module_transformers:
            tree = transformer.visit(tree)
        validated_tree = ast.fix_missing_locations(tree)

        code = compile(validated_tree, filename=filename, mode="exec")
        exec(code, module.__dict__)
