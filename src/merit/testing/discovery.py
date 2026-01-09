"""Test discovery for merit_* files and functions."""

import importlib.util
import importlib.abc
import inspect
import sys
import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

from merit.assertions.transformers import AssertTransformer, InjectAssertionDependenciesTransformer
from merit.testing.parametrize import get_parameter_sets
from merit.testing.repeat import get_repeat_data
from merit.testing.tags import TagData, get_tag_data, merge_tag_data


TFunction = TypeVar("TFunction", ast.FunctionDef, ast.AsyncFunctionDef)


class MeritFunctionTransformer(ast.NodeTransformer):
    """Finds all functions in the module that start with `merit_` and transforms them."""

    def __init__(
        self,
        transformers: list[ast.NodeTransformer]
    ) -> None:
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
        
        transformer = MeritFunctionTransformer(
            transformers=[
                InjectAssertionDependenciesTransformer(),
                AssertTransformer(source),
            ]
        )
        tree = ast.parse(source, filename=filename)
        transformed_tree = transformer.visit(tree)
        validated_tree = ast.fix_missing_locations(transformed_tree)

        code = compile(validated_tree, filename=filename, mode="exec")
        exec(code, module.__dict__) 


@dataclass
class TestItem:
    """A discovered test function or method."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    name: str
    fn: Callable[..., Any]
    module_path: Path
    is_async: bool
    params: list[str] = field(default_factory=list)
    class_name: str | None = None
    param_values: dict[str, Any] | None = None
    id_suffix: str | None = None
    tags: set[str] = field(default_factory=set)
    skip_reason: str | None = None
    xfail_reason: str | None = None
    xfail_strict: bool = False
    fail_fast: bool = False
    repeat_count: int = 1
    repeat_min_passes: int | None = None

    @property
    def full_name(self) -> str:
        """Full qualified name for display."""
        if self.class_name:
            base = f"{self.module_path.stem}::{self.class_name}::{self.name}"
        else:
            base = f"{self.module_path.stem}::{self.name}"
        if self.id_suffix:
            return f"{base}[{self.id_suffix}]"
        return base


def _load_module(path: Path) -> ModuleType:
    """Dynamically load a Python module from path."""
    spec = importlib.util.spec_from_file_location(
        path.stem,
        path,
        loader=MeritModuleLoader(fullname=path.stem, path=path),
    )
    if spec is None or spec.loader is None:
        msg = f"Cannot load module from {path}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _extract_test_params(fn: Callable[..., Any]) -> list[str]:
    """Extract parameter names from function signature (excluding 'self')."""
    sig = inspect.signature(fn)
    return [p for p in sig.parameters if p != "self"]


def _collect_from_module(module: ModuleType, module_path: Path) -> list[TestItem]:
    """Collect all merit_* tests from a module."""
    items: list[TestItem] = []

    for name, obj in inspect.getmembers(module):
        # Collect merit_* functions
        if name.startswith("merit_") and inspect.isfunction(obj):
            items.extend(_build_items_for_callable(obj, name, module_path))

        # Collect Merit* classes with merit_* methods
        elif name.startswith("Merit") and inspect.isclass(obj):
            class_tags = get_tag_data(obj)
            for method_name, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                if method_name.startswith("merit_"):
                    items.extend(
                        _build_items_for_callable(
                            method,
                            method_name,
                            module_path,
                            class_name=name,
                            parent_tags=class_tags,
                        )
                    )

    return items


def _build_items_for_callable(
    fn: Callable[..., Any],
    name: str,
    module_path: Path,
    class_name: str | None = None,
    parent_tags: TagData | None = None,
) -> list[TestItem]:
    """Create TestItems for a callable, expanding parametrizations if present."""
    combined_tags = merge_tag_data(parent_tags, get_tag_data(fn))
    repeat_data = get_repeat_data(fn)

    base_kwargs: dict[str, Any] = {
        "name": name,
        "fn": fn,
        "module_path": module_path,
        "is_async": inspect.iscoroutinefunction(fn),
        "params": _extract_test_params(fn),
        "class_name": class_name,
        "skip_reason": combined_tags.skip_reason,
        "xfail_reason": combined_tags.xfail_reason,
        "xfail_strict": combined_tags.xfail_strict,
        "repeat_count": repeat_data.count if repeat_data else 1,
        "repeat_min_passes": repeat_data.min_passes if repeat_data else None,
    }

    parameter_sets = get_parameter_sets(fn)
    if not parameter_sets:
        return [
            TestItem(
                **base_kwargs,
                tags=set(combined_tags.tags),
            )
        ]

    expanded: list[TestItem] = []
    for param_set in parameter_sets:
        expanded.append(
            TestItem(
                **base_kwargs,
                tags=set(combined_tags.tags),
                param_values=param_set.values,
                id_suffix=param_set.id_suffix,
            )
        )
    return expanded


def collect(path: Path | str | None = None) -> list[TestItem]:
    """Discover all merit_* tests from path.

    Args:
        path: File or directory to search. Defaults to current directory.

    Returns:
        List of discovered TestItem objects.

    Example:
        items = collect()  # Current directory
        items = collect("merit_agents.py")  # Specific file
        items = collect("./tests/")  # Directory
    """
    if path is None:
        path = Path.cwd()
    elif isinstance(path, str):
        path = Path(path)

    path = path.resolve()
    items: list[TestItem] = []

    if path.is_file():
        if path.name.startswith("merit_") and path.suffix == ".py":
            module = _load_module(path)
            items.extend(_collect_from_module(module, path))
    elif path.is_dir():
        for file_path in path.rglob("merit_*.py"):
            module = _load_module(file_path)
            items.extend(_collect_from_module(module, file_path))

    return items
