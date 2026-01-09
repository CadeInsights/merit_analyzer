"""Tree-building discovery for the new execution model."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from merit.testing.discovery import _extract_test_params, _load_module
from merit.testing.new_runner import ExecutionBody, TestNode
from merit.testing.parametrize import get_parameter_sets
from merit.testing.repeat import get_repeat_data
from merit.testing.tags import TagData, get_tag_data, merge_tag_data


def _build_node_tree(
    fn: Callable[..., Any],
    name: str,
    module_path: Path,
    class_name: str | None = None,
    parent_tags: TagData | None = None,
) -> TestNode:
    """Build a TestNode tree for a callable, handling parametrize and repeat."""
    combined_tags = merge_tag_data(parent_tags, get_tag_data(fn))
    repeat_data = get_repeat_data(fn)
    parameter_sets = get_parameter_sets(fn)
    
    is_async = inspect.iscoroutinefunction(fn)
    param_names = _extract_test_params(fn)
    
    # Base full_name
    if class_name:
        base_full_name = f"{module_path.stem}::{class_name}::{name}"
    else:
        base_full_name = f"{module_path.stem}::{name}"
    
    def make_leaf(
        full_name: str,
        param_values: dict[str, Any],
        id_suffix: str | None = None,
    ) -> TestNode:
        """Create a leaf node with ExecutionBody."""
        display_name = f"{full_name}[{id_suffix}]" if id_suffix else full_name
        return TestNode(
            name=name,
            full_name=display_name,
            module_path=module_path,
            tags=set(combined_tags.tags),
            body=ExecutionBody(
                fn=fn,
                param_names=param_names,
                param_values=param_values,
                is_async=is_async,
                class_name=class_name,
                skip_reason=combined_tags.skip_reason,
                xfail_reason=combined_tags.xfail_reason,
                xfail_strict=combined_tags.xfail_strict,
            ),
        )
    
    def wrap_with_repeat(node: TestNode, full_name: str) -> TestNode:
        """Wrap a node with repeat children if repeat_data is set."""
        if not repeat_data or repeat_data.count <= 1:
            return node
        
        # Create N copies of the node as children
        children = []
        for i in range(repeat_data.count):
            child = TestNode(
                name=node.name,
                full_name=f"{node.full_name}#run{i+1}",
                module_path=node.module_path,
                tags=set(node.tags),
                body=ExecutionBody(
                    fn=fn,
                    param_names=node.body.param_names if node.body else param_names,
                    param_values=node.body.param_values if node.body else {},
                    is_async=is_async,
                    class_name=class_name,
                    skip_reason=combined_tags.skip_reason,
                    xfail_reason=combined_tags.xfail_reason,
                    xfail_strict=combined_tags.xfail_strict,
                ),
            )
            children.append(child)
        
        # Parent node aggregates children
        return TestNode(
            name=name,
            full_name=full_name,
            module_path=module_path,
            tags=set(combined_tags.tags),
            children=children,
            min_passes=repeat_data.min_passes,
        )
    
    # No parametrization: single node (possibly wrapped with repeat)
    if not parameter_sets:
        leaf = make_leaf(base_full_name, {})
        return wrap_with_repeat(leaf, base_full_name)
    
    # With parametrization: create parent with children per param combo
    children = []
    for param_set in parameter_sets:
        child_full_name = f"{base_full_name}[{param_set.id_suffix}]"
        child = make_leaf(child_full_name, param_set.values, param_set.id_suffix)
        child = wrap_with_repeat(child, child_full_name)
        children.append(child)
    
    return TestNode(
        name=name,
        full_name=base_full_name,
        module_path=module_path,
        tags=set(combined_tags.tags),
        children=children,
    )


def _collect_from_module(module: ModuleType, module_path: Path) -> list[TestNode]:
    """Collect test nodes from a module, creating class containers as needed."""
    nodes: list[TestNode] = []
    
    for name, obj in inspect.getmembers(module):
        # Collect merit_* functions as direct children
        if name.startswith("merit_") and inspect.isfunction(obj):
            nodes.append(_build_node_tree(obj, name, module_path))
        
        # Collect Merit* classes with merit_* methods
        elif name.startswith("Merit") and inspect.isclass(obj):
            class_tags = get_tag_data(obj)
            class_children = []
            
            for method_name, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                if method_name.startswith("merit_"):
                    child = _build_node_tree(
                        method,
                        method_name,
                        module_path,
                        class_name=name,
                        parent_tags=class_tags,
                    )
                    class_children.append(child)
            
            if class_children:
                class_node = TestNode(
                    name=name,
                    full_name=f"{module_path.stem}::{name}",
                    module_path=module_path,
                    children=class_children,
                )
                nodes.append(class_node)
    
    return nodes


def collect(path: Path | str | None = None) -> list[TestNode]:
    """Discover all merit_* tests and return root nodes (one per module).
    
    Args:
        path: File or directory to search. Defaults to current directory.
    
    Returns:
        List of TestNode objects representing module-level containers.
    """
    if path is None:
        path = Path.cwd()
    elif isinstance(path, str):
        path = Path(path)
    
    path = path.resolve()
    nodes: list[TestNode] = []
    
    if path.is_file():
        if path.name.startswith("merit_") and path.suffix == ".py":
            module = _load_module(path)
            module_children = _collect_from_module(module, path)
            if module_children:
                module_node = TestNode(
                    name=path.stem,
                    full_name=path.stem,
                    module_path=path,
                    children=module_children,
                )
                nodes.append(module_node)
    elif path.is_dir():
        for file_path in path.rglob("merit_*.py"):
            module = _load_module(file_path)
            module_children = _collect_from_module(module, file_path)
            if module_children:
                module_node = TestNode(
                    name=file_path.stem,
                    full_name=file_path.stem,
                    module_path=file_path,
                    children=module_children,
                )
                nodes.append(module_node)
    
    return nodes
