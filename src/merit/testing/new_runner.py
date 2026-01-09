"""New test runner with execution tree and worker pool pattern."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from merit.context import (
    ResolverContext,
    resolver_context_scope,
    node_context_scope,
)
from merit.testing.resources import ResourceResolver, Scope, get_registry

if TYPE_CHECKING:
    from merit.testing.runner import MeritRun


class TestStatus(Enum):
    """Test execution status with tree-aware states."""

    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    XFAILED = "xfailed"
    XPASSED = "xpassed"


@dataclass
class ExecutionBody:
    """Everything needed to run a single test."""

    fn: Callable[..., Any]
    param_names: list[str]
    param_values: dict[str, Any]
    is_async: bool
    class_name: str | None = None
    skip_reason: str | None = None
    xfail_reason: str | None = None
    xfail_strict: bool = False


@dataclass
class RunContext:
    """Shared context for test execution workers."""

    queue: asyncio.Queue[TestNode | None]
    resolver: ResourceResolver
    merit_run: MeritRun


@dataclass
class TestNode:
    """A node in the execution tree.
    
    Leaf nodes have a body and are executed by workers.
    Container nodes have children and aggregate their status.
    """

    name: str
    full_name: str
    module_path: Path
    body: ExecutionBody | None = None
    children: list[TestNode] = field(default_factory=list)
    status: TestStatus = TestStatus.IDLE
    duration_ms: float = 0
    error: Exception | None = None
    tags: set[str] = field(default_factory=set)
    
    # For repeat aggregation
    min_passes: int | None = None
    
    # Internal: event for leaf nodes to signal completion
    _completion_event: asyncio.Event | None = field(default=None, repr=False)

    async def execute(self, run_context: RunContext) -> None:
        """Execute this node. Leaf nodes queue for workers, containers run children."""
        self.status = TestStatus.IN_PROGRESS
        start = time.perf_counter()
        
        with node_context_scope(self):
            if self.body:
                # Leaf node: queue for worker execution
                self._completion_event = asyncio.Event()
                await run_context.queue.put(self)
                await self._completion_event.wait()
            elif self.children:
                # Container: execute all children concurrently
                await asyncio.gather(*[child.execute(run_context) for child in self.children])
                self._aggregate_status()
        
        self.duration_ms = (time.perf_counter() - start) * 1000

    def mark_complete(self, status: TestStatus, error: Exception | None = None) -> None:
        """Mark this leaf node as complete (called by worker)."""
        self.status = status
        self.error = error
        if self._completion_event:
            self._completion_event.set()

    def _aggregate_status(self) -> None:
        """Aggregate status from children based on node type."""
        if not self.children:
            return
        
        # Repeat parent: check min_passes threshold
        if self.min_passes is not None:
            passed_count = sum(1 for c in self.children if c.status == TestStatus.PASSED)
            self.status = TestStatus.PASSED if passed_count >= self.min_passes else TestStatus.FAILED
            return
        
        # Parametrize/Class/Module parent: any failure means failure
        for child in self.children:
            if child.status in {TestStatus.FAILED, TestStatus.ERROR}:
                self.status = TestStatus.FAILED
                return
        
        # All children passed (or skipped/xfailed)
        self.status = TestStatus.PASSED


class Runner:
    """Executes test nodes using a worker pool pattern."""

    DEFAULT_MAX_CONCURRENCY = 10

    def __init__(
        self,
        *,
        concurrency: int = 1,
        timeout: float | None = None,
    ) -> None:
        self.concurrency = concurrency if concurrency > 0 else self.DEFAULT_MAX_CONCURRENCY
        self.timeout = timeout

    async def run(self, root_nodes: list[TestNode], merit_run: MeritRun) -> MeritRun:
        """Run all test nodes and return the completed MeritRun."""
        run_context = RunContext(
            queue=asyncio.Queue(),
            resolver=ResourceResolver(get_registry()),
            merit_run=merit_run,
        )
        
        # Start worker pool
        workers = [
            asyncio.create_task(self._worker(run_context))
            for _ in range(self.concurrency)
        ]
        
        # Execute all root nodes (they'll queue leaf nodes for workers)
        await asyncio.gather(*[node.execute(run_context) for node in root_nodes])
        
        # Signal workers to stop
        for _ in workers:
            await run_context.queue.put(None)
        await asyncio.gather(*workers)
        
        # Teardown resources
        await run_context.resolver.teardown()
        
        return merit_run

    async def _worker(self, run_context: RunContext) -> None:
        """Worker that processes queued leaf nodes."""
        while True:
            node = await run_context.queue.get()
            if node is None:
                break
            
            await self._execute_leaf(node, run_context)

    async def _execute_leaf(self, node: TestNode, run_context: RunContext) -> None:
        """Execute a single leaf node's body."""
        body = node.body
        if not body:
            node.mark_complete(TestStatus.ERROR, RuntimeError("No body to execute"))
            return

        # IMPORTANT: fork a per-test resolver so Scope.CASE is isolated per leaf execution.
        # This matches the behavior of the original runner's concurrent path.
        resolver = run_context.resolver.fork_for_case()
        
        start = time.perf_counter()
        
        # Handle skip
        if body.skip_reason:
            node.duration_ms = (time.perf_counter() - start) * 1000
            node.mark_complete(TestStatus.SKIPPED, AssertionError(body.skip_reason))
            return
        
        expect_failure = body.xfail_reason is not None
        
        # Prepare kwargs from param_values and resolve resources
        kwargs = dict(body.param_values)
        
        resolver_ctx = ResolverContext(consumer_name=node.name)
        with resolver_context_scope(resolver_ctx):
            for param in body.param_names:
                if param in kwargs:
                    continue
                try:
                    kwargs[param] = await resolver.resolve(param)
                except ValueError:
                    pass  # Not a resource
        
        # Instantiate class if method
        instance = None
        if body.class_name:
            cls = body.fn.__globals__.get(body.class_name)
            if cls:
                instance = cls()
        
        # Execute
        try:
            if instance:
                if body.is_async:
                    await body.fn(instance, **kwargs)
                else:
                    body.fn(instance, **kwargs)
            elif body.is_async:
                await body.fn(**kwargs)
            else:
                body.fn(**kwargs)
            
            node.duration_ms = (time.perf_counter() - start) * 1000
            
            if expect_failure:
                if body.xfail_strict:
                    node.mark_complete(TestStatus.FAILED, AssertionError(body.xfail_reason or "xfail marked test passed"))
                else:
                    node.mark_complete(TestStatus.XPASSED)
            else:
                node.mark_complete(TestStatus.PASSED)
                
        except AssertionError as e:
            node.duration_ms = (time.perf_counter() - start) * 1000
            if expect_failure:
                node.mark_complete(TestStatus.XFAILED, AssertionError(body.xfail_reason) if body.xfail_reason else e)
            else:
                node.mark_complete(TestStatus.FAILED, e)
                
        except Exception as e:
            node.duration_ms = (time.perf_counter() - start) * 1000
            if expect_failure:
                node.mark_complete(TestStatus.XFAILED, AssertionError(body.xfail_reason) if body.xfail_reason else e)
            else:
                node.mark_complete(TestStatus.ERROR, e)
        
        finally:
            await resolver.teardown_scope(Scope.CASE)
