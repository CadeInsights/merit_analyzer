"""New test runner with execution tree and worker pool pattern."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

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


RUN_CONTEXT: ContextVar[RunContext | None] = ContextVar("run_context", default=None)


@contextmanager
def run_context_scope(ctx: RunContext) -> Iterator[None]:
    token = RUN_CONTEXT.set(ctx)
    try:
        yield
    finally:
        RUN_CONTEXT.reset(token)


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

    async def execute(self, *, _worker: bool = False) -> None:
        """Execute this node.

        - Container nodes run children and aggregate their status.
        - Leaf nodes are queued for workers when called from the scheduler.
        - Leaf nodes run their body when called from a worker.
        """
        run_context = RUN_CONTEXT.get()
        assert run_context is not None

        # Leaf node: scheduler path (queue for worker and wait)
        if self.body and not _worker:
            self.status = TestStatus.IN_PROGRESS
            self._completion_event = asyncio.Event()
            await run_context.queue.put(self)
            await self._completion_event.wait()
            return

        start = time.perf_counter()

        with node_context_scope(self):
            # Leaf node: worker path (run body)
            if self.body and _worker:
                body = self.body
                resolver = run_context.resolver.fork_for_case()

                # Handle skip
                if body.skip_reason:
                    self.duration_ms = (time.perf_counter() - start) * 1000
                    self.mark_complete(TestStatus.SKIPPED, AssertionError(body.skip_reason))
                    await resolver.teardown_scope(Scope.CASE)
                    return

                expect_failure = body.xfail_reason is not None

                # Prepare kwargs from param_values and resolve resources
                kwargs = dict(body.param_values)

                resolver_ctx = ResolverContext(consumer_name=self.name)
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

                    self.duration_ms = (time.perf_counter() - start) * 1000

                    if expect_failure:
                        if body.xfail_strict:
                            self.mark_complete(
                                TestStatus.FAILED,
                                AssertionError(
                                    body.xfail_reason or "xfail marked test passed"
                                ),
                            )
                        else:
                            self.mark_complete(TestStatus.XPASSED)
                    else:
                        self.mark_complete(TestStatus.PASSED)

                except AssertionError as e:
                    self.duration_ms = (time.perf_counter() - start) * 1000
                    if expect_failure:
                        self.mark_complete(
                            TestStatus.XFAILED,
                            AssertionError(body.xfail_reason) if body.xfail_reason else e,
                        )
                    else:
                        self.mark_complete(TestStatus.FAILED, e)

                except Exception as e:
                    self.duration_ms = (time.perf_counter() - start) * 1000
                    if expect_failure:
                        self.mark_complete(
                            TestStatus.XFAILED,
                            AssertionError(body.xfail_reason) if body.xfail_reason else e,
                        )
                    else:
                        self.mark_complete(TestStatus.ERROR, e)

                finally:
                    await resolver.teardown_scope(Scope.CASE)

            # Container: execute all children concurrently
            elif self.children:
                self.status = TestStatus.IN_PROGRESS
                await asyncio.gather(*[child.execute() for child in self.children])
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

        with run_context_scope(run_context):
            # Start worker pool (tasks inherit current contextvars)
            workers = [
                asyncio.create_task(self._worker())
                for _ in range(self.concurrency)
            ]

            # Execute all root nodes (they'll queue leaf nodes for workers)
            await asyncio.gather(*[node.execute() for node in root_nodes])

            # Signal workers to stop
            for _ in workers:
                await run_context.queue.put(None)
            await asyncio.gather(*workers)

            # Teardown resources
            await run_context.resolver.teardown()
        
        return merit_run

    async def _worker(self) -> None:
        """Worker that processes queued leaf nodes."""
        run_context = RUN_CONTEXT.get()
        assert run_context is not None

        while True:
            node = await run_context.queue.get()
            if node is None:
                break

            await node.execute(_worker=True)
