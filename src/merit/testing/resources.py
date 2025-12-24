"""Resource system for dependency injection in tests.

Similar to pytest fixtures, resources provide injectable dependencies
to test functions based on parameter name matching.
"""

import inspect
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ParamSpec, TypeVar


P = ParamSpec("P")
T = TypeVar("T")


class Scope(Enum):
    """Resource lifecycle scope."""

    CASE = "case"  # Fresh instance per test
    SUITE = "suite"  # Shared across tests in same file
    SESSION = "session"  # Shared across entire test run


@dataclass
class ResourceDef:
    """Definition of a registered resource."""

    name: str
    fn: Callable[..., Any]
    scope: Scope
    is_async: bool
    is_generator: bool
    is_async_generator: bool
    dependencies: list[str] = field(default_factory=list)


_registry: dict[str, ResourceDef] = {}


def resource(
    fn: Callable[P, T] | None = None,
    *,
    scope: Scope | str = Scope.CASE,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Register a function as a resource for dependency injection.

    Args:
        fn: The resource factory function.
        scope: Lifecycle scope - "case", "suite", or "session".

    Example:
        @resource
        def api_client():
            return APIClient()

        @resource(scope="suite")
        async def db_connection():
            conn = await connect()
            yield conn
            await conn.close()
    """

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        nonlocal scope
        if isinstance(scope, str):
            scope = Scope(scope)

        sig = inspect.signature(fn)
        deps = [p for p in sig.parameters if p != "self"]

        is_async = inspect.iscoroutinefunction(fn)
        is_async_gen = inspect.isasyncgenfunction(fn)
        is_gen = inspect.isgeneratorfunction(fn)

        defn = ResourceDef(
            name=fn.__name__,
            fn=fn,
            scope=scope,
            is_async=is_async or is_async_gen,
            is_generator=is_gen,
            is_async_generator=is_async_gen,
            dependencies=deps,
        )
        _registry[defn.name] = defn
        return fn

    if fn is not None:
        return decorator(fn)
    return decorator


def get_registry() -> dict[str, ResourceDef]:
    """Get the global resource registry."""
    return _registry


def clear_registry() -> None:
    """Clear all registered resources."""
    _registry.clear()


class ResourceResolver:
    """Resolves and caches resources for test execution."""

    def __init__(
        self,
        registry: dict[str, ResourceDef] | None = None,
        *,
        parent: "ResourceResolver | None" = None,
    ) -> None:
        self._registry = registry if registry is not None else _registry
        self._cache: dict[tuple[Scope, str], Any] = {}
        self._teardowns: list[tuple[Scope, Generator[Any, None, None] | AsyncGenerator[Any, None]]] = []
        self._parent = parent

    def fork_for_case(self) -> "ResourceResolver":
        """Create a child resolver for isolated CASE-scope execution.

        Shares SUITE/SESSION cache with parent. SUITE/SESSION teardowns
        are registered with the parent to ensure proper cleanup.
        """
        child = ResourceResolver(self._registry, parent=self)
        # Share higher-scope cached values
        for key, value in self._cache.items():
            if key[0] in {Scope.SUITE, Scope.SESSION}:
                child._cache[key] = value
        return child

    def _register_teardown(self, scope: Scope, gen: Generator[Any, None, None] | AsyncGenerator[Any, None]) -> None:
        """Register a teardown, delegating to parent for SUITE/SESSION scopes."""
        if scope in {Scope.SUITE, Scope.SESSION} and self._parent:
            self._parent._register_teardown(scope, gen)
        else:
            self._teardowns.append((scope, gen))

    async def resolve(self, name: str) -> Any:
        """Resolve a resource by name, including its dependencies."""
        if name not in self._registry:
            msg = f"Unknown resource: {name}"
            raise ValueError(msg)

        defn = self._registry[name]
        cache_key = (defn.scope, name)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Resolve dependencies first
        kwargs = {}
        for dep in defn.dependencies:
            kwargs[dep] = await self.resolve(dep)

        # Call the factory
        if defn.is_async_generator:
            gen = defn.fn(**kwargs)
            value = await gen.__anext__()
            self._register_teardown(defn.scope, gen)
        elif defn.is_generator:
            gen = defn.fn(**kwargs)
            value = next(gen)
            self._register_teardown(defn.scope, gen)
        elif defn.is_async:
            value = await defn.fn(**kwargs)
        else:
            value = defn.fn(**kwargs)

        self._cache[cache_key] = value
        # Sync cache to parent for SUITE/SESSION scopes
        if defn.scope in {Scope.SUITE, Scope.SESSION} and self._parent:
            self._parent._cache[cache_key] = value
        return value

    async def resolve_many(self, names: list[str]) -> dict[str, Any]:
        """Resolve multiple resources."""
        return {name: await self.resolve(name) for name in names}

    async def teardown(self) -> None:
        """Run teardown for all generator-based resources (LIFO order)."""
        for _, gen in reversed(self._teardowns):
            if isinstance(gen, AsyncGenerator):
                try:
                    await gen.__anext__()
                except StopAsyncIteration:
                    pass
            else:
                try:
                    next(gen)
                except StopIteration:
                    pass
        self._teardowns.clear()

    async def teardown_scope(self, scope: Scope) -> None:
        """Run teardown for resources in a specific scope and clear cache."""
        remaining = []
        for s, gen in reversed(self._teardowns):
            if s == scope:
                if isinstance(gen, AsyncGenerator):
                    try:
                        await gen.__anext__()
                    except StopAsyncIteration:
                        pass
                else:
                    try:
                        next(gen)
                    except StopIteration:
                        pass
            else:
                remaining.append((s, gen))

        self._teardowns = list(reversed(remaining))

        keys_to_remove = [k for k in self._cache if k[0] == scope]
        for key in keys_to_remove:
            del self._cache[key]


# Built-in resources


@resource(scope=Scope.CASE)
def trace_context():
    """Provide access to trace data for the current test.

    Yields a TraceContext that allows querying child spans, LLM calls,
    and setting custom attributes on the test span.

    Automatically clears captured spans on teardown.
    """
    from merit.tracing import TraceContext, get_span_collector

    collector = get_span_collector()
    ctx = TraceContext.from_current(collector)
    yield ctx
    if collector:
        collector.clear(ctx.trace_id)
