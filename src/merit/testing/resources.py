"""Resource system for dependency injection in tests.

Similar to pytest fixtures, resources provide injectable dependencies
to test functions based on parameter name matching.
"""

import inspect
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ParamSpec, TypeVar

from merit.context import ResolverContext, resolver_context_scope


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
    on_resolve: Callable[[Any], Any] | None = None
    on_injection: Callable[[Any], Any] | None = None
    on_teardown: Callable[[Any], Any] | None = None


_registry: dict[str, ResourceDef] = {}
_builtin_registry: dict[str, ResourceDef] = {}


def resource(
    fn: Callable[P, T] | None = None,
    *,
    scope: Scope | str = Scope.CASE,
    on_resolve: Callable[[Any], Any] | None = None,
    on_injection: Callable[[Any], Any] | None = None,
    on_teardown: Callable[[Any], Any] | None = None,
) -> Any:
    """Register a function as a resource for dependency injection.

    Args:
        fn: The resource factory function.
        scope: Lifecycle scope - "case", "suite", or "session".
        on_resolve: Callback invoked only when the resource is first created.
        on_injection: Callback invoked every time the resource is injected.
        on_teardown: Callback invoked after generator teardown (post-yield code) runs.

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

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
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
            on_resolve=on_resolve,
            on_injection=on_injection,
            on_teardown=on_teardown,
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
    _registry.update(_builtin_registry)


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
        self._teardowns: list[
            tuple[Scope, str, Generator[Any, None, None] | AsyncGenerator[Any, None]]
        ] = []
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

    def _register_teardown(
        self, scope: Scope, name: str, gen: Generator[Any, None, None] | AsyncGenerator[Any, None]
    ) -> None:
        """Register a teardown, delegating to parent for SUITE/SESSION scopes."""
        if scope in {Scope.SUITE, Scope.SESSION} and self._parent:
            self._parent._register_teardown(scope, name, gen)
        else:
            self._teardowns.append((scope, name, gen))

    async def resolve(self, name: str) -> Any:
        """Resolve a resource by name, including its dependencies."""
        if name not in self._registry:
            msg = f"Unknown resource: {name}"
            raise ValueError(msg)

        defn = self._registry[name]
        cache_key = (defn.scope, name)

        if cache_key in self._cache:
            value = self._cache[cache_key]

            # hook returns updated value on resource injection from cache
            if defn.on_injection:
                try:
                    value = defn.on_injection(value)
                    if inspect.iscoroutine(value):
                        value = await value
                    return value
                except Exception as e:
                    raise RuntimeError(
                        f"Hook {defn.on_injection.__name__} failed for resource '{name}': {e}"
                    ) from e
            return value

        # Resolve dependencies first
        kwargs = {}
        resolver_ctx = ResolverContext(
            consumer_name=name,
        )
        with resolver_context_scope(resolver_ctx):
            for dep in defn.dependencies:
                kwargs[dep] = await self.resolve(dep)

        # Call the factory
        if defn.is_async_generator:
            gen = defn.fn(**kwargs)
            value = await gen.__anext__()
            self._register_teardown(defn.scope, name, gen)
        elif defn.is_generator:
            gen = defn.fn(**kwargs)
            value = next(gen)
            self._register_teardown(defn.scope, name, gen)
        elif defn.is_async:
            value = await defn.fn(**kwargs)
        else:
            value = defn.fn(**kwargs)

        # hook returns updated value on resource creation
        if defn.on_resolve:
            try:
                value = defn.on_resolve(value)
                if inspect.iscoroutine(value):
                    value = await value
            except Exception as e:
                raise RuntimeError(
                    f"Hook {defn.on_resolve.__name__} failed for resource '{name}': {e}"
                ) from e

        self._cache[cache_key] = value
        # Sync cache to parent for SUITE/SESSION scopes
        if defn.scope in {Scope.SUITE, Scope.SESSION} and self._parent:
            self._parent._cache[cache_key] = value

        # hook returns updated value on resource injection
        if defn.on_injection:
            try:
                value = defn.on_injection(value)
                if inspect.iscoroutine(value):
                    value = await value
            except Exception as e:
                raise RuntimeError(
                    f"Hook {defn.on_injection.__name__} failed for resource '{name}': {e}"
                ) from e

        return value

    async def resolve_many(self, names: list[str]) -> dict[str, Any]:
        """Resolve multiple resources."""
        return {name: await self.resolve(name) for name in names}

    async def teardown(self) -> None:
        """Run teardown for all generator-based resources (LIFO order)."""
        for s, name, gen in reversed(self._teardowns):
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

            defn = self._registry.get(name)
            if defn and defn.on_teardown:
                cache_key = (s, name)
                if cache_key in self._cache:
                    try:
                        result = defn.on_teardown(self._cache[cache_key])
                        if inspect.iscoroutine(result):
                            await result
                    except Exception as e:
                        raise RuntimeError(
                            f"Hook {defn.on_teardown.__name__} failed for resource '{name}': {e}"
                        ) from e

        self._teardowns.clear()

    async def teardown_scope(self, scope: Scope) -> None:
        """Run teardown for resources in a specific scope and clear cache."""
        remaining = []
        for s, name, gen in reversed(self._teardowns):
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

                defn = self._registry.get(name)
                if defn and defn.on_teardown:
                    cache_key = (s, name)
                    if cache_key in self._cache:
                        try:
                            result = defn.on_teardown(self._cache[cache_key])
                            if inspect.iscoroutine(result):
                                await result
                        except Exception as e:
                            raise RuntimeError(
                                f"Hook {defn.on_teardown.__name__} failed for resource '{name}': {e}"
                            ) from e
            else:
                remaining.append((s, name, gen))

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
    if collector is None:
        raise RuntimeError("Tracing is not enabled; cannot resolve trace_context resource.")
    ctx = TraceContext.from_current(collector)
    yield ctx
    if collector:
        collector.clear(ctx.trace_id)


_builtin_registry["trace_context"] = _registry["trace_context"]

