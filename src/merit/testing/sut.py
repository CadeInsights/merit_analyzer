"""System-Under-Test (SUT) decorator for traced test targets.

The @sut decorator registers a callable as a traced resource that can be
injected into Merit tests. All invocations are wrapped in OpenTelemetry spans,
and any LLM calls made within are automatically captured as child spans.
"""

import inspect
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from merit.testing.resources import Scope, resource
from merit.tracing import get_tracer


P = ParamSpec("P")
T = TypeVar("T")


import re


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def sut(fn: Callable[P, T] | type | None = None, *, name: str | None = None) -> Any:
    """Register a callable as a traced system-under-test resource.

    Supports sync functions, async functions, and classes with __call__.
    Each invocation creates a span that captures input/output and nests
    LLM calls as children.

    For classes, the resource name is converted from CamelCase to snake_case
    (e.g., MyAgent -> my_agent) for consistent parameter naming.

    Args:
        fn: The callable to register as SUT.
        name: Optional name override (defaults to function/class name).

    Example:
        @sut
        async def my_agent(prompt: str) -> str:
            return await llm.generate(prompt)

        @sut
        class RAGPipeline:
            def __call__(self, query: str) -> str:
                return self.generate(self.retrieve(query))

        def merit_test(my_agent, rag_pipeline):
            result = my_agent("Hello")
            assert "world" in result.lower()
    """
    if fn is None:
        return lambda f: sut(f, name=name)

    if name:
        sut_name = name
    elif inspect.isclass(fn):
        sut_name = _to_snake_case(fn.__name__)
    else:
        sut_name = fn.__name__

    if inspect.isclass(fn):
        return _wrap_class(fn, sut_name)

    return _wrap_callable(fn, sut_name)


def _wrap_callable(fn: Callable[P, T], sut_name: str) -> Callable[[], Callable[P, T]]:
    """Wrap a function (sync or async) with tracing and register as resource.

    Returns a factory that creates the traced callable, so the resource
    system doesn't try to resolve the original function's parameters.
    """
    is_async = inspect.iscoroutinefunction(fn)

    if is_async:

        async def traced(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()
            with tracer.start_as_current_span(f"sut.{sut_name}") as span:
                _set_input_attrs(span, args, kwargs)
                result = await fn(*args, **kwargs)
                _set_output_attrs(span, result)
                return result

        def factory() -> Callable[P, T]:
            return traced  # type: ignore[return-value]

    else:

        def traced(*args: P.args, **kwargs: P.kwargs) -> T:  # type: ignore[misc]
            tracer = get_tracer()
            with tracer.start_as_current_span(f"sut.{sut_name}") as span:
                _set_input_attrs(span, args, kwargs)
                result = fn(*args, **kwargs)
                _set_output_attrs(span, result)
                return result

        def factory() -> Callable[P, T]:
            return traced  # type: ignore[return-value]

    factory.__name__ = sut_name
    factory.__doc__ = fn.__doc__
    return resource(factory, scope=Scope.SESSION)


def _wrap_class(cls: type, sut_name: str) -> Callable[[], Any]:
    """Wrap a class with __call__ and register instance as resource."""

    def factory() -> Any:
        instance = cls()
        original_call = instance.__call__
        is_async = inspect.iscoroutinefunction(original_call)

        if is_async:

            class TracedWrapper:
                def __init__(self, wrapped: Any):
                    self._wrapped = wrapped
                    self._original_call = wrapped.__call__

                async def __call__(self, *args: Any, **kwargs: Any) -> Any:
                    tracer = get_tracer()
                    with tracer.start_as_current_span(f"sut.{sut_name}") as span:
                        _set_input_attrs(span, args, kwargs)
                        result = await self._original_call(*args, **kwargs)
                        _set_output_attrs(span, result)
                        return result

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._wrapped, name)

            return TracedWrapper(instance)

        class TracedWrapper:
            def __init__(self, wrapped: Any):
                self._wrapped = wrapped
                self._original_call = wrapped.__call__

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                with tracer.start_as_current_span(f"sut.{sut_name}") as span:
                    _set_input_attrs(span, args, kwargs)
                    result = self._original_call(*args, **kwargs)
                    _set_output_attrs(span, result)
                    return result

            def __getattr__(self, name: str) -> Any:
                return getattr(self._wrapped, name)

        return TracedWrapper(instance)

    # Set name before registering so resource lookup works
    factory.__name__ = sut_name
    return resource(factory, scope=Scope.SESSION)


def _set_input_attrs(span: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Set input attributes on span, respecting trace content settings."""
    import os

    if os.environ.get("MERIT_TRACE_CONTENT", "true").lower() != "true":
        span.set_attribute("sut.input.count", len(args) + len(kwargs))
        return

    if args:
        span.set_attribute("sut.input.args", _truncate_repr(args))
    if kwargs:
        span.set_attribute("sut.input.kwargs", _truncate_repr(kwargs))


def _set_output_attrs(span: Any, result: Any) -> None:
    """Set output attributes on span, respecting trace content settings."""
    import os

    if os.environ.get("MERIT_TRACE_CONTENT", "true").lower() != "true":
        span.set_attribute("sut.output.type", type(result).__name__)
        return

    span.set_attribute("sut.output", _truncate_repr(result))


def _truncate_repr(value: Any, max_len: int = 1000) -> str:
    """Truncate a repr string if too long."""
    s = repr(value)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
