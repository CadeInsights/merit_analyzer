"""Tests for merit.testing.sut module."""

import asyncio

import pytest

from merit.testing.resources import ResourceResolver, Scope, clear_registry, get_registry
from merit.testing.sut import sut
from merit.tracing import clear_traces, set_trace_output_path


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the global registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


@pytest.fixture(scope="module", autouse=True)
def setup_tracing_once(tmp_path_factory):
    """Initialize tracing once for all tests in this module."""
    tmp_dir = tmp_path_factory.mktemp("traces")
    output_path = tmp_dir / "sut_traces.jsonl"
    set_trace_output_path(output_path=str(output_path))


@pytest.fixture(autouse=True)
def clear_traces_each(tmp_path):
    """Point tracing to a temp file and clear before/after each test."""
    output_path = tmp_path / "traces.jsonl"
    set_trace_output_path(output_path)

    clear_traces()
    yield
    clear_traces()


class TestSutDecorator:
    """Tests for the @sut decorator."""

    def test_registers_sync_function(self):
        @sut
        def my_sut(x: int) -> int:
            return x * 2

        registry = get_registry()
        assert "my_sut" in registry
        defn = registry["my_sut"]
        assert defn.scope == Scope.SESSION
        assert defn.dependencies == []

    def test_registers_async_function(self):
        @sut
        async def async_sut(x: int) -> int:
            return x * 2

        registry = get_registry()
        assert "async_sut" in registry
        defn = registry["async_sut"]
        assert defn.scope == Scope.SESSION

    def test_registers_class_with_snake_case_name(self):
        @sut
        class MyTestAgent:
            def __call__(self, x: int) -> int:
                return x * 2

        registry = get_registry()
        assert "my_test_agent" in registry
        defn = registry["my_test_agent"]
        assert defn.scope == Scope.SESSION

    def test_custom_name_override(self):
        @sut(name="custom_name")
        def original_name(x: int) -> int:
            return x

        registry = get_registry()
        assert "custom_name" in registry
        assert "original_name" not in registry

    def test_class_custom_name(self):
        @sut(name="custom_class")
        class OriginalClass:
            def __call__(self) -> str:
                return "result"

        registry = get_registry()
        assert "custom_class" in registry


class TestSutResolution:
    """Tests for resolving SUT resources."""

    @pytest.mark.asyncio
    async def test_resolves_sync_sut(self):
        @sut
        def adder(x: int, y: int) -> int:
            return x + y

        resolver = ResourceResolver(get_registry())
        resolved = await resolver.resolve("adder")

        assert callable(resolved)
        assert resolved(2, 3) == 5

    @pytest.mark.asyncio
    async def test_resolves_async_sut(self):
        @sut
        async def async_adder(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x + y

        resolver = ResourceResolver(get_registry())
        resolved = await resolver.resolve("async_adder")

        assert callable(resolved)
        result = await resolved(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_resolves_class_sut(self):
        @sut
        class Multiplier:
            def __init__(self):
                self.factor = 3

            def __call__(self, x: int) -> int:
                return x * self.factor

        resolver = ResourceResolver(get_registry())
        resolved = await resolver.resolve("multiplier")

        assert callable(resolved)
        assert resolved(4) == 12

    @pytest.mark.asyncio
    async def test_session_scope_shared(self):
        instance_count = 0

        @sut
        class SharedInstance:
            def __init__(self):
                nonlocal instance_count
                instance_count += 1

            def __call__(self) -> int:
                return instance_count

        resolver = ResourceResolver(get_registry())

        # Resolve twice - should return same instance
        result1 = await resolver.resolve("shared_instance")
        result2 = await resolver.resolve("shared_instance")

        # Factory called once due to session scope
        assert instance_count == 1
        assert result1 is result2


class TestSutTracing:
    """Tests for SUT tracing functionality."""

    @pytest.mark.asyncio
    async def test_sync_sut_creates_span(self, tmp_path):
        import json

        @sut
        def traced_sut(x: int) -> int:
            return x * 2

        resolver = ResourceResolver(get_registry())
        resolved = await resolver.resolve("traced_sut")
        resolved(5)

        from merit.tracing.lifecycle import _exporter

        assert _exporter is not None
        output = _exporter.output_path
        lines = output.read_text().strip().splitlines()
        assert len(lines) >= 1

        span_names = [json.loads(line)["name"] for line in lines]
        assert "sut.traced_sut" in span_names

    @pytest.mark.asyncio
    async def test_async_sut_creates_span(self, tmp_path):
        import json

        @sut
        async def async_traced(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        resolver = ResourceResolver(get_registry())
        resolved = await resolver.resolve("async_traced")
        await resolved(5)

        from merit.tracing.lifecycle import _exporter

        assert _exporter is not None
        output = _exporter.output_path
        lines = output.read_text().strip().splitlines()
        assert len(lines) >= 1

        span_names = [json.loads(line)["name"] for line in lines]
        assert "sut.async_traced" in span_names

    @pytest.mark.asyncio
    async def test_class_sut_creates_span(self, tmp_path):
        import json

        @sut
        class TracedPipeline:
            def __call__(self, query: str) -> str:
                return f"Result: {query}"

        resolver = ResourceResolver(get_registry())
        resolved = await resolver.resolve("traced_pipeline")
        resolved("test query")

        from merit.tracing.lifecycle import _exporter

        assert _exporter is not None
        output = _exporter.output_path
        lines = output.read_text().strip().splitlines()
        assert len(lines) >= 1

        span_names = [json.loads(line)["name"] for line in lines]
        assert "sut.traced_pipeline" in span_names


class TestSnakeCaseConversion:
    """Tests for CamelCase to snake_case conversion."""

    def test_simple_camel_case(self):
        @sut
        class MyAgent:
            def __call__(self) -> str:
                return "result"

        assert "my_agent" in get_registry()

    def test_consecutive_caps(self):
        @sut
        class HTTPClient:
            def __call__(self) -> str:
                return "result"

        assert "http_client" in get_registry()

    def test_already_snake_case(self):
        @sut
        def already_snake() -> str:
            return "result"

        assert "already_snake" in get_registry()

    def test_single_word(self):
        @sut
        class Agent:
            def __call__(self) -> str:
                return "result"

        assert "agent" in get_registry()

    @pytest.mark.asyncio
    async def test_sut_spans_have_merit_attribute(self):
        """Verify that SUT spans have the merit.sut attribute set."""
        import json

        @sut
        def my_test_function(x: int) -> int:
            return x

        resolver = ResourceResolver(get_registry())
        resolved = await resolver.resolve("my_test_function")
        resolved(1)

        from merit.tracing.lifecycle import _exporter
        assert _exporter is not None
        lines = _exporter.output_path.read_text().strip().splitlines()
        
        span = None
        for line in lines:
            s = json.loads(line)
            if s["name"] == "sut.my_test_function":
                span = s
                break
        
        assert span is not None, "SUT span not found"
        attributes = span["attributes"]
        assert attributes.get("merit.sut") is True
        assert attributes.get("merit.sut.name") == "my_test_function"
