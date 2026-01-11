"""Tests for merit.testing.resources module."""

from pathlib import Path

import pytest

from merit.context import (
    RESOLVER_CONTEXT,
    TEST_CONTEXT,
    TestContext as Ctx,
    test_context_scope as context_scope,
)
from merit.testing.discovery import TestItem
from merit.testing.resources import (
    ResourceResolver,
    Scope,
    clear_registry,
    get_registry,
    resource,
)


def _make_item(name: str = "merit_fn", id_suffix: str | None = None) -> TestItem:
    """Create a minimal TestItem for testing."""
    return TestItem(
        name=name,
        fn=lambda: None,
        module_path=Path("test.py"),
        is_async=False,
        id_suffix=id_suffix,
    )


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the global registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


class TestResourceDecorator:
    """Tests for the @resource decorator."""

    def test_registers_sync_function(self):
        @resource
        def my_resource():
            return "value"

        registry = get_registry()
        assert "my_resource" in registry
        defn = registry["my_resource"]
        assert defn.name == "my_resource"
        assert defn.scope == Scope.CASE
        assert not defn.is_async
        assert not defn.is_generator

    def test_registers_async_function(self):
        @resource
        async def async_resource():
            return "async_value"

        defn = get_registry()["async_resource"]
        assert defn.is_async
        assert not defn.is_generator

    def test_registers_sync_generator(self):
        @resource
        def gen_resource():
            yield "gen_value"

        defn = get_registry()["gen_resource"]
        assert defn.is_generator
        assert not defn.is_async

    def test_registers_async_generator(self):
        @resource
        async def async_gen_resource():
            yield "async_gen_value"

        defn = get_registry()["async_gen_resource"]
        assert defn.is_async_generator
        assert defn.is_async

    def test_scope_as_string(self):
        @resource(scope="suite")
        def suite_resource():
            return "suite"

        defn = get_registry()["suite_resource"]
        assert defn.scope == Scope.SUITE

    def test_scope_as_enum(self):
        @resource(scope=Scope.SESSION)
        def session_resource():
            return "session"

        defn = get_registry()["session_resource"]
        assert defn.scope == Scope.SESSION

    def test_detects_dependencies(self):
        @resource
        def base():
            return 1

        @resource
        def dependent(base, other):
            return base + other

        defn = get_registry()["dependent"]
        assert defn.dependencies == ["base", "other"]


class TestResourceResolver:
    """Tests for ResourceResolver."""

    @pytest.mark.asyncio
    async def test_resolves_sync_resource(self):
        @resource
        def simple():
            return 42

        resolver = ResourceResolver(get_registry())
        value = await resolver.resolve("simple")
        assert value == 42

    @pytest.mark.asyncio
    async def test_resolves_async_resource(self):
        @resource
        async def async_simple():
            return "async_result"

        resolver = ResourceResolver(get_registry())
        value = await resolver.resolve("async_simple")
        assert value == "async_result"

    @pytest.mark.asyncio
    async def test_caches_case_scope(self):
        call_count = 0

        @resource(scope="case")
        def counted():
            nonlocal call_count
            call_count += 1
            return call_count

        resolver = ResourceResolver(get_registry())
        v1 = await resolver.resolve("counted")
        v2 = await resolver.resolve("counted")
        assert v1 == v2 == 1
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_resolves_dependencies(self):
        @resource
        def base_val():
            return 10

        @resource
        def derived(base_val):
            return base_val * 2

        resolver = ResourceResolver(get_registry())
        value = await resolver.resolve("derived")
        assert value == 20

    @pytest.mark.asyncio
    async def test_unknown_resource_raises(self):
        resolver = ResourceResolver(get_registry())
        with pytest.raises(ValueError, match="Unknown resource: unknown"):
            await resolver.resolve("unknown")

    @pytest.mark.asyncio
    async def test_resolve_many(self):
        @resource
        def res_a():
            return "a"

        @resource
        def res_b():
            return "b"

        resolver = ResourceResolver(get_registry())
        values = await resolver.resolve_many(["res_a", "res_b"])
        assert values == {"res_a": "a", "res_b": "b"}


class TestResourceTeardown:
    """Tests for resource teardown."""

    @pytest.mark.asyncio
    async def test_sync_generator_teardown(self):
        teardown_called = False

        @resource
        def gen_res():
            yield "value"
            nonlocal teardown_called
            teardown_called = True

        resolver = ResourceResolver(get_registry())
        value = await resolver.resolve("gen_res")
        assert value == "value"
        assert not teardown_called

        await resolver.teardown()
        assert teardown_called

    @pytest.mark.asyncio
    async def test_async_generator_teardown(self):
        teardown_called = False

        @resource
        async def async_gen_res():
            yield "async_value"
            nonlocal teardown_called
            teardown_called = True

        resolver = ResourceResolver(get_registry())
        value = await resolver.resolve("async_gen_res")
        assert value == "async_value"

        await resolver.teardown()
        assert teardown_called

    @pytest.mark.asyncio
    async def test_teardown_scope_case(self):
        case_torn = False
        suite_torn = False

        @resource(scope="case")
        def case_res():
            yield "case"
            nonlocal case_torn
            case_torn = True

        @resource(scope="suite")
        def suite_res():
            yield "suite"
            nonlocal suite_torn
            suite_torn = True

        resolver = ResourceResolver(get_registry())
        await resolver.resolve("case_res")
        await resolver.resolve("suite_res")

        await resolver.teardown_scope(Scope.CASE)
        assert case_torn
        assert not suite_torn

        await resolver.teardown()
        assert suite_torn

    @pytest.mark.asyncio
    async def test_teardown_clears_cache(self):
        call_count = 0

        @resource(scope="case")
        def counted_case():
            nonlocal call_count
            call_count += 1
            return call_count

        resolver = ResourceResolver(get_registry())
        v1 = await resolver.resolve("counted_case")
        assert v1 == 1

        await resolver.teardown_scope(Scope.CASE)

        v2 = await resolver.resolve("counted_case")
        assert v2 == 2


class TestForkForCase:
    """Tests for fork_for_case and parent/child isolation."""

    @pytest.mark.asyncio
    async def test_child_inherits_suite_cache(self):
        @resource(scope="suite")
        def suite_val():
            return "shared"

        parent = ResourceResolver(get_registry())
        await parent.resolve("suite_val")

        child = parent.fork_for_case()
        value = await child.resolve("suite_val")
        assert value == "shared"

    @pytest.mark.asyncio
    async def test_child_case_scope_isolated(self):
        call_count = 0

        @resource(scope="case")
        def case_val():
            nonlocal call_count
            call_count += 1
            return call_count

        parent = ResourceResolver(get_registry())
        parent_val = await parent.resolve("case_val")
        assert parent_val == 1

        child = parent.fork_for_case()
        child_val = await child.resolve("case_val")
        assert child_val == 2  # New instance for child

    @pytest.mark.asyncio
    async def test_child_suite_teardown_delegates_to_parent(self):
        teardown_called = False

        @resource(scope="suite")
        def suite_gen():
            yield "suite_value"
            nonlocal teardown_called
            teardown_called = True

        parent = ResourceResolver(get_registry())
        child = parent.fork_for_case()

        # Resolve in child - should register teardown with parent
        await child.resolve("suite_gen")

        # Child teardown should not touch SUITE
        await child.teardown_scope(Scope.CASE)
        assert not teardown_called

        # Parent teardown should run SUITE teardown
        await parent.teardown()
        assert teardown_called

    @pytest.mark.asyncio
    async def test_child_syncs_suite_cache_to_parent(self):
        @resource(scope="suite")
        def suite_new():
            return "new_suite"

        parent = ResourceResolver(get_registry())
        child = parent.fork_for_case()

        # Child resolves a new suite resource
        await child.resolve("suite_new")

        # Parent should now have it cached
        assert (Scope.SUITE, "suite_new") in parent._cache

    @pytest.mark.asyncio
    async def test_multiple_children_share_suite(self):
        call_count = 0

        @resource(scope="suite")
        def shared_suite():
            nonlocal call_count
            call_count += 1
            return call_count

        parent = ResourceResolver(get_registry())

        # First resolve in parent to populate cache
        parent_val = await parent.resolve("shared_suite")
        assert parent_val == 1

        # Children should inherit from parent cache
        child1 = parent.fork_for_case()
        child2 = parent.fork_for_case()

        v1 = await child1.resolve("shared_suite")
        v2 = await child2.resolve("shared_suite")

        assert v1 == v2 == 1
        assert call_count == 1


class TestResourceHooks:
    """Tests for on_resolve, on_injection and on_teardown hooks."""

    @pytest.mark.asyncio
    async def test_on_injection_called(self):
        injection_calls = []

        def track_injection(value):
            injection_calls.append(value)
            return value

        @resource(on_injection=track_injection)
        def simple():
            return 42

        resolver = ResourceResolver(get_registry())
        value = await resolver.resolve("simple")

        assert value == 42
        assert injection_calls == [42]

    @pytest.mark.asyncio
    async def test_on_injection_transforms_value(self):
        @resource(on_injection=lambda v: v * 2)
        def doubled():
            return 10

        resolver = ResourceResolver(get_registry())
        value = await resolver.resolve("doubled")

        assert value == 20

    @pytest.mark.asyncio
    async def test_on_resolve_called_once(self):
        resolve_calls = []

        def track_resolve(value):
            resolve_calls.append(value)
            return value

        @resource(on_resolve=track_resolve)
        def simple():
            return 42

        resolver = ResourceResolver(get_registry())
        await resolver.resolve("simple")
        await resolver.resolve("simple")

        assert resolve_calls == [42]

    @pytest.mark.asyncio
    async def test_on_teardown_called_after_generator_teardown(self):
        call_order = []

        def on_teardown_hook(value):
            call_order.append(("hook", value))

        @resource(on_teardown=on_teardown_hook)
        def gen_res():
            yield "value"
            call_order.append(("generator_teardown",))

        resolver = ResourceResolver(get_registry())
        await resolver.resolve("gen_res")
        await resolver.teardown()

        assert call_order == [("generator_teardown",), ("hook", "value")]

    @pytest.mark.asyncio
    async def test_on_teardown_with_teardown_scope(self):
        teardown_hook_called = False

        def on_teardown_hook(value):
            nonlocal teardown_hook_called
            teardown_hook_called = True

        @resource(scope="case", on_teardown=on_teardown_hook)
        def case_gen():
            yield "case_value"

        resolver = ResourceResolver(get_registry())
        await resolver.resolve("case_gen")

        assert not teardown_hook_called
        await resolver.teardown_scope(Scope.CASE)
        assert teardown_hook_called

    @pytest.mark.asyncio
    async def test_hooks_with_async_generator(self):
        injection_value = None
        teardown_value = None

        def on_injection_hook(value):
            nonlocal injection_value
            injection_value = value
            return value

        def on_teardown_hook(value):
            nonlocal teardown_value
            teardown_value = value

        @resource(on_injection=on_injection_hook, on_teardown=on_teardown_hook)
        async def async_gen():
            yield "async_value"

        resolver = ResourceResolver(get_registry())
        value = await resolver.resolve("async_gen")

        assert value == "async_value"
        assert injection_value == "async_value"

        await resolver.teardown()
        assert teardown_value == "async_value"

    @pytest.mark.asyncio
    async def test_on_injection_receives_custom_context(self):
        received_name = None

        def hook(value):
            nonlocal received_name
            if test_ctx := TEST_CONTEXT.get():
                received_name = test_ctx.item.name
            else:
                received_name = "unknown"
            return value

        @resource(on_injection=hook)
        def simple():
            return 42

        resolver = ResourceResolver(get_registry())
        with context_scope(Ctx(item=_make_item("my_test"))):
            await resolver.resolve("simple")
        assert received_name == "my_test"

    @pytest.mark.asyncio
    async def test_on_injection_receives_consumer_name_for_dependencies(self):
        contexts = {}

        def hook(value):
            if resolver_ctx := RESOLVER_CONTEXT.get():
                contexts[resolver_ctx.consumer_name] = value
            else:
                contexts[("unknown", value)] = value
            return value

        @resource(on_injection=hook)
        def dependency():
            return "dep_val"

        @resource
        def consumer(dependency):
            return f"got {dependency}"

        resolver = ResourceResolver(get_registry())
        await resolver.resolve("consumer")

        # dependency's hook should have been called with consumer_name="consumer"
        assert contexts["consumer"] == "dep_val"

    @pytest.mark.asyncio
    async def test_nested_dependency_context(self):
        history = []

        def hook(value):
            if resolver_ctx := RESOLVER_CONTEXT.get():
                history.append((resolver_ctx.consumer_name, value))
            else:
                history.append(("unknown", value))
            return value

        @resource(on_injection=hook)
        def leaf():
            return "leaf"

        @resource(on_injection=hook)
        def middle(leaf):
            return f"middle({leaf})"

        @resource
        def top(middle):
            return f"top({middle})"

        resolver = ResourceResolver(get_registry())
        await resolver.resolve("top")

        assert history == [
            ("middle", "leaf"),
            ("top", "middle(leaf)"),
        ]

    @pytest.mark.asyncio
    async def test_on_injection_called_for_cached_resource(self):
        call_count = 0

        def hook(value):
            nonlocal call_count
            call_count += 1
            return value

        @resource(on_injection=hook)
        def cached_res():
            return "val"

        resolver = ResourceResolver(get_registry())
        await resolver.resolve("cached_res")
        await resolver.resolve("cached_res")

        # Hook called twice, once for each injection
        assert call_count == 2
