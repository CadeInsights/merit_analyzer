"""Tests for merit.testing.resources module."""

import pytest

from merit.testing.resources import (
    ResourceDef,
    ResourceResolver,
    Scope,
    clear_registry,
    get_registry,
    resource,
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
