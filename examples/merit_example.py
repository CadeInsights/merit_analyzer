"""Example merit tests demonstrating the testing framework."""

import asyncio

import merit


# === Resources (similar to pytest fixtures) ===


@merit.resource
def config():
    """Simple sync resource."""
    return {"api_url": "https://api.example.com", "timeout": 30}


@merit.resource
def api_client(config):
    """Resource with dependency on another resource."""
    return {"url": config["api_url"], "connected": True}


@merit.resource(scope="suite")
def expensive_model():
    """Suite-scoped resource - shared across all tests in this file."""
    return "loaded-model-v1"


@merit.resource
async def async_db():
    """Async resource with teardown via yield."""
    db = {"connected": True, "data": []}
    yield db
    db["connected"] = False  # Teardown


# === Test Functions ===


def merit_config_has_url(config):
    """Test that config resource provides api_url."""
    assert "api_url" in config
    assert config["api_url"].startswith("https://")


def merit_client_connects(api_client):
    """Test that api_client resource is connected."""
    assert api_client["connected"] is True


def merit_model_is_loaded(expensive_model):
    """Test suite-scoped resource."""
    assert expensive_model == "loaded-model-v1"


async def merit_async_db_works(async_db):
    """Async test with async resource."""
    assert async_db["connected"] is True
    async_db["data"].append("test-entry")
    await asyncio.sleep(0.001)  # Simulate async work
    assert len(async_db["data"]) == 1


def merit_multiple_resources(config, api_client, expensive_model):
    """Test with multiple resource dependencies."""
    assert config is not None
    assert api_client is not None
    assert expensive_model is not None


# === Test Classes ===


class MeritAPITests:
    """Group related tests in a class."""

    def merit_client_has_url(self, api_client, config):
        """Class method test with resources."""
        assert api_client["url"] == config["api_url"]

    async def merit_async_operations(self, async_db):
        """Async class method test."""
        async_db["data"].append("class-test")
        await asyncio.sleep(0.001)
        assert "class-test" in async_db["data"]
