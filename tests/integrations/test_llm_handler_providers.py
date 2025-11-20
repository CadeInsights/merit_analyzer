"""Integration tests for Anthropic handler across providers.

Tests the LLMClaude handler with:
- Direct Anthropic API
- AWS Bedrock
- Google Vertex AI

Each test validates:
1. generate_embeddings() - Returns embeddings (uses local model fallback)
2. create_object() - Returns structured Pydantic objects
3. compile_agent() - Creates agent with tools and policies
4. run_agent() - Executes agent tasks with file operations
"""

import os
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from pydantic import BaseModel

from merit_analyzer.core.llm_driver import get_llm_client, AGENT, TOOL, FILE_ACCESS_POLICY


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with test files."""
    with TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create test input file
        input_file = workspace / "input.txt"
        input_file.write_text("Hello World\nThis is a test file\nWith multiple lines")

        # Create test directory structure
        (workspace / "data").mkdir()
        (workspace / "data" / "file1.txt").write_text("Content 1")
        (workspace / "data" / "file2.txt").write_text("Content 2")

        yield workspace


@pytest.fixture
def sample_schema():
    """Sample Pydantic schema for testing create_object."""

    class TestResponse(BaseModel):
        result: str
        confidence: float

    return TestResponse


# ============================================================================
# Test: Anthropic Direct
# ============================================================================


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
@pytest.mark.asyncio
class TestAnthropicDirect:
    """Test Anthropic handler with direct API."""

    @pytest.fixture(autouse=True)
    async def setup(self):
        """Set environment for direct Anthropic."""
        os.environ["MODEL_VENDOR"] = "anthropic"
        os.environ["INFERENCE_VENDOR"] = "anthropic"
        self.client = await get_llm_client()

    async def test_generate_embeddings(self):
        """Test embeddings generation (falls back to local model)."""
        inputs = ["test input 1", "test input 2", "test input 3"]

        embeddings = await self.client.generate_embeddings(inputs)

        assert len(embeddings) == 3
        assert len(embeddings[0]) == 384  # Granite embedding dimension
        assert embeddings[0] != embeddings[1]  # Different inputs = different embeddings
        print(f"✅ Embeddings generated: {len(embeddings)} vectors of dimension {len(embeddings[0])}")

    async def test_create_object(self, sample_schema):
        """Test structured object creation."""
        prompt = 'Analyze this: "The system works correctly". Return JSON with result and confidence.'

        response = await self.client.create_object(prompt, sample_schema)

        assert isinstance(response, sample_schema)
        assert hasattr(response, "result")
        assert hasattr(response, "confidence")
        assert 0.0 <= response.confidence <= 1.0
        print(f"✅ Structured object created: {response}")

    async def test_compile_agent_read_only(self):
        """Test compiling an agent with READ_ONLY policy."""
        self.client.compile_agent(
            agent_name=AGENT.ERROR_ANALYZER,
            system_prompt="You are a file reader. Read files and analyze their content.",
            file_access=FILE_ACCESS_POLICY.READ_ONLY,
            standard_tools=[TOOL.READ, TOOL.GREP, TOOL.GLOB, TOOL.LS],
        )

        assert AGENT.ERROR_ANALYZER in self.client.compiled_agents
        options = self.client.compiled_agents[AGENT.ERROR_ANALYZER]
        assert options.permission_mode == "default"  # READ_ONLY maps to "default"
        print(f"✅ Agent compiled with tools: {options.allowed_tools}")

    async def test_run_agent_read_file(self, temp_workspace):
        """Test agent reading a file."""
        # Change to temp workspace
        original_cwd = Path.cwd()
        os.chdir(temp_workspace)

        try:
            # Compile agent
            self.client.compile_agent(
                agent_name=AGENT.ERROR_ANALYZER,
                system_prompt="Read the file and return its content.",
                file_access=FILE_ACCESS_POLICY.READ_ONLY,
                standard_tools=[TOOL.READ],
            )

            # Run agent
            task = "Read the file input.txt and tell me what it contains."
            result = await self.client.run_agent(agent=AGENT.ERROR_ANALYZER, task=task, output_type=str)

            assert isinstance(result, str)
            assert len(result) > 0
            print(f"✅ Agent read file successfully. Result length: {len(result)} chars")

        finally:
            os.chdir(original_cwd)


# ============================================================================
# Test: AWS Bedrock
# ============================================================================


@pytest.mark.skipif(
    not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")), reason="AWS credentials not set"
)
@pytest.mark.asyncio
class TestAnthropicBedrock:
    """Test Anthropic handler via AWS Bedrock."""

    @pytest.fixture(autouse=True)
    async def setup(self):
        """Set environment for Bedrock."""
        os.environ["MODEL_VENDOR"] = "anthropic"
        os.environ["INFERENCE_VENDOR"] = "aws"
        os.environ["AWS_REGION"] = os.getenv("AWS_REGION", "us-east-1")
        # Clear cache to force new client
        from merit_analyzer.core.llm_driver import cached_client, cached_key

        globals()["cached_client"] = None
        globals()["cached_key"] = None

        self.client = await get_llm_client()

    async def test_generate_embeddings(self):
        """Test embeddings via Bedrock (uses local model)."""
        inputs = ["bedrock test 1", "bedrock test 2"]

        embeddings = await self.client.generate_embeddings(inputs)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        print(f"✅ Bedrock: Embeddings generated via local model")

    async def test_create_object(self, sample_schema):
        """Test structured object creation via Bedrock."""
        prompt = 'Return JSON with result="bedrock works" and confidence=0.95'

        response = await self.client.create_object(prompt, sample_schema)

        assert isinstance(response, sample_schema)
        print(f"✅ Bedrock: Structured object created: {response}")

    async def test_compile_and_run_agent(self, temp_workspace):
        """Test compiling and running agent via Bedrock."""
        original_cwd = Path.cwd()
        os.chdir(temp_workspace)

        try:
            # Compile agent with READ and WRITE tools
            self.client.compile_agent(
                agent_name=AGENT.SUITE_SCHEMA_BUILDER,
                system_prompt="You can read and write files.",
                file_access=FILE_ACCESS_POLICY.READ_AND_WRITE,
                standard_tools=[TOOL.READ, TOOL.WRITE, TOOL.LS],
            )

            # Run agent to list files
            task = "List all files in the current directory."
            result = await self.client.run_agent(agent=AGENT.SUITE_SCHEMA_BUILDER, task=task, output_type=str)

            assert isinstance(result, str)
            print(f"✅ Bedrock: Agent executed via Bedrock")

        finally:
            os.chdir(original_cwd)


# ============================================================================
# Test: Google Vertex AI
# ============================================================================


@pytest.mark.skipif(not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), reason="GOOGLE_APPLICATION_CREDENTIALS not set")
@pytest.mark.asyncio
class TestAnthropicVertex:
    """Test Anthropic handler via Google Vertex AI."""

    @pytest.fixture(autouse=True)
    async def setup(self):
        """Set environment for Vertex."""
        os.environ["MODEL_VENDOR"] = "anthropic"
        os.environ["INFERENCE_VENDOR"] = "gcp"
        os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        # Clear cache
        from merit_analyzer.core.llm_driver import cached_client, cached_key

        globals()["cached_client"] = None
        globals()["cached_key"] = None

        self.client = await get_llm_client()

    async def test_generate_embeddings(self):
        """Test embeddings via Vertex (uses local model)."""
        inputs = ["vertex test 1", "vertex test 2"]

        embeddings = await self.client.generate_embeddings(inputs)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        print(f"✅ Vertex: Embeddings generated via local model")

    async def test_create_object(self, sample_schema):
        """Test structured object creation via Vertex."""
        prompt = 'Return JSON with result="vertex works" and confidence=0.9'

        response = await self.client.create_object(prompt, sample_schema)

        assert isinstance(response, sample_schema)
        print(f"✅ Vertex: Structured object created: {response}")


# ============================================================================
# Test: File Operations with Agents
# ============================================================================


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
@pytest.mark.asyncio
class TestAgentFileOperations:
    """Test various file operations with compiled agents."""

    async def _get_fresh_client(self):
        """Get a fresh client for each test to avoid state pollution."""
        import merit_analyzer.core.llm_driver as llm_driver
        import asyncio

        # Clear ALL cached state
        llm_driver.cached_client = None
        llm_driver.cached_key = None
        llm_driver.validated_once = False
        os.environ["MODEL_VENDOR"] = "anthropic"
        os.environ["INFERENCE_VENDOR"] = "anthropic"
        # Clear Bedrock/Vertex flags that previous tests may have set
        os.environ.pop("CLAUDE_CODE_USE_BEDROCK", None)
        os.environ.pop("CLAUDE_CODE_USE_VERTEX", None)
        # Small delay to let API settle
        await asyncio.sleep(0.5)
        return await get_llm_client()

    async def test_agent_read_multiple_files(self, temp_workspace):
        """Test agent reading multiple files."""
        client = await self._get_fresh_client()
        original_cwd = Path.cwd()
        os.chdir(temp_workspace)

        try:
            client.compile_agent(
                agent_name=AGENT.ERROR_ANALYZER,
                system_prompt="Read files and analyze them.",
                file_access=FILE_ACCESS_POLICY.READ_ONLY,
                standard_tools=[TOOL.READ, TOOL.LS, TOOL.GLOB],
                cwd=temp_workspace,
            )

            task = "Find all .txt files in the data directory and tell me how many there are."
            result = await client.run_agent(agent=AGENT.ERROR_ANALYZER, task=task, output_type=str)

            assert isinstance(result, str)
            print(f"✅ Agent found files: {result[:100]}...")

        finally:
            os.chdir(original_cwd)

    async def test_agent_write_file(self, temp_workspace):
        """Test agent writing a new file."""
        client = await self._get_fresh_client()
        original_cwd = Path.cwd()
        os.chdir(temp_workspace)

        try:
            client.compile_agent(
                agent_name=AGENT.SUITE_SCHEMA_BUILDER,
                system_prompt="You can write files.",
                file_access=FILE_ACCESS_POLICY.READ_AND_WRITE,
                standard_tools=[TOOL.WRITE, TOOL.READ],
                cwd=temp_workspace,
            )

            task = "Create a new file called output.txt with the content 'Test output from agent'."
            result = await client.run_agent(agent=AGENT.SUITE_SCHEMA_BUILDER, task=task, output_type=str)

            # Verify file was created
            output_file = temp_workspace / "output.txt"
            assert output_file.exists()
            content = output_file.read_text()
            assert "Test output" in content or "agent" in content.lower()
            print(f"✅ Agent wrote file: {content}")

        finally:
            os.chdir(original_cwd)

    async def test_agent_edit_file(self, temp_workspace):
        """Test agent editing an existing file."""
        client = await self._get_fresh_client()
        original_cwd = Path.cwd()
        os.chdir(temp_workspace)

        try:
            client.compile_agent(
                agent_name=AGENT.SUITE_SCHEMA_BUILDER,
                system_prompt="You can read and edit files.",
                file_access=FILE_ACCESS_POLICY.READ_AND_WRITE,
                standard_tools=[TOOL.READ, TOOL.EDIT],
                cwd=temp_workspace,
            )

            task = 'Read input.txt and replace "World" with "Universe".'
            result = await client.run_agent(agent=AGENT.SUITE_SCHEMA_BUILDER, task=task, output_type=str)

            # Verify file was edited
            content = (temp_workspace / "input.txt").read_text()
            assert "Universe" in content
            assert "World" not in content
            print(f"✅ Agent edited file successfully")

        finally:
            os.chdir(original_cwd)

    async def test_agent_grep_search(self, temp_workspace):
        """Test agent searching files with grep."""
        client = await self._get_fresh_client()
        original_cwd = Path.cwd()
        os.chdir(temp_workspace)

        try:
            client.compile_agent(
                agent_name=AGENT.ERROR_ANALYZER,
                system_prompt="Search files for patterns.",
                file_access=FILE_ACCESS_POLICY.READ_ONLY,
                standard_tools=[TOOL.GREP, TOOL.READ],
                cwd=temp_workspace,
            )

            task = "Search for the word 'Content' in all files."
            result = await client.run_agent(agent=AGENT.ERROR_ANALYZER, task=task, output_type=str)

            assert isinstance(result, str)
            print(f"✅ Agent grep search completed")

        finally:
            os.chdir(original_cwd)
