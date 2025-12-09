"""Example merit tests demonstrating OpenTelemetry tracing with SUT.

This example shows how to:
1. Use @sut to register a traced system-under-test
2. Trace custom steps within tests using trace_step
3. Capture LLM calls automatically via OpenLLMetry instrumentation

Run with tracing enabled:
    merit examples/merit_example_tracing.py --trace

The traces will be exported to traces.json at the end of the run.
"""

import merit


# === System Under Test Examples ===


@merit.sut
def simple_sut(prompt: str) -> str:
    """Simple sync SUT - all calls inside are traced."""
    # In a real scenario, this would call an LLM
    return f"Response to: {prompt}"


@merit.sut
async def async_sut(prompt: str) -> str:
    """Async SUT - works the same way."""
    # Simulating async LLM call
    import asyncio

    await asyncio.sleep(0.01)
    return f"Async response to: {prompt}"


@merit.sut
class PipelineSUT:
    """Class-based SUT - __call__ is automatically traced."""

    def __init__(self) -> None:
        self.context = "initialized"

    def __call__(self, query: str) -> str:
        """Main entry point - traced automatically."""
        retrieved = self._retrieve(query)
        return self._generate(retrieved, query)

    def _retrieve(self, query: str) -> list[str]:
        """Internal method - use trace_step for finer granularity."""
        with merit.trace_step("retrieve", {"query_length": len(query)}):
            return [f"doc1 about {query}", f"doc2 about {query}"]

    def _generate(self, docs: list[str], query: str) -> str:
        """Internal method with trace step."""
        with merit.trace_step("generate", {"doc_count": len(docs)}):
            return f"Answer based on {len(docs)} docs for: {query}"


# === Test Functions ===


def merit_simple_sut_works(simple_sut):
    """Test that simple SUT is invoked and traced."""
    result = simple_sut("Hello, world!")
    assert "Hello" in result


async def merit_async_sut_works(async_sut):
    """Test async SUT with tracing."""
    result = await async_sut("Async question")
    assert "Async" in result


def merit_pipeline_works(pipeline_sut):
    """Test class-based SUT with internal trace steps."""
    result = pipeline_sut("What is Python?")
    assert "Answer" in result
    assert "Python" in result


def merit_custom_trace_steps(simple_sut):
    """Demonstrate custom trace steps in test logic."""
    # Custom preprocessing step
    with merit.trace_step("preprocessing"):
        prompt = "processed: test input"

    # SUT invocation (automatically traced)
    result = simple_sut(prompt)

    # Custom validation step
    with merit.trace_step("validation", {"result_length": len(result)}):
        assert len(result) > 0
        assert "processed" in result


def merit_multiple_sut_calls(simple_sut, pipeline_sut):
    """Test with multiple SUT resources - each call is a separate span."""
    result1 = simple_sut("First call")
    result2 = pipeline_sut("Second call")

    assert "First" in result1
    assert "Second" in result2


# === Example with external client (would be traced if real) ===


@merit.sut
def agent_with_external_client(task: str) -> str:
    """SUT that would use an externally instantiated client.

    Even if the client is created outside this function,
    OpenLLMetry instruments at the SDK level, so all LLM
    calls become child spans of this SUT span.
    """
    # In real usage:
    # from openai import OpenAI
    # client = OpenAI()  # Even if created at module level
    # response = client.chat.completions.create(...)
    # All these calls would be traced under the sut.agent_with_external_client span
    return f"Completed task: {task}"


def merit_external_client_pattern(agent_with_external_client):
    """Test showing the external client pattern."""
    result = agent_with_external_client("Summarize this document")
    assert "Completed" in result
