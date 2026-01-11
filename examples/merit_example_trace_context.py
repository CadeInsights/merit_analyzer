"""Examples showing how to use trace_context in merit methods.

The trace_context resource provides access to span data captured during
test execution, enabling assertions on LLM calls, SUT behavior, and
custom trace attributes.

Run with: uv run merit examples/merit_example_trace_context.py --trace
"""

import merit


@merit.sut
def simple_pipeline(query: str) -> str:
    """A simple SUT that processes a query."""
    with merit.trace_step("process", {"query_length": len(query)}):
        return f"Processed: {query}"


@merit.sut
def multi_step_pipeline(query: str) -> str:
    """A pipeline with multiple trace steps."""
    with merit.trace_step("retrieve"):
        docs = [f"doc about {query}"]

    with merit.trace_step("generate"):
        result = f"Generated from {len(docs)} docs: {query}"

    return result


# Basic: Set custom attributes on test span
def merit_set_custom_attributes(simple_pipeline, trace_context):
    """Demonstrate setting custom attributes on the test span."""
    result = simple_pipeline("hello world")

    trace_context.set_attribute("response.length", len(result))
    trace_context.set_attribute("response.has_prefix", result.startswith("Processed"))

    assert "hello" in result


# Query child spans created during test
def merit_query_child_spans(multi_step_pipeline, trace_context):
    """Demonstrate querying child spans from SUT execution."""
    result = multi_step_pipeline("test query")

    # Get all spans created during this test
    child_spans = trace_context.get_child_spans()

    # Verify expected trace steps occurred
    span_names = [s.name for s in child_spans]

    assert "retrieve" in span_names, f"Expected 'retrieve' in {span_names}"
    assert "generate" in span_names, f"Expected 'generate' in {span_names}"

    assert "test query" in result


# Check if tracing is enabled
def merit_check_tracing_enabled(simple_pipeline, trace_context):
    """Demonstrate checking if tracing is enabled."""
    result = simple_pipeline("check tracing")

    if trace_context.is_enabled:
        # Tracing is on, we can query spans
        spans = trace_context.get_child_spans()
        trace_context.set_attribute("spans.count", len(spans))
    else:
        # Tracing is off, spans list will be empty
        pass

    assert result is not None


# Access trace and span IDs
def merit_access_trace_ids(simple_pipeline, trace_context):
    """Demonstrate accessing trace and span identifiers."""
    result = simple_pipeline("get ids")

    # These are useful for correlating with external trace systems
    trace_id = trace_context.trace_id
    span_id = trace_context.span_id

    assert len(trace_id) == 32, "trace_id should be 32 hex chars"
    assert len(span_id) == 16, "span_id should be 16 hex chars"


# Filter for SUT spans specifically
def merit_get_sut_spans(multi_step_pipeline, trace_context):
    """Demonstrate filtering for @merit.sut decorated function spans."""
    result = multi_step_pipeline("sut test")

    # Get spans from @merit.sut functions (marked with merit.sut attribute)
    sut_spans = trace_context.get_sut_spans()

    # Note: trace_step spans are not SUT spans
    all_spans = trace_context.get_child_spans()

    assert len(all_spans) >= len(sut_spans)


# Use trace data for assertions
def merit_assert_on_trace_data(multi_step_pipeline, trace_context):
    """Demonstrate using trace data in test assertions."""
    result = multi_step_pipeline("assertion test")

    child_spans = trace_context.get_child_spans()

    # Assert minimum number of operations occurred
    assert len(child_spans) >= 2, "Expected at least 2 trace steps"

    # Find the 'process' step and check its attributes
    for span in child_spans:
        if span.name == "retrieve":
            # Span completed successfully
            assert span.status.status_code.name == "UNSET" or span.status.status_code.name == "OK"
            break

# Detailed SUT span inspection
def merit_inspect_sut_span(simple_pipeline, trace_context: merit.TraceContext):
    """Demonstrate inspecting the SUT span itself."""
    query = "test query"
    result = simple_pipeline(query)

    sut_spans = trace_context.get_sut_spans()
    assert len(sut_spans) == 1
    
    span = sut_spans[0]
    assert span.name == "sut.simple_pipeline"
    assert span.attributes.get("merit.sut") is True
    
    # Check automatically captured inputs/outputs
    # Note: Attribute keys might depend on configuration, but "sut.input.args" is standard
    if "sut.input.args" in span.attributes:
        assert query in span.attributes["sut.input.args"]
    
    if "sut.output" in span.attributes:
        assert result in span.attributes["sut.output"]

# Filter SUT spans by name
def merit_filter_sut_spans(simple_pipeline, multi_step_pipeline, trace_context):
    """Demonstrate filtering SUT spans by name."""
    simple_pipeline("query 1")
    multi_step_pipeline("query 2")

    simple_spans = trace_context.get_sut_spans(name="simple_pipeline")
    multi_spans = trace_context.get_sut_spans(name="multi_step_pipeline")
    all_sut_spans = trace_context.get_sut_spans()

    assert len(simple_spans) == 1
    assert simple_spans[0].name == "sut.simple_pipeline"
    
    assert len(multi_spans) == 1
    assert multi_spans[0].name == "sut.multi_step_pipeline"
    
    assert len(all_sut_spans) == 2
