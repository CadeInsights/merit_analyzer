# Merit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Testing framework for AI systems with a familiar pytest-like interface and semantic assertions.

---

## Installation

```bash
uv add appmerit
```

Or with pip:

```bash
pip install appmerit
```

For more installation options, see the [installation documentation](https://docs.appmerit.com/installation).

---

## A Simple Example

```python
import merit

def chatbot(prompt: str) -> str:
    """Your AI system under test."""
    return f"Hello, {prompt}!"

def merit_chatbot_responds():
    """Test function - discovered automatically."""
    response = chatbot("World")
    assert response == "Hello, World!"
    assert len(response) > 0
```

Run it:

```bash
merit test
```

Output:

```
Merit Test Runner
=================

Collected 1 test

test_example.py::merit_chatbot_responds ✓

==================== 1 passed in 0.08s ====================
```

---

## Example with Resources

Resources are similar to pytest fixtures - they provide setup and teardown for your tests:

```python
import merit

@merit.resource
def api_client():
    """Setup resource with teardown."""
    client = {"connected": True, "url": "https://api.example.com"}
    yield client
    # Teardown after test completes
    client["connected"] = False

@merit.resource(scope="suite")
def expensive_model():
    """Suite-scoped resource - created once, shared across all tests."""
    model = load_expensive_model()
    return model

def merit_client_works(api_client):
    """Test receives resource as parameter."""
    assert api_client["connected"] is True
    assert "url" in api_client

async def merit_async_test(api_client):
    """Async tests work too."""
    result = await make_api_call(api_client)
    assert result is not None
```

---

## Parametrization

Test multiple inputs easily:

```python
import merit

def greet(name: str) -> str:
    return f"Hello, {name}!"

@merit.parametrize(
    "name,expected",
    [
        ("World", "Hello, World!"),
        ("Alice", "Hello, Alice!"),
        ("Bob", "Hello, Bob!"),
    ],
)
def merit_greetings(name: str, expected: str):
    """Runs 3 tests automatically."""
    assert greet(name) == expected
```

Output:

```
test_example.py::merit_greetings[World] ✓
test_example.py::merit_greetings[Alice] ✓
test_example.py::merit_greetings[Bob] ✓

==================== 3 passed in 0.10s ====================
```

---

## Test Cases

For more complex testing scenarios, use `Case` objects:

```python
import merit
from merit import Case

def chatbot(prompt: str) -> str:
    return f"Response to: {prompt}"

@merit.iter_cases([
    Case(
        sut_input_values={"prompt": "Hello"},
        references={"expected": "Response to: Hello"}
    ),
    Case(
        sut_input_values={"prompt": "Hi"},
        references={"expected": "Response to: Hi"}
    ),
])
def merit_chatbot_cases(case: Case):
    """Test using Case objects."""
    result = chatbot(**case.sut_input_values)
    if case.references:
        assert result == case.references["expected"]
```

---

## AI Predicates

Merit includes semantic assertions for testing AI outputs:

```python
import merit
from merit.predicates import has_facts, has_unsupported_facts

def chatbot(prompt: str) -> str:
    return "Paris is the capital of France and home to the Eiffel Tower."

async def merit_chatbot_accuracy():
    """Test with semantic assertions."""
    response = chatbot("Tell me about Paris")
    
    # Semantic fact checking
    assert await has_facts(response, "Paris is the capital of France")
    
    # Hallucination detection
    assert not await has_unsupported_facts(
        response, 
        "Paris is the capital of France. The Eiffel Tower is in Paris."
    )
```

Available predicates:
- `has_facts` - Verify required information is present
- `has_unsupported_facts` - Detect hallucinations
- `has_conflicting_facts` - Find contradictions
- `matches_facts` - Bidirectional fact matching
- `has_topics` - Check topic coverage
- `follows_policy` - Validate policy compliance
- `matches_writing_style` - Check writing style consistency
- `matches_writing_layout` - Verify document structure

See [AI Predicates documentation](https://docs.appmerit.com/predicates/overview) for details.

---

## Running Tests

Run all tests:

```bash
merit test
```

Run specific file:

```bash
merit test test_example.py
```

Run tests matching pattern:

```bash
merit test -k chatbot
```

Run with concurrency:

```bash
merit test --concurrency 10
```

See [Running Tests documentation](https://docs.appmerit.com/core/running-tests) for more options.

---

## Test Discovery

Merit automatically discovers:

- **Test files**: Files starting with `test_` or ending with `_test.py`
- **Test functions**: Functions starting with `merit_`
- **Test classes**: Classes starting with `Merit`

Example:

```python
# test_example.py

def merit_simple_test():
    """Discovered as a test."""
    assert True

class MeritCalculator:
    """Test class - discovered automatically."""
    
    def merit_addition(self):
        """Test method."""
        assert 2 + 2 == 4
    
    def merit_subtraction(self):
        """Another test method."""
        assert 5 - 3 == 2
```

---

## Configuration

Configure Merit using environment variables or a `.env` file:

```bash
# For AI predicates (optional)
MERIT_API_BASE_URL=https://api.appmerit.com
MERIT_API_KEY=your_api_key_here
```

Merit automatically loads `.env` files from your project directory.

See [configuration documentation](https://docs.appmerit.com/configuration) for all options.

---

## Features

In summary, you declare your tests as functions with standard Python syntax.

You do that with:

- Test functions (starting with `merit_`)
- Resources (similar to pytest fixtures)
- Parametrization for testing multiple inputs
- Case objects for complex test scenarios
- Async/await support throughout

...and with that you get:

- **Editor support**: Completion, type checks, linting
- **Automatic test discovery**: No configuration needed
- **Concurrent execution**: Run tests in parallel
- **Semantic assertions**: LLM-as-a-judge predicates for AI testing
- **Tracing integration**: Built-in OpenTelemetry support
- **Familiar interface**: Pytest-like API

---

## Documentation

Full documentation: **[docs.appmerit.com](https://docs.appmerit.com)**

**Getting Started:**
- [Quick Start](https://docs.appmerit.com/quickstart) - Get up and running in 2 minutes
- [Installation](https://docs.appmerit.com/installation) - Install Merit and set up environment
- [Your First Test](https://docs.appmerit.com/first-test) - Write and run your first test

**Core Concepts:**
- [Writing Tests](https://docs.appmerit.com/core/writing-tests) - Learn how to write tests
- [Resources](https://docs.appmerit.com/core/resources) - Dependency injection (like pytest fixtures)
- [Test Cases](https://docs.appmerit.com/core/test-cases) - Structure complex test scenarios
- [Running Tests](https://docs.appmerit.com/core/running-tests) - Execute and control your tests

**AI Predicates:**
- [Overview](https://docs.appmerit.com/predicates/overview) - LLM-as-a-Judge assertions
- [Fact Checking](https://docs.appmerit.com/predicates/fact-checking) - Verify facts and detect hallucinations
- [Topics & Policy](https://docs.appmerit.com/predicates/topics-policy) - Check topical coverage
- [Style & Structure](https://docs.appmerit.com/predicates/style-structure) - Match writing style

**Advanced:**
- [Parametrization](https://docs.appmerit.com/advanced/parametrize) - Test multiple inputs
- [Tags & Filters](https://docs.appmerit.com/advanced/tags-filters) - Organize and filter tests
- [Repeat Tests](https://docs.appmerit.com/advanced/repeat-tests) - Test reliability and flakiness
- [Tracing](https://docs.appmerit.com/advanced/tracing) - OpenTelemetry integration

**API Reference:**
- [Testing API](https://docs.appmerit.com/api-reference/testing) - Core functions and decorators
- [Predicates API](https://docs.appmerit.com/api-reference/predicates) - AI assertion functions
- [Metrics API](https://docs.appmerit.com/api-reference/metrics) - Test metrics and scoring

---

## Contributing

We welcome contributions! To get started:

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/merit.git`
3. Create a branch: `git checkout -b your-feature-name`
4. Install dependencies: `uv sync`
5. Make your changes
6. Run tests: `uv run merit test`
7. Run lints: `uv run ruff check .`
8. Submit a pull request

For more details, see [CONTRIBUTING.md](CONTRIBUTING.md).

**Development Setup:**

```bash
# Clone the repository
git clone https://github.com/appMerit/merit.git
cd merit

# Install dependencies
uv sync

# Run tests
uv run merit test

# Run lints
uv run ruff check .
uv run mypy .
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

- **Documentation**: [docs.appmerit.com](https://docs.appmerit.com)
- **GitHub Issues**: [github.com/appMerit/merit/issues](https://github.com/appMerit/merit/issues)
- **Email**: support@appmerit.com
