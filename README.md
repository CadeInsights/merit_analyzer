<div align="center">

# Merit

**Testing framework for AI systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

[Documentation](https://docs.appmerit.com) | [GitHub](https://github.com/appMerit/merit)

</div>

---

**Merit** is a modern testing framework built specifically for LLMs and AI agents. Stop guessing. Start testing your AI with semantic assertions that understand what your AI actually says.

The key features are:

* **Semantic assertions**: LLM-as-a-Judge checks facts, not strings. Detects hallucinations, missing info, and contradictions automatically.
* **Familiar syntax**: Pytest-like interface. If you know pytest, you know Merit. Resources, parametrization, async support.
* **Production-ready**: Built for scale. Concurrent testing, tracing, CI/CD integration.

---

## Installation

```bash
pip install git+https://github.com/appMerit/merit.git
```

**Requirements**: Python 3.12+

---

## Example

### Create it

Create a file `test_chatbot.py`:

```python
import merit
from merit.predicates import has_facts, has_unsupported_facts

def chatbot(prompt: str) -> str:
    """Your AI system."""
    return "Paris is the capital of France and home to the Eiffel Tower."

async def merit_chatbot_accuracy():
    """Test chatbot with semantic assertions."""
    response = chatbot("What is the capital of France?")
    
    # ✓ Semantic fact checking (not string matching)
    assert await has_facts(response, "Paris is the capital of France")
    
    # ✓ Hallucination detection
    assert not await has_unsupported_facts(
        response, 
        "Paris is the capital of France. The Eiffel Tower is in Paris."
    )
```

### Run it

```bash
merit
```

**Output:**

```
Merit Test Runner
=================

Collected 1 test

test_chatbot.py::merit_chatbot_accuracy ✓

==================== 1 passed in 0.45s ====================
```

That's it! Merit handles the complexity of semantic evaluation for you.

---

## What You Can Test

Merit's LLM-as-a-Judge assertions understand **meaning**, not just text:

* **`has_facts`** - Catches incomplete responses that skip critical details
* **`has_unsupported_facts`** - Detects when your LLM invents information (hallucinations)
* **`has_conflicting_facts`** - Finds contradictions with your source material
* **`has_topics`** - Ensures all required subjects are covered
* **`follows_policy`** - Validates compliance with brand guidelines
* **`matches_writing_style`** - Checks tone, voice, and writing patterns
* **`has_structure`** - Validates format and organization
* **`is_valid_json`** - Checks JSON structure and schema

[See all predicates →](https://docs.appmerit.com/predicates/overview)

---

## Parametrization

Test multiple cases easily:

```python
import merit

def chatbot(prompt: str) -> str:
    return f"Hello, {prompt}!"

@merit.parametrize(
    "name,expected",
    [
        ("World", "Hello, World!"),
        ("Alice", "Hello, Alice!"),
        ("Bob", "Hello, Bob!"),
    ],
)
def merit_chatbot_greetings(name: str, expected: str):
    """Test multiple greetings."""
    response = chatbot(name)
    assert response == expected
```

**Output:**

```
test_chatbot.py::merit_chatbot_greetings[World] ✓
test_chatbot.py::merit_chatbot_greetings[Alice] ✓
test_chatbot.py::merit_chatbot_greetings[Bob] ✓

==================== 3 passed in 0.12s ====================
```

---

## Resources (Fixtures)

Merit supports pytest-like resources for setup and teardown:

```python
import merit

@merit.resource
def api_client():
    """Setup resource."""
    client = {"connected": True}
    yield client
    client["connected"] = False  # Teardown

@merit.resource(scope="suite")
def expensive_model():
    """Suite-scoped resource - shared across all tests."""
    return "loaded-model-v1"

def merit_client_works(api_client):
    """Test using resource."""
    assert api_client["connected"] is True

async def merit_async_test(api_client):
    """Async tests work too."""
    assert api_client["connected"] is True
```

---

## Configuration

For AI predicates, set up your Merit API credentials:

```bash
# .env file
MERIT_API_BASE_URL=https://api.appmerit.com
MERIT_API_KEY=your_api_key_here
```

AI predicates use the Merit cloud service. [Contact us](mailto:daniel@appmerit.com) to get access.

**Alternative**: You can also use local LLM providers (OpenAI, Anthropic, AWS Bedrock, etc.). See [Configuration docs](https://docs.appmerit.com/configuration) for details.

---

## Documentation

**Getting Started:**
- [Quick Start](https://docs.appmerit.com/quickstart) - Get up and running in 2 minutes
- [Your First Test](https://docs.appmerit.com/first-test) - Write and run your first test
- [Writing Tests](https://docs.appmerit.com/core/writing-tests) - Learn about resources and test structure

**Core Features:**
- [AI Predicates](https://docs.appmerit.com/predicates/overview) - LLM-as-a-Judge assertions
- [Test Cases](https://docs.appmerit.com/core/test-cases) - Organize tests with Case objects
- [Resources](https://docs.appmerit.com/core/resources) - Setup and teardown (fixtures)
- [Running Tests](https://docs.appmerit.com/core/running-tests) - CLI options and test discovery

**Advanced:**
- [Parametrization](https://docs.appmerit.com/advanced/parametrize) - Test multiple inputs
- [Tags & Filters](https://docs.appmerit.com/advanced/tags-filters) - Organize and filter tests
- [Tracing](https://docs.appmerit.com/advanced/tracing) - Distributed tracing integration

---

## Who Uses Merit?

- **Chatbot Developers** - Test response accuracy, detect hallucinations, ensure brand voice
- **Document AI** - Verify summaries, check extractions, validate transformations at scale
- **AI Agent Teams** - Test complex workflows, validate tool usage, ensure reliable behavior
- **RAG Systems** - Validate groundedness, catch hallucinations, test retrieval quality
- **Content Generation** - Check facts, verify style, ensure quality across 1000s of outputs

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Merit is open source! We provide this SDK and CLI freely under the MIT License. 

For premium cloud features and enterprise support, visit: https://appmerit.com
