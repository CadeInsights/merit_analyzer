# Merit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/appMerit/merit/actions/workflows/test.yml/badge.svg)](https://github.com/appMerit/merit/actions/workflows/test.yml)
[![Checks](https://github.com/appMerit/merit/actions/workflows/check.yml/badge.svg)](https://github.com/appMerit/merit/actions/workflows/check.yml)

Merit is a Python testing framework for AI projects. It follows pytest syntax and culture while introducing components essential for testing AI software: metrics, typed datasets, semantic predicates (LLM-as-a-Judge), and OTEL traces.

---

## Installation

```bash
uv add appmerit
```

---

# Merit 101

Follow pytest habits...

- Create 'merit_*.py' files
- Write 'def merit_*' functions
- Use 'merit.resource' instead of 'pytest.fixture'
- Add 'assert' expressions within the functions
- Run 'uv run merit test'

...while leveraging Merit APIs.
- Use 'with metrics()' context to turn failed assertions into quality metrics
- Use 'has_facts()' and other semantic predicates for asserting natural language
- Access OTEL span data and assert it with 'follows_policy()' predicate
- Parse datasets into clearly typed and validated data objects

---


## Example

```python
import merit
from merit import Case, Metric, metrics
from merit.predicates import has_unsupported_facts, follows_policy

from pydantic import BaseModel

@merit.sut
def store_chatbot(prompt: str) -> str:
    return call_llm(prompt)

@merit.metric
def accuracy():
    metric = Metric()
    yield metric

    assert metric.mean > 0.8
    yield metric.mean

class Refs(BaseModel):
    kb: str
    expected_tool: str | None = None

cases = [
    Case(sut_input_values={"prompt": "When are you open?"}, references=Refs(kb="Store hours: 9 AM - 6 PM, Monday-Saturday. Closed Sundays.")),
    Case(sut_input_values={"prompt": "Return policy?"}, references=Refs(kb="30-day returns with receipt.")),
    Case(sut_input_values={"prompt": "How much for the Nike Air Max?"}, references=Refs(kb="Nike Air Max: $129.99", expected_tool="offer_product")),
]

@merit.iter_cases(cases)
@merit.repeat(3)
async def merit_chatbot_no_hallucinations(
    case: Case[Refs],
    store_chatbot,
    accuracy: Metric,
    trace_context):
    """AI agent relies on knowledge base and tool calls for transactional questions"""
    response = store_chatbot(**case.sut_input_values)

    # Verify the answer don't have any unsupported facts
    with metrics(accuracy):
        assert not await has_unsupported_facts(response, case.references.kb)

    # Verify tool was called when expected
    if expected_tool := case.references.expected_tool:
        spans = trace_context.get_sut_spans(store_chatbot)
        tool_called = spans[1].attributes.get("llm.request.functions.0.name")

        assert tool_called == expected_tool
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

## Documentation

Full documentation: **[docs.appmerit.com](https://docs.appmerit.com)**

**Getting Started:**
- [Quick Start](https://docs.appmerit.com/get-started/quick-start) - Get up and running in 5 minutes

**Usage:**
- [Writing Merits](https://docs.appmerit.com/usage/writing-merits) - How to define a proper merit suite
- [Running Merits](https://docs.appmerit.com/usage/running-merits) - How to execute suits and merits

**Concepts:**
- [Merit](https://docs.appmerit.com/concepts/merit) - Like test but better
- [Resource](https://docs.appmerit.com/concepts/resource) - Like fixtures but better
- [Case](https://docs.appmerit.com/concepts/case) - Container for parsed dataset entities
- [Metric](https://docs.appmerit.com/concepts/metric) - Aggregating assertions
- [Semantic Predicates](https://docs.appmerit.com/concepts/semantic-predicates) - Asserting language and logs
- [SUT (System Under Test)](https://docs.appmerit.com/concepts/sut) - Collecting and accesing traces

**API Reference:**
- [Merit Definitions APIs](https://docs.appmerit.com/apis/testing) - Tune discovery and execution
- [Merit Predicates APIs](https://docs.appmerit.com/apis/predicates) - Build your own semantic predicates
- [Merit Metric APIs](https://docs.appmerit.com/apis/metrics) - Build complex metric systems
- [Merit Tracing APIs](https://docs.appmerit.com/apis/tracing) - OpenTelemetry integration

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
