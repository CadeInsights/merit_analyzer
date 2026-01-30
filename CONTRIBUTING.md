# Contributing to Merit

Thank you for your interest in contributing to Merit! We welcome contributions from the community.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Git

### Development Setup

1. **Fork the repository**

   Click the "Fork" button on GitHub to create your own copy of the repository.

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/merit.git
   cd merit
   ```

3. **Add upstream remote**

   ```bash
   git remote add upstream https://github.com/appMerit/merit.git
   ```

4. **Install dependencies**

   ```bash
   uv sync
   ```

   This will install all dependencies including development and linting tools.

5. **Verify installation**

   ```bash
   uv run merit --version
   ```

---

## Development Workflow

### Creating a Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-new-predicate` for new features
- `fix/issue-123` for bug fixes
- `docs/improve-readme` for documentation

### Making Changes

1. Make your changes in your branch
2. Write or update tests as needed
3. Update documentation if you're changing functionality
4. Ensure your code follows the project style

### Running Tests

Run the full test suite:

```bash
uv run merit test
```

Run specific tests:

```bash
uv run merit test tests/unit/test_runner.py
uv run merit test -k test_specific_function
```

Run with coverage (using pytest for coverage reporting):

```bash
uv run pytest --cov=merit --cov-report=html
```

### Code Quality

Merit uses `ruff` for linting and `mypy` for type checking.

**Run linter:**

```bash
uv run ruff check .
```

**Auto-fix issues:**

```bash
uv run ruff check --fix .
```

**Format code:**

```bash
uv run ruff format .
```

**Type checking:**

```bash
uv run mypy .
```

**Run all checks:**

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy .

# Test
uv run merit test
```

---

## Code Style

Merit follows these coding standards:

- **Python 3.12+** syntax and features
- **Type hints** on all functions and methods
- **Docstrings** using Google style for all public APIs
- **Line length** of 100 characters
- **OOP and DRY** principles
- Keep code concise - smaller is better

### Example

```python
def has_facts(actual: str, reference: str, *, strict: bool = False) -> bool:
    """Check if actual text contains facts from reference.

    Args:
        actual: The text to check for facts.
        reference: The reference text containing expected facts.
        strict: If True, requires explicit statements. Defaults to False.

    Returns:
        True if all facts from reference are present in actual.

    Example:
        >>> await has_facts("Paris is France's capital", "Paris is the capital of France")
        True
    """
    # Implementation
    pass
```

---

## Testing Guidelines

- Write tests for all new features
- Maintain or improve code coverage
- Use descriptive test names: `test_has_facts_detects_missing_information`
- Use pytest fixtures for common setup
- Test both success and failure cases
- Test edge cases

### Test Structure

```python
import merit

def test_resource_provides_dependency():
    """Test that resources can depend on other resources."""
    @merit.resource
    def config():
        return {"url": "https://api.example.com"}

    @merit.resource
    def client(config):
        return {"url": config["url"], "connected": True}

    # Create a test function that receives the resource
    def merit_test_client(client):
        assert client["connected"] is True
        assert client["url"] == "https://api.example.com"
```

---

## Documentation

When adding features, update:

1. **Docstrings** - All public APIs need docstrings
2. **Type hints** - Use proper type annotations
3. **Examples** - Add usage examples in docstrings
4. **README** - Update if adding major features
5. **API docs** - Update relevant documentation pages

---

## Commit Guidelines

Write clear, descriptive commit messages:

```bash
# Good commits
git commit -m "Add has_topics predicate for topic coverage checking"
git commit -m "Fix resource cleanup in async tests"
git commit -m "Update documentation for parametrize decorator"

# Avoid
git commit -m "fix bug"
git commit -m "updates"
```

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Example:**

```
feat: Add matches_writing_style predicate

Implements LLM-as-a-Judge predicate for checking writing style
consistency. Supports both strict and lenient modes.

Closes #123
```

---

## Submitting a Pull Request

1. **Update your branch**

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**

   - Go to the Merit repository on GitHub
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill out the PR template

4. **PR Description should include:**

   - What changes you made
   - Why you made them
   - How to test the changes
   - Related issue numbers (if any)

5. **Wait for review**

   - Address any feedback from reviewers
   - Make requested changes in new commits
   - Push updates to your branch

---

## Pull Request Checklist

Before submitting, ensure:

- [ ] Tests pass: `uv run merit test`
- [ ] Linting passes: `uv run ruff check .`
- [ ] Type checking passes: `uv run mypy .`
- [ ] Code is formatted: `uv run ruff format .`
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] PR description is complete

---

## Reporting Issues

### Bug Reports

When reporting bugs, include:

- Merit version: `merit --version`
- Python version: `python --version`
- Operating system
- Minimal code to reproduce the issue
- Expected behavior
- Actual behavior
- Error messages and stack traces

### Feature Requests

When requesting features:

- Describe the problem you're trying to solve
- Provide examples of how you'd like to use the feature
- Explain why this would benefit others

---

## Areas for Contribution

We welcome contributions in these areas:

- **New predicates** - Add LLM-as-a-Judge assertions
- **Bug fixes** - Fix reported issues
- **Documentation** - Improve docs, add examples
- **Tests** - Increase test coverage
- **Performance** - Optimize slow operations
- **Examples** - Add real-world usage examples

---

## Questions?

- **Documentation**: [docs.appmerit.com](https://docs.appmerit.com)
- **GitHub Issues**: [github.com/appMerit/merit/issues](https://github.com/appMerit/merit/issues)
- **Email**: support@appmerit.com

---

## Code of Conduct

Be respectful and constructive in all interactions. We're all here to make Merit better together.

---

## License

By contributing to Merit, you agree that your contributions will be licensed under the MIT License.
