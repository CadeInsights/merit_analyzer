"""Simple example of using Merit testing framework."""

from merit import Case, CaseSet, ExactMatch, PassRate, Suite


# 1. Define your system under test
def simple_chatbot(prompt: str) -> str:
    """A simple chatbot that adds 'Hello, ' prefix."""
    return f"Hello, {prompt}!"


# 2. Create test cases
cases = CaseSet(
    cases=[
        Case(input="World", expected_output="Hello, World!"),
        Case(input="Alice", expected_output="Hello, Alice!"),
        Case(input="Bob", expected_output="Hello, Bob!"),
    ]
)

# 3. Create suite with assertions
suite = Suite(name="Chatbot Tests", case_set=cases, assertions=ExactMatch())

# 4. Run tests
results = suite.run(simple_chatbot)

# 5. Calculate metrics
pass_rate = PassRate()
score = pass_rate(results)

# Show individual results
print("\nDetailed Results:")
for result in results:
    status = "✓" if result.passed else "✗"
    print(f"{status} {result.assertion_name}: {result.message or 'passed'}")
