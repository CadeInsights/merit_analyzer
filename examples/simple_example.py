# TODO: introduce new examples when checks endpoint is ready

# """Simple example of using Merit testing framework."""

# from merit import Assertion, AssertionResult, Case, ExactMatch, PassRate, Suite


# class StartsWithAssertion(Assertion):
#     """Custom assertion example."""

#     name = "StartsWithAssertion"

#     def __init__(self, expected: str):
#         self.expected = expected

#     def __call__(self, actual: str, case: Case) -> AssertionResult:
#         passed = actual.lower().startswith(self.expected.lower())
#         return AssertionResult(
#             assertion_name=self.name,
#             passed=passed,
#             score=1.0 if passed else 0.0,
#             confidence=1.0,
#             message=None if passed else f"Expected (case-insensitive): {self.expected!r}, Got: {actual!r}",
#         )


# # 1. Define your system under test
# def simple_chatbot(prompt: str) -> str:
#     """A simple chatbot that adds 'Hello, ' prefix."""
#     return f"Hello, {prompt}!"


# # 2. Create test cases
# cases = [
#     Case(input="World", assertions=[ExactMatch("Hello, World!")]),
#     Case(input="Alice", assertions=[ExactMatch("Hello, Alice!")]),
#     Case(input="Bob", assertions=[ExactMatch("Hello, Bob!")]),
# ]

# # 3. Create suite with assertions
# suite = Suite(name="Chatbot Tests", cases=cases, assertions=StartsWithAssertion("Hello"))

# # 4. Run tests
# results = suite.run(simple_chatbot)

# # 5. Calculate metrics
# pass_rate = PassRate()
# score = pass_rate(results)

# # Show individual results
# print("\nDetailed Results:")
# for result in results:
#     status = "✓" if result.passed else "✗"
#     print(f"{status} {result.assertion_name}: {result.message or 'passed'}")
