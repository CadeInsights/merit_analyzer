#TODO: introduce new examples when checks endpoint is ready

# """Demonstrates using custom Assertions alongside regular Python asserts."""

# import merit
# from merit.assertions import Contains, ExactMatch, StartsWith


# def simple_chatbot(prompt: str) -> str:
#     """Simple chatbot that greets users."""
#     return f"Hello, {prompt}! How can I help you today?"


# # Using custom Assertions - they raise on failure like regular assert
# def merit_exact_match_passes():
#     """Custom Assertion that passes."""
#     response = simple_chatbot("Alice")
#     ExactMatch("Hello, Alice! How can I help you today?")(response)


# def merit_exact_match_fails():
#     """Custom Assertion that fails - raises AssertionFailedError with details."""
#     response = simple_chatbot("Alice")
#     ExactMatch("Goodbye, Alice!")(response)


# # Multiple assertions in one test - fail-fast behavior
# def merit_multiple_assertions():
#     """Multiple assertions - stops at first failure."""
#     response = simple_chatbot("Bob")
#     StartsWith("Hello")(response)  # Passes
#     Contains("Bob")(response)  # Passes
#     Contains("Goodbye")(response)  # Fails - test stops here
#     ExactMatch("never reached")(response)


# # Mixing custom Assertions with regular Python assert
# def merit_mixed_assertions():
#     """Mix of custom Assertions and regular assert statements."""
#     response = simple_chatbot("Charlie")

#     # Custom assertion
#     StartsWith("Hello")(response)

#     # Regular Python assert
#     assert len(response) > 10, "Response too short"

#     # Another custom assertion
#     Contains("Charlie")(response)

#     # Regular assert
#     assert "?" in response


# # Using assertions with parametrize
# @merit.parametrize(
#     "name,expected_greeting",
#     [
#         ("World", "Hello, World!"),
#         ("Alice", "Hello, Alice!"),
#         ("Bob", "Hello, Bob!"),
#     ],
#     ids=["world", "alice", "bob"],
# )
# def merit_parametrized_with_assertions(name: str, expected_greeting: str):
#     """Parametrized test using StartsWith assertion."""
#     response = simple_chatbot(name)
#     StartsWith(expected_greeting)(response)


# def merit_failed_python_assert():
#     """Regular Python assert that fails."""
#     response = simple_chatbot("Dave")
#     assert "Goodbye" in response, "Response does not contain 'Goodbye'"
