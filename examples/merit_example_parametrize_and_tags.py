"""Demonstrates Merit parametrization and tag utilities."""

import merit


def simple_chatbot(prompt: str) -> str:
    return f"Hello, {prompt}!"

def goodbye_chatbot(prompt: str) -> str:
    return f"Goodbye, {prompt}!"

@merit.parametrize(
    "prompt,expected",
    [
        ("World", "Hello, World!"),
        ("Alice", "Hello, Alice!"),
        ("Bob", "Hello, Bob!"),
    ],
    ids=["world", "alice", "bob"],
)
@merit.tag("smoke", "chatbot")
def merit_chatbot_greetings(prompt: str, expected: str) -> None:
    """This test runs three times, once per parameter set."""
    assert simple_chatbot(prompt) == expected

@merit.repeat(count=2)
@merit.parametrize(
    "prompt,expected",
    [
        ("World", "Goodbye, World!"),
        ("Alice", "Goodbye, Alice!"),
        ("Bob", "Goodbye, Bob!"),
    ],
    ids=["world", "alice", "bob"],
)
@merit.tag("smoke", "chatbot")
def merit_chatbot_goodbyes(prompt: str, expected: str) -> None:
    """This test runs three times, once per parameter set."""
    assert goodbye_chatbot(prompt) == expected

@merit.tag.skip(reason="Dependency still offline")
def merit_external_dependency() -> None:
    """Example of permanently skipped test with a reason."""
    raise RuntimeError("Should never execute")


@merit.tag.xfail(reason="Farewell flow not implemented yet")
def merit_chatbot_farewell() -> None:
    response = simple_chatbot("friend")
    assert response.endswith("Goodbye!")


@merit.repeat(count=5)
def merit_chatbot_stability() -> None:
    """Test that the chatbot consistently responds correctly."""
    response = simple_chatbot("tester")
    assert response == "Hello, tester!"


@merit.repeat(count=10, min_passes=8)
def merit_mostly_fail():
    """A test that fails too often and won't meet the minimum pass threshold.

    This demonstrates a flaky test that passes sometimes,
    but requires 8 passes to be considered successful.
    """
    import random

    random.seed()  # Different seed each run
    # Only passes 20% of the time - will fail overall
    assert random.random() < 0.2, "Random failure"
