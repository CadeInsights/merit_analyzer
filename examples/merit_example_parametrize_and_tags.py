"""Demonstrates Merit parametrization and tag utilities."""

import merit


def simple_chatbot(prompt: str) -> str:
    return f"Hello, {prompt}!"


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


@merit.tag.skip(reason="Dependency still offline")
def merit_external_dependency() -> None:
    """Example of permanently skipped test with a reason."""
    raise RuntimeError("Should never execute")


@merit.tag.xfail(reason="Farewell flow not implemented yet")
def merit_chatbot_farewell() -> None:
    response = simple_chatbot("friend")
    assert response.endswith("Goodbye!")
