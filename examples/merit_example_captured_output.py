"""Example: Using captured_output resource to inspect stdout/stderr.

Run with:
    merit test examples/merit_example_captured_output.py

Run with live output visible:
    merit test examples/merit_example_captured_output.py -s
"""

import merit


class Greeter:
    def __init__(self, prefix: str = "Hello"):
        self.prefix = prefix

    def greet(self, name: str) -> str:
        message = f"{self.prefix}, {name}!"
        print(message)
        return message

    def greet_many(self, names: list[str]) -> list[str]:
        messages = []
        for name in names:
            messages.append(self.greet(name))
        return messages


@merit.resource
def greeter() -> Greeter:
    """Prints greetings."""
    return Greeter()


def merit_capture_stdout(captured_output):
    """Test that captures and inspects stdout."""
    print("Hello from the test!")
    print("Another line")

    out, err = captured_output.readouterr()

    assert "Hello from the test!" in out
    assert "Another line" in out
    assert err == ""


def merit_capture_stderr(captured_output):
    """Test that captures and inspects stderr."""
    import sys

    sys.stderr.write("Error message\n")

    out, err = captured_output.readouterr()

    assert out == ""
    assert "Error message" in err


def merit_capture_multiple_reads(captured_output):
    """Test that readouterr() clears the buffer each time."""
    print("First")
    out1, _ = captured_output.readouterr()

    print("Second")
    out2, _ = captured_output.readouterr()

    assert out1 == "First\n"
    assert out2 == "Second\n"


def merit_capture_from_function_call(captured_output):
    """Test capturing output from code under test."""

    def say_hello(name: str) -> None:
        print(f"Hi, {name}!")

    say_hello("World")

    out, _ = captured_output.readouterr()
    assert out == "Hi, World!\n"


def merit_capture_output(greeter, captured_output):
    """Test capturing stdout from a SUT."""
    result = greeter.greet("Alice")

    out, _ = captured_output.readouterr()

    assert result == "Hello, Alice!"
    assert out == "Hello, Alice!\n"


def merit_capture_multiple_calls(greeter, captured_output):
    """Test capturing stdout from multiple SUT calls."""
    greeter.greet_many(["Alice", "Bob", "Charlie"])

    out, _ = captured_output.readouterr()

    assert "Hello, Alice!" in out
    assert "Hello, Bob!" in out
    assert "Hello, Charlie!" in out


def merit_disabled_bypasses_capture(captured_output):
    """Test that disabled() allows output to pass through to real stdout."""
    print("this is captured")

    with captured_output.disabled():
        print("this goes to real stdout (useful for debugging)")

    print("this is captured again")

    out, _ = captured_output.readouterr()

    assert "this is captured" in out
    assert "this is captured again" in out
    assert "this goes to real stdout" not in out
