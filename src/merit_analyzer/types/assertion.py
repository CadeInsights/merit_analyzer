from dataclasses import dataclass, field


@dataclass
class AssertionsResult:
    """Outcome of assertions for a single test case."""

    passed: bool
    errors: list[str] = field(default_factory=list)
