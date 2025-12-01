"""Test suite for organizing and running test cases."""

from collections.abc import Callable
from typing import Any

from merit.assertions._base import Assertion, AssertionResult
from merit.testing.case import Case


class Suite:
    """A collection of test cases grouped together for execution.

    Attributes:
    ----------
    name : str
        Name of the test suite
    cases : list[Case]
        Collection of test cases
    assertions : list[Assertion]
        Suite-level assertions to run on each case
    """

    def __init__(self, name: str, cases: list[Case], assertions: list[Assertion] | Assertion | None = None):
        """Args:
        name : str
            Name for this suite
        cases : list[Case]
            Collection of test cases to run
        assertions : list[Assertion] | Assertion | None
            Suite-level assertion(s) to evaluate on each case
        """
        self.name = name
        self.cases = cases
        self.assertions = self._normalize_assertions(assertions)

    def _normalize_assertions(self, assertions: list | Assertion | None) -> list[Assertion]:
        """Normalize assertions to list."""
        if assertions is None:
            return []
        if isinstance(assertions, Assertion):
            return [assertions]
        return list(assertions)

    def run(self, system_under_test: Callable[[Any], Any]) -> list[AssertionResult]:
        """Run all test cases in the suite.

        Parameters
        ----------
        system_under_test : Callable
            Function to test. Takes case.input, returns output.

        Returns:
        -------
        list[AssertionResult]
            Results from all assertions on all cases
        """
        results = []

        for case in self.cases:
            actual = case.execute(system_under_test)

            # Merge suite-level and case-level assertions
            case_assertions = self._normalize_assertions(case.assertions)
            all_assertions = self.assertions + case_assertions

            for assertion in all_assertions:
                result = assertion(actual, case)
                results.append(result)

        return results
    
    @classmethod
    def from_csv(cls, path: str) -> "Suite":
        """Load test cases from a CSV file."""
        # TODO: Implementation

    @classmethod
    def from_json(cls, path: str) -> "Suite":
        """Load test cases from a JSON file."""
        # TODO: Implementation

