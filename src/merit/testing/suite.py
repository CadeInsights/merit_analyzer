"""Test suite for organizing and running test cases."""

from collections.abc import Callable
from typing import Any

from merit.assertions._base import Assertion, AssertionResult
from merit.testing.case import CaseSet


class Suite:
    """A collection of test cases grouped together for execution.

    Attributes:
    ----------
    name : str
        Name of the test suite
    case_set : CaseSet
        Collection of test cases
    assertions : list[Assertion]
        List of assertions to run on each case
    """

    def __init__(self, name: str, case_set: CaseSet, assertions: list[Assertion] | Assertion):
        """Args:
        name : str
            Name for this suite
        case_set : CaseSet
            Collection of test cases to run
        assertions : list[Assertion] | Assertion
            Assertion(s) to evaluate on each case
        """
        self.name = name
        self.case_set = case_set

        # Normalize to list
        if isinstance(assertions, Assertion):
            self.assertions = [assertions]
        else:
            self.assertions = assertions

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

        for case in self.case_set.cases:
            # Execute system under test
            actual = case.execute(system_under_test)

            # Run all assertions
            for assertion in self.assertions:
                result = assertion(actual, case)
                results.append(result)

        return results
