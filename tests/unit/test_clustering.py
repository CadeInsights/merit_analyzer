from merit_analyzer.processors.clustering import cluster_failures
from merit_analyzer.types import AssertionsResult, TestCase, TestCaseValues


async def test_cluster_failures_integration() -> None:
    analyses = [
        "User service timed out during login attempts.",
        "Login request failed because the user service took too much time.",
        "Payment gateway rejected the transaction with code 502.",
        "Charge attempt hit an error from the Stripe gateway.",
        "Formatting mismatch for CSV export columns.",
    ]
    assertions = [
        TestCase(
            case_data=TestCaseValues(case_input=f"case-{idx}", reference_value="expected"),
            output_for_assertions=None,
            assertions_result=AssertionsResult(False, [text]),
        )
        for idx, text in enumerate(analyses)
    ]

    groups = await cluster_failures(assertions)

    assert isinstance(groups, list)
    assert len(groups) == 3
