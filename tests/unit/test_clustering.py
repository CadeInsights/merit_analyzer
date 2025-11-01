from error_analysis.engines.clustering import cluster_failures
from error_analysis.models import AssertionState, AssertionStateGroup, StateFailureReason, TestCase

async def test_cluster_failures_integration() -> None:
    analyses = [
        "User service timed out during login attempts.",
        "Login request failed because the user service took too much time.",
        "Payment gateway rejected the transaction with code 502.",
        "Charge attempt hit an error from the Stripe gateway.",
        "Formatting mismatch for CSV export columns.",
    ]
    assertions = [
        AssertionState(
            test_case=TestCase(input_value=f"case-{idx}", expected="expected"),
            return_value=None,
            passed=False,
            confidence=1.0,
            failure_reason=StateFailureReason(analysis=text),
        )
        for idx, text in enumerate(analyses)
    ]

    groups = await cluster_failures(assertions)

    assert isinstance(groups, list)
    assert len(groups) == 3