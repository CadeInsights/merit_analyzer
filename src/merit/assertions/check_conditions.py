from merit.assertions.base import AssertionResult, AssertionMetadata
from merit.assertions.assertions_api_client import AssertionAPIClient, ConditionsCheckRequest, AssertionAPIRoute


async def conditions_met(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> AssertionResult:
    """
    Verify the actual text satisfies the conditions described in the reference text using the Merit conditions endpoint.

    Parameters
    ----------
    actual : str
        Text to evaluate.
    reference : str
        Conditions, constraints, policies, or other text with rule-based requirements.
    strict : bool, default True
        Restrict model from implying and deriving.
    metrics : list, optional
        Reserved for future metric aggregation; unused by this call.

    Returns
    -------
    AssertionResult
        Structured result from the `/conditions` API containing pass flag, confidence, and message.
    """
    metadata = AssertionMetadata(actual=actual, reference=reference, strict=strict)
    request = ConditionsCheckRequest(actual=actual, reference=reference, strict=strict, check="met", with_context=context)

    async with AssertionAPIClient(
        base_url="https://api.appmerit.com/v1/assertions",
        token="test_token"
        ) as client:
        response = await client.get_assertion_result(AssertionAPIRoute.CONDITIONS_CHECK, request)

        return AssertionResult(
            metadata=metadata,
            passed=response.passed,
            confidence=response.confidence,
            message=response.message,
        )