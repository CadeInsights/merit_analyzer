from merit.assertions.base import AssertionResult, AssertionMetadata
from merit.assertions.assertions_api_client import AssertionAPIClient, StyleCheckRequest, AssertionAPIRoute


async def style_match(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> AssertionResult:
    """
    Verify the writing style of the actual text matches the reference text using the Merit style endpoint.

    Parameters
    ----------
    actual : str
        Actual text to evaluate.
    reference : str
        Style exemplar or ground-truth text.
    strict : bool, default True
        Whether to require strict style adherence.
    metrics : list, optional
        Reserved for future metric aggregation; unused by this call.

    Returns
    -------
    AssertionResult
        Structured result from the `/style` API containing pass flag, confidence, and message.
    """
    metadata = AssertionMetadata(actual=actual, reference=reference, strict=strict)
    request = StyleCheckRequest(actual=actual, reference=reference, strict=strict, check="match", with_context=context)

    async with AssertionAPIClient(
        base_url="https://api.appmerit.com/v1/assertions",
        token="test_token"
        ) as client:
        response = await client.get_assertion_result(AssertionAPIRoute.STYLE_CHECK, request)

    return AssertionResult(
        metadata=metadata,
        passed=response.passed,
        confidence=response.confidence,
        message=response.message,
    )

