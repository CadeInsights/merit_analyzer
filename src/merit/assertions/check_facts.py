"""Functions for checking facts in the actual text against the reference text."""

from merit.assertions.base import AssertionResult, AssertionMetadata
from merit.assertions.assertions_api_client import AssertionAPIClient, FactsCheckRequest, AssertionAPIRoute


async def facts_not_contradict(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> AssertionResult:
    """
    Verify facts in the actual text do not contradict any facts in the reference text using the Merit facts endpoint.

    Parameters
    ----------
    actual : str
        Text to evaluate.
    reference : str
        Ground-truth or source text.
    strict : bool, default True
        Restrict model from implying and deriving facts.
    metrics : list, optional
        Reserved for future metric aggregation; unused by this call.

    Returns
    -------
    AssertionResult
        Structured result from the `/facts` API containing pass flag, confidence, and message.
    """
    metadata = AssertionMetadata(actual=actual, reference=reference, strict=strict)
    request = FactsCheckRequest(actual=actual, reference=reference, strict=strict, check="contradictions", with_context=context)

    async with AssertionAPIClient(
        base_url="https://api.appmerit.com/v1/assertions", 
        token="test_token"
        ) as client:

        response = await client.get_assertion_result(AssertionAPIRoute.FACTS_CHECK, request)
        
        return AssertionResult(
            metadata=metadata,
            passed=response.passed,
            confidence=response.confidence,
            message=response.message,
        )


async def facts_supported(actual: str, reference: str, strict: bool = True, metrics: list | None = None) -> AssertionResult:
    """
    Verify facts in the actual text are supported by any facts in the reference text using the Merit facts endpoint.

    Parameters
    ----------
    actual : str
        Generated text to evaluate.
    reference : str
        Ground-truth or source text.
    strict : bool, default True
        Restrict model from implying and deriving facts.
    metrics : list, optional
        Reserved for future metric aggregation; unused by this call.

    Returns
    -------
    AssertionResult
        Structured result from the `/facts` API containing pass flag, confidence, and message.
    """
    metadata = AssertionMetadata(actual=actual, reference=reference, strict=strict)
    request = FactsCheckRequest(actual=actual, reference=reference, strict=strict, check="supported", with_context=context)

    async with AssertionAPIClient(
        base_url="https://api.appmerit.com/v1/assertions", 
        token="test_token"
        ) as client:

        response = await client.get_assertion_result(AssertionAPIRoute.FACTS_CHECK, request)
        
        return AssertionResult(
            metadata=metadata,
            passed=response.passed,
            confidence=response.confidence,
            message=response.message,
        )


async def facts_full_match(actual: str, reference: str, strict: bool = True, metrics: list | None = None) -> AssertionResult:
    """
    Verify facts in the actual text fully match any facts in the reference text using the Merit facts endpoint.

    Parameters
    ----------
    actual : str
        Generated text to evaluate.
    reference : str
        Ground-truth or source text.
    strict : bool, default True
        Whether to apply the API's strict full-match policy.
    metrics : list, optional
        Reserved for future metric aggregation; unused by this call.

    Returns
    -------
    AssertionResult
        Structured result from the `/facts` API containing pass flag, confidence, and message.

    Notes
    -----
    Requires the shared `client` to be initialized (e.g., via `async with client`).
    """
    metadata = AssertionMetadata(actual=actual, reference=reference, strict=strict)
    request = FactsCheckRequest(actual=actual, reference=reference, strict=strict, check="full_match", with_context=context)

    async with AssertionAPIClient(
        base_url="https://api.appmerit.com/v1/assertions", 
        token="test_token"
        ) as client:

        response = await client.get_assertion_result(AssertionAPIRoute.FACTS_CHECK, request)
        
        return AssertionResult(
            metadata=metadata,
            passed=response.passed,
            confidence=response.confidence,
            message=response.message,
        )