from .base import PredicateResult
from .client import get_predicate_api_client

async def contradicts(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> PredicateResult:
    """
    Planned remote check; expected usage:
    client = await get_predicate_api_client()
    await client.check(actual=actual, reference=reference, check="contradicts")
    """

    raise NotImplementedError("Not implemented")


async def supported(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> PredicateResult:
    """
    Planned remote check; expected usage:
    client = await get_predicate_api_client()
    await client.check(actual=actual, reference=reference, check="supported")
    """

    raise NotImplementedError("Not implemented")


async def contains(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> PredicateResult:
    """
    Planned remote check; expected usage:
    client = await get_predicate_api_client()
    await client.check(actual=actual, reference=reference, check="contains")
    """

    raise NotImplementedError("Not implemented")


async def matches(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> PredicateResult:
    """
    Planned remote check; expected usage:
    client = await get_predicate_api_client()
    await client.check(actual=actual, reference=reference, check="matches")
    """

    raise NotImplementedError("Not implemented")
