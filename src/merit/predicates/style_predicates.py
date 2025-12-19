from .base import PredicateResult
from .client import get_predicate_api_client

async def layout_matches(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> PredicateResult:
    """
    Formatting / structure check; expected usage:
    client = await get_predicate_api_client()
    await client.check(actual=actual, reference=reference, check="layout_matches")
    """

    raise NotImplementedError("Not implemented")


async def syntax_matches(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> PredicateResult:
    """
    Grammar / punctuation check; expected usage:
    client = await get_predicate_api_client()
    await client.check(actual=actual, reference=reference, check="syntax_matches")
    """

    raise NotImplementedError("Not implemented")


async def tone_matches(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> PredicateResult:
    """
    Tone / sentiment check; expected usage:
    client = await get_predicate_api_client()
    await client.check(actual=actual, reference=reference, check="tone_matches")
    """

    raise NotImplementedError("Not implemented")


async def vocabulary_matches(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> PredicateResult:
    """
    Vocabulary / phrasing check; expected usage:
    client = await get_predicate_api_client()
    await client.check(actual=actual, reference=reference, check="vocabulary_matches")
    """

    raise NotImplementedError("Not implemented")
