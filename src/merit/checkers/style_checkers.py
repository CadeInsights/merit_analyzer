from .base import CheckerResult
from .client import get_checker_api_client

async def layout_matches(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> CheckerResult:
    """
    Formatting / structure check; expected usage:
    client = await get_remote_checks_client()
    await client.check(actual=actual, reference=reference, check="layout_matches")
    """

    raise NotImplementedError("Not implemented")


async def syntax_matches(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> CheckerResult:
    """
    Grammar / punctuation check; expected usage:
    client = await get_remote_checks_client()
    await client.check(actual=actual, reference=reference, check="syntax_matches")
    """

    raise NotImplementedError("Not implemented")


async def tone_matches(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> CheckerResult:
    """
    Tone / sentiment check; expected usage:
    client = await get_remote_checks_client()
    await client.check(actual=actual, reference=reference, check="tone_matches")
    """

    raise NotImplementedError("Not implemented")


async def vocabulary_matches(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> CheckerResult:
    """
    Vocabulary / phrasing check; expected usage:
    client = await get_remote_checks_client()
    await client.check(actual=actual, reference=reference, check="vocabulary_matches")
    """

    raise NotImplementedError("Not implemented")