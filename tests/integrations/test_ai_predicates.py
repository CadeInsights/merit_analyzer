import os

import dotenv
import pytest


dotenv.load_dotenv()

from merit.predicates.ai_predicates import (
    follows_policy,
    has_conflicting_facts,
    has_facts,
    has_topics,
    has_unsupported_facts,
    matches_facts,
    matches_writing_layout,
    matches_writing_style,
)
from merit.predicates.client import close_predicate_api_client


# Skip tests if API credentials are not provided
pytestmark = pytest.mark.skipif(
    not (os.getenv("MERIT_API_BASE_URL") and os.getenv("MERIT_API_KEY")),
    reason="MERIT_API_BASE_URL and MERIT_API_KEY must be set for integration tests",
)


@pytest.fixture(autouse=True)
async def setup_predicate_client():
    """Ensure the global predicate client is initialized and closed."""
    from merit.predicates.client import create_predicate_api_client

    create_predicate_api_client()
    yield
    await close_predicate_api_client()


@pytest.mark.asyncio
async def test_has_conflicting_facts_integration():
    # Value is True if contradictions are found

    # Case 1: Contradiction
    assert await has_conflicting_facts(
        actual="The apple is red.",
        reference="The apple is green.",
    )

    # Case 2: No contradiction
    assert not await has_conflicting_facts(
        actual="The apple is red.",
        reference="The sky is blue.",
    )


@pytest.mark.asyncio
async def test_has_unsupported_facts_integration():
    # Value is True if unsupported facts are found

    # Case 1: Unsupported fact
    assert await has_unsupported_facts(
        actual="The apple is red and it was grown in France.",
        reference="The apple is red. It costs $10.",
    )

    # Case 2: Supported facts
    assert not await has_unsupported_facts(
        actual="The apple is red.",
        reference="The apple is red. It costs $10. It was grown in France.",
    )


@pytest.mark.asyncio
async def test_has_facts_integration():
    # Value is True if all facts from reference are present in actual

    # Case 1: Missing facts
    assert not await has_facts(
        actual="The apple is red. It costs $10.",
        reference="The apple is red and it was grown in France.",
    )

    # Case 2: All facts present
    assert await has_facts(
        actual="The apple is red. It costs $10. It was grown in France.",
        reference="The apple is red. It was grown in France.",
    )


@pytest.mark.asyncio
async def test_has_topics_integration():
    # Value is True if all topics from reference are present in actual

    # Case 1: Missing topics
    assert not await has_topics(actual="The apple is red.", reference="Agriculture, Economics.")

    # Case 2: All topics present
    assert await has_topics(
        actual="The apple is red. It costs $10.", reference="Agriculture, Economics."
    )


@pytest.mark.asyncio
async def test_matches_facts_integration():
    # Value is True if facts match

    # Case 1: Match
    assert await matches_facts(
        actual="The capital of France is Paris. The apple is red.",
        reference="Paris is the French capital. The apple is red.",
    )

    # Case 2: Mismatch
    assert not await matches_facts(
        actual="The capital of France is Paris. The apple is red.",
        reference="Paris is the French capital. The apple is green.",
    )


@pytest.mark.asyncio
async def test_follows_policy_integration():
    # Value is True if policy is followed

    # Case 1: Compliant
    assert await follows_policy(
        actual="AI: Hello, how are you?", reference="AI response must start with 'Hello'."
    )

    # Case 2: Non-compliant
    assert not await follows_policy(
        actual="AI: How are you?", reference="AI response must start with 'Hello'."
    )


@pytest.mark.asyncio
async def test_matches_writing_layout_integration():
    # Case 1: Match
    assert await matches_writing_layout(
        actual="From: John Doe; To: Jane Smith; Subject: Meeting Agenda",
        reference="From: Michael Smith; To: Brandon Johnson; Subject: New Project Proposal",
    )

    # Case 2: Mismatch
    assert not await matches_writing_layout(
        actual="<From>: John Doe; <To>: Jane Smith; <Subject>: Meeting Agenda",
        reference="# Michael Smith writing to Brandon Johnson about a new project proposal",
        strict=True,
    )  # TODO: doesn't work good with strict=False, needs to be fixed


@pytest.mark.asyncio
async def test_matches_writing_style_integration():
    # Case 1: Match
    assert await matches_writing_style(
        actual="It's pretty cool, right?", reference="Check out this new project proposal"
    )

    # Case 2: Mismatch
    assert not await matches_writing_style(
        actual="It's pretty cool, right?",
        reference="Dear John, I'm writing to you about the new project proposal.",
    )
