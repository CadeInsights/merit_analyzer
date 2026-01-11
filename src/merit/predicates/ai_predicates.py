from merit.predicates.base import PredicateMetadata, PredicateResult
from merit.predicates.client import PredicateAPIRequest, PredicateType, get_predicate_api_client


async def has_conflicting_facts(
    actual: str,
    reference: str,
    strict: bool = False,
) -> PredicateResult:
    """Check if any facts in the actual text contradict any facts in the reference text.

    Parameters
    ----------
    actual : str
        The text produced by the system under test.
    reference : str
        The source text or context used for grounding.
    strict : bool, default False
        Whether to use strict matching (explicit contradictions) or
        lenient matching (allowing for semantic inference).

    Returns:
    -------
    PredicateResult
        The evaluation result. `value` is True if contradictions are found.

    Examples:
    --------
    >>> actual = ("The apple is red.",)
    >>> reference = "The apple is green."
    True
    ...
    >>> actual = ("The apple is red.",)
    >>> reference = "The sky is blue."
    False
    """
    client = await get_predicate_api_client()
    resp = await client.request_predicate(
        PredicateAPIRequest(
            assertion_type=PredicateType.FACTS_NOT_CONTRADICT,
            actual=actual,
            reference=reference,
            strict=strict,
        )
    )
    return PredicateResult(
        predicate_metadata=PredicateMetadata(
            predicate_name="has_conflicting_facts",
            actual=actual,
            reference=reference,
            strict=strict,
        ),
        value=not resp.passed,
        confidence=resp.confidence,
        message=resp.reasoning,
    )


async def has_unsupported_facts(
    actual: str,
    reference: str,
    strict: bool = False,
) -> PredicateResult:
    """Check if any facts in the actual text don't have evidence in the reference text.

    Parameters
    ----------
    actual : str
        The text produced by the system under test.
    reference : str
        The source text or context used for grounding.
    strict : bool, default False
        Whether to require explicit support in the reference text.

    Returns:
    -------
    PredicateResult
        The evaluation result. `value` is True if unsupported facts are found.

    Examples:
    --------
    >>> actual = ("The apple is red and it was grown in France.",)
    >>> reference = "The apple is red. It costs $10."
    True
    ...
    >>> actual = ("The apple is red.",)
    >>> reference = "The apple is red. It costs $10. It was grown in France."
    False
    """
    client = await get_predicate_api_client()
    resp = await client.request_predicate(
        PredicateAPIRequest(
            assertion_type=PredicateType.FACTS_SUPPORTED,
            actual=actual,
            reference=reference,
            strict=strict,
        )
    )
    return PredicateResult(
        predicate_metadata=PredicateMetadata(
            predicate_name="has_unsupported_facts",
            actual=actual,
            reference=reference,
            strict=strict,
        ),
        value=not resp.passed,
        confidence=resp.confidence,
        message=resp.reasoning,
    )


async def has_facts(
    actual: str,
    reference: str,
    strict: bool = False,
) -> PredicateResult:
    """Check if all facts from the reference text are present in the actual text.

    Parameters
    ----------
    actual : str
        The text produced by the system under test.
    reference : str
        The reference text containing expected facts.
    strict : bool, default False
        Whether to require an exact factual match.

    Returns:
    -------
    PredicateResult
        The evaluation result.

    Examples:
    --------
    >>> actual = ("The apple is red. It costs $10.",)
    >>> reference = "The apple was grown in France."
    False
    ...
    >>> actual = ("The apple is red. It costs $10. It was grown in France.",)
    >>> reference = "The apple was grown in France."
    True
    """
    client = await get_predicate_api_client()
    resp = await client.request_predicate(
        PredicateAPIRequest(
            assertion_type=PredicateType.FACTS_NOT_MISSING,
            actual=actual,
            reference=reference,
            strict=strict,
        )
    )
    return PredicateResult(
        predicate_metadata=PredicateMetadata(
            predicate_name="has_facts", actual=actual, reference=reference, strict=strict
        ),
        value=resp.passed,
        confidence=resp.confidence,
        message=resp.reasoning,
    )


async def matches_facts(
    actual: str,
    reference: str,
    strict: bool = False,
) -> PredicateResult:
    """Check if the actual text and reference text have the same set of facts.

    Parameters
    ----------
    actual : str
        The text produced by the system under test.
    reference : str
        The ground truth text to compare against.
    strict : bool, default False
        Whether to require strict semantic equality of facts.

    Returns:
    -------
    PredicateResult
        The evaluation result. `value` is True if the texts match factually.

    Examples:
    --------
    >>> actual = ("The capital of France is Paris. The apple is red.",)
    >>> reference = "Paris is the French capital. The apple is red."
    True
    ...
    >>> actual = ("The capital of France is Paris. The apple is red.",)
    >>> reference = "Paris is the French capital. The apple is green."
    False
    """
    client = await get_predicate_api_client()
    resp = await client.request_predicate(
        PredicateAPIRequest(
            assertion_type=PredicateType.FACTS_FULL_MATCH,
            actual=actual,
            reference=reference,
            strict=strict,
        )
    )
    return PredicateResult(
        predicate_metadata=PredicateMetadata(
            predicate_name="matches_facts", actual=actual, reference=reference, strict=strict
        ),
        value=resp.passed,
        confidence=resp.confidence,
        message=resp.reasoning,
    )


async def has_topics(
    actual: str,
    reference: str,
    strict: bool = False,
) -> PredicateResult:
    """Check if the actual text contains all topics from the reference text.

    Parameters
    ----------
    actual : str
        The text produced by the system under test.
    reference : str
        The reference text containing expected topics.
    strict : bool, default False
        Whether to require an exact topic match.

    Returns:
    -------
    PredicateResult
        The evaluation result.

    Examples:
    --------
    >>> actual = ("The apple is red.",)
    >>> reference = "Agriculture, Economics."
    False
    ...
    >>> actual = ("The apple is red. It costs $10.",)
    >>> reference = "Agriculture, Economics."
    True
    """
    client = await get_predicate_api_client()
    resp = await client.request_predicate(
        PredicateAPIRequest(
            assertion_type=PredicateType.HAS_TOPICS,
            actual=actual,
            reference=reference,
            strict=strict,
        )
    )
    return PredicateResult(
        predicate_metadata=PredicateMetadata(
            predicate_name="has_topics", actual=actual, reference=reference, strict=strict
        ),
        value=resp.passed,
        confidence=resp.confidence,
        message=resp.reasoning,
    )


async def follows_policy(
    actual: str,
    reference: str,
    strict: bool = False,
) -> PredicateResult:
    """Check if the actual text follows all rules and instructions in the reference text.

    Parameters
    ----------
    actual : str
        The text produced by the system under test.
    reference : str
        The set of policies, requirements, or instructions.
    strict : bool, default False
        Whether to enforce strict adherence to the policy.

    Returns:
    -------
    PredicateResult
        The evaluation result. `value` is True if the text is compliant.

    Examples:
    --------
    >>> actual = ("AI: Hello, how are you?",)
    >>> reference = "AI response must start with 'Hello'."
    True
    ...
    >>> actual = ("AI: How are you?",)
    >>> reference = "AI response must start with 'Hello'."
    False
    """
    client = await get_predicate_api_client()
    resp = await client.request_predicate(
        PredicateAPIRequest(
            assertion_type=PredicateType.CONDITIONS_MET,
            actual=actual,
            reference=reference,
            strict=strict,
        )
    )
    return PredicateResult(
        predicate_metadata=PredicateMetadata(
            predicate_name="follows_policy", actual=actual, reference=reference, strict=strict
        ),
        value=resp.passed,
        confidence=resp.confidence,
        message=resp.reasoning,
    )


async def matches_writing_layout(
    actual: str,
    reference: str,
    strict: bool = False,
) -> PredicateResult:
    """Check if the actual text follows the same structure and formatting as the reference.

    Parameters
    ----------
    actual : str
        The text produced by the system under test.
    reference : str
        An example document demonstrating the desired structure.
    strict : bool, default False
        Whether to require strict structural alignment.

    Returns:
    -------
    PredicateResult
        The evaluation result. `value` is True if the layout matches.

    Examples:
    --------
    >>> result = await has_same_writing_layout(
    >>> actual = ("From: John Does; To: Jane Smith; Subject: Meeting Agenda",)
    >>> reference = "From: Michael Smith; To: Brandon Johnson; Subject: New Project Proposal"
    True
    ...
    >>> actual = ("From: John Does; To: Jane Smith; Subject: Meeting Agenda",)
    >>> reference = "Michael Smith writing to Brandon Johnson about a new project proposal"
    False
    """
    client = await get_predicate_api_client()
    resp = await client.request_predicate(
        PredicateAPIRequest(
            assertion_type=PredicateType.STRUCTURE_MATCH,
            actual=actual,
            reference=reference,
            strict=strict,
        )
    )
    return PredicateResult(
        predicate_metadata=PredicateMetadata(
            predicate_name="matches_writing_layout",
            actual=actual,
            reference=reference,
            strict=strict,
        ),
        value=resp.passed,
        confidence=resp.confidence,
        message=resp.reasoning,
    )


async def matches_writing_style(
    actual: str,
    reference: str,
    strict: bool = False,
) -> PredicateResult:
    """Check if the actual text has the same writing style as the reference text.

    Parameters
    ----------
    actual : str
        The text produced by the system under test.
    reference : str
        An example document demonstrating the desired style.
    strict : bool, default False
        Whether to require strict stylistic matching.

    Returns:
    -------
    PredicateResult
        The evaluation result. `value` is True if the style matches.

    Examples:
    --------
    >>> actual = ("It's pretty cool, right?",)
    >>> reference = "Check out this new project proposal"
    True
    ...
    >>> actual = ("It's pretty cool, right?",)
    >>> reference = "Dear John, I'm writing to you about the new project proposal."
    False
    """
    client = await get_predicate_api_client()
    resp = await client.request_predicate(
        PredicateAPIRequest(
            assertion_type=PredicateType.STYLE_MATCH,
            actual=actual,
            reference=reference,
            strict=strict,
        )
    )
    return PredicateResult(
        predicate_metadata=PredicateMetadata(
            predicate_name="matches_writing_style",
            actual=actual,
            reference=reference,
            strict=strict,
        ),
        value=resp.passed,
        confidence=resp.confidence,
        message=resp.reasoning,
    )
