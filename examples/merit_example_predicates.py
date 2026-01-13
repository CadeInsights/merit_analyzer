"""Examples for AI predicates."""

from __future__ import annotations

from merit.predicates import (
    follows_policy,
    has_conflicting_facts,
    has_facts,
    has_topics,
    has_unsupported_facts,
    matches_facts,
    matches_writing_layout,
    matches_writing_style,
)


# =============================== Define SUT ===============================


def simple_chatbot(text: str) -> str:
    if "France" in text:
        return """
        France is a beautiful country located in Western Europe. 
        Its capital city is Paris, which is known for the Eiffel 
        Tower and croissants.
        """
    if "Germany" in text:
        if "markdown" in text:
            return """
            # Germany
            ## Location
            Germany is located in Central Europe.
            ## Capital
            Berlin is the capital of Germany.
            ## Famous for
            Berlin is famous for its history, museums, and vibrant 
            cultural scene.
            ## Population
            Germany has a population of 83 million people.
            """
        return """
            Germany, located in Central Europe, has Berlin as its capital. 
            Berlin is famous for its history, museums, and vibrant 
            cultural scene.
            """
    if "rock" in text:
        if "verbose" in text:
            return """
            Metallica is the greatest rock band of all time. 
            James Hetfield is the lead singer and rhythm guitarist.
            Lars Ulrich is the drummer.
            Kirk Hammett is the lead guitarist.
            Robert Trujillo is the bass guitarist.
            """
        return """
            Metallica is the greatest rock band of all time. 
            James Hetfield is the lead singer and rhythm guitarist.
            """
    return """
        I don't know.
        """


# =============================== Run predicate tests ===============================


async def merit_predicates_combining():
    answer = simple_chatbot("What is the greatest rock band of all time? Be verbose.")

    # check if the answer is a string, has content and is long enough
    assert isinstance(answer, str)
    assert answer != "I don't know."
    assert len(answer) > 50

    # check the answer has the expected facts
    assert await has_facts(answer, "Metallica is the greatest rock band of all time")


async def merit_predicates_facts_and_topics():
    answer = simple_chatbot(
        "Tell me about France. Don't include sports, sociology, politics, religion."
    )

    # has_facts and has_topics check if text contains all facts and topics from the reference text
    assert await has_facts(answer, "Paris is a capital of France. France is a country in Europe.")
    assert not await has_topics(answer, "Sports, sociology, politics, religion.")


async def merit_predicates_conflicting_and_unsupported_facts():
    answer = simple_chatbot("What is the greatest rock band of all time?")

    # check if answer contradicts the reference
    assert not await has_conflicting_facts(
        answer, "Lady Gaga is the greatest pop singer of all time."
    )

    # check if any facts in the answer don't have evidence in the reference
    assert not await has_unsupported_facts(
        answer,
        """James Alan Hetfield (born August 3, 1963) is an American musician. 
        He is the lead vocalist, rhythm guitarist, co-founder, and a primary songwriter 
        of the heavy metal band Metallica. Metallica is the greatest rock band of all time.
        """,
    )


async def merit_predicates_facts_match():
    answer = simple_chatbot("Tell me about France.")

    # check if answer has the same set of facts as the reference
    assert await matches_facts(
        answer,
        """France is a country. 
        Location - Western Europe. 
        Capital - Paris. 
        Paris is famous for the Eiffel Tower and croissants.
        """,
    )


async def merit_predicates_policy_follows():
    answer = simple_chatbot("What is the greatest rock band of all time?")

    # check if answer follows the policy
    assert await follows_policy(answer, "Answer must be in English.")
    assert await follows_policy(answer, "Must have at least one subjective statement.")


async def merit_predicates_style_match():
    answer = simple_chatbot("Tell me about Germany.")

    # check if answer has the same writing style as the reference
    # semantics of the text is ignored
    assert await matches_writing_style(
        answer,
        """Metallica is an American heavy metal band. 
        It was formed in Los Angeles in 1981 by vocalist/guitarist 
        James Hetfield and drummer Lars Ulrich, and has been based 
        in San Francisco for most of its career.""",
    )


async def merit_predicates_structure_match():
    answer = simple_chatbot("Tell me about Germany in markdown format.")

    # check if answer has the same layout as the reference
    # semantics of the text is ignored
    assert await matches_writing_layout(
        answer,
        """# Country
        ## Location
        Some info here.
        ## Capital
        Some info here.
        ## Famous for
        Some info here.
        ## Population
        Some info here.
        """,
    )
