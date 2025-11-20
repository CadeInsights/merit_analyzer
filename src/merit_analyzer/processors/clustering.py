from collections import defaultdict

from hdbscan import HDBSCAN
from typing import List

from ..core import get_llm_client
from ..types import TestCase, TestCaseGroup, GroupMetadata

CLUSTERING_PROMPT = """
<task>
    You are a professional QA analyst. You are given a cluster of errors. These errors happened inside a probabalistic function or class.
    Each error has a small explanation why it happened. You must analyze error carefully and provide a clear and comprehensive name and
    description for this cluster of errors.

    Data you provide are essential for debugging AI systems. Engineers will rely on your aggregated information to find and fix problems. 
</task>

<instructions>
    <formatting_instructions>
        Your response must be a JSON object.
    <formatting_instructions>

    <content_instructions>
        <field="name" description="Provide a clear and comprehensive name">
            <rule="Be as specific as possible">
                <good_example>
                    PRICE_PARSED_FOR_WRONG_CAR
                </good_example>
                <bad_example>
                    INCORRECT_PRICE
                </bad_example>
            </rule>

            <rule="Address the root of the problem.">
                <good_example>
                    AI_ANSWERS_IN_WRONG_LANGUAGE
                </good_example>
                <bad_example>
                    WRONG_LANGUAGE_USED
                </bad_example>
            </rule>
        </field>

        <field="description" description="Provide a clear and comprehensive description">
            <rule="Focus on specific patterns">
                <good_example>
                    Parsed price value is truncated when price in listing is abbreviated to thousands (20G, 20k, 20)
                </good_example>
                <bad_example>
                    Wrong value for price for listings with unconventional price tags.
                </bad_example>
            </rule>

            <rule="Follow the 'when X then Y' or 'X causes Y' format if possible.">
                <good_example>
                    When customer switches to Spanish, AI keeps answering in English. 
                </good_example>
                <bad_example>
                    Customer language is not the same as AI language.
                </bad_example>
            </rule>
        </field>
</instructions>

===== PREPARE TO RECEIVE THE CLUSTER OF ERRORS FOR ANALYSIS =====

<errors_for_analysis>
    {errors}
</errors_for_analysis>

===== ANALYZE ERRORS AND FOLLOW GIVEN INSTRUCTIONS =====
"""


async def cluster_failures(failed_test_cases: List[TestCase]) -> List[TestCaseGroup]:
    """Cluster failed test cases for further error analysis"""

    llm_client = await get_llm_client()

    embeddings = await llm_client.generate_embeddings(
        input_values=[
            "\n".join(test_case.assertions_result.errors if test_case.assertions_result else [])
            for test_case in failed_test_cases
        ],
    )  # TODO: if test_case.test_case_result.errors have multiple errors - cluster them separately

    labels = HDBSCAN(min_cluster_size=2, min_samples=1).fit_predict(embeddings)  # type: ignore

    aggregated = defaultdict(list)
    groups = []

    for test_case, label in zip(failed_test_cases, labels, strict=True):
        aggregated[label].append(test_case)

    for label in sorted(aggregated):
        states = aggregated[label]
        cluster_errors = "\n".join(
            "\n".join(test_case.assertions_result.errors if test_case.assertions_result else []) for test_case in states
        )

        if label == -1:
            metadata = GroupMetadata(
                name="Unclustered errors",
                description="These failed test cases have not been grouped due to their uniqueness",
            )
        else:
            metadata = await llm_client.create_object(
                prompt=CLUSTERING_PROMPT.format(errors=cluster_errors), schema=GroupMetadata
            )

        groups.append(
            TestCaseGroup(
                metadata=metadata,  # type: ignore
                test_cases=states,
            )
        )

    return groups
