from collections import defaultdict

from dotenv import load_dotenv
from hdbscan import HDBSCAN
from typing import List

from ..engines import llm_client
from ..types import AssertionState, AssertionStateGroup, StateGroupMetadata

load_dotenv()

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

async def cluster_failures(failed_assertions: List[AssertionState]) -> List[AssertionStateGroup]:
    """Cluster failed assertions for further error analysis"""

    embeddings = await llm_client.generate_embeddings(
        model="text-embedding-3-small",
        input_values=[assertion.failure_reason.analysis for assertion in failed_assertions], #type: ignore
    )

    labels = HDBSCAN(
        min_cluster_size=2, 
        min_samples=1
        ).fit_predict(embeddings) #type: ignore

    grouped = defaultdict(list)
    clusters = []

    for assertion, label in zip(failed_assertions, labels, strict=True):
        grouped[label].append(assertion)

    for label in sorted(grouped):
        states = grouped[label]
        cluster_errors = "\n".join(state.failure_reason.analysis for state in states)

        if label == -1:
            metadata = StateGroupMetadata(
                name="Unclustered errors",
                description="These failed test cases have not been grouped due to their uniqueness",
            )
        else:
            metadata = await llm_client.create_object(
                model="gpt-5-2025-08-07",
                prompt=CLUSTERING_PROMPT.format(errors=cluster_errors),
                schema=StateGroupMetadata
            )
        
        clusters.append(AssertionStateGroup(
            metadata=metadata, #type: ignore
            assertion_states=states,
            grouped_by="failed"))

    return clusters
