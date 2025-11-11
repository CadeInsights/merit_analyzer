from __future__ import annotations

from pydantic import BaseModel, Field
from dataclasses import dataclass, asdict
from typing import Any, List, Literal, Tuple, TypedDict

from .assertion import AssertionsResult
from .error import ErrorDescription, ErrorAnalysis

from ..core import get_llm_client

# Relevant prompts

GENERATE_ERROR_DATA = """
<task>
    You are a professional QA analyst. You are given a failed test case data:
    - The input value for a function
    - The reference value we expected to get
    - The actual value returned by a function

    This test case is missing an error message. Your job is to generate this message.
</task>

<requirements>
    - Error message must be focused on content rather than types. We already have pytest traces — we need more context.
    - Error message must be clear and comrehensible. Drop small talk, get straight to the point.
    - If there are multiple errors: list them all.
</requirements>

<good_examples>
    <good_example_1>
        'Price abbreviation led to incorrectly parsed value. The listing mentions 20G. "G" is a common abbreviation for "Grands" 
         which means "thousands". That's why the expected value is 20,000, but actual value is 20.'
    </good_example_1>
    <good_example_2>
        [
            'Trim was incorrectly parsed as model. "LX" is a common trim for Lexus cars. It should have been parsed as trim.', 
            'Odometer value was parsed from a wrong car. 89,000 miles is related to Honda. There is no information on odometer value for Lexus.'
        ]
    </good_example_2>
</good_examples>

<bad_examples>
    <bad_example_1>
        'The expected value is 20,000. The actual value is 20. The value was parsed incorrectly.
    </bad_example_1>
    <2>
        [
            'Trim should be parsed properly.', 
            'Function returned the wrong value for odometer.'
        ]
    </2>
</bad_examples>

====== PREPARE TO RECEIVE THE ERROR DATA FOR ANALYSIS ======

<error_data>
    {error_data}
</error_data>

====== START ERROR MESSAGE GENERATION ======
"""

# Core objects

@dataclass
class TestCase:
    case_data: TestCaseValues
    output_for_assertions: Any | None
    assertions_result: AssertionsResult | None

    async def generate_error_data(self) -> None:
        error_data = {
            "test_input_value": self.case_data.case_input,
            "expected_value": self.case_data.reference_value,
            "actual_value": self.output_for_assertions
        }
        
        client = await get_llm_client()
        error_message = await client.create_object(
            prompt=GENERATE_ERROR_DATA.format(error_data=error_data), 
            schema=ErrorDescription,
            model=None
            )
        if not self.assertions_result:
            self.assertions_result = AssertionsResult(False, [])

        self.assertions_result.errors = error_message.errors

@dataclass
class TestCaseGroup:
    metadata: GroupMetadata
    test_cases: List[TestCase]
    error_analysis: ErrorAnalysis | None = None

@dataclass
class TestCaseValues:
    case_input: str
    reference_value: str

# Schemas and secondary objects

class GroupMetadata(BaseModel):
    name: str = Field(description="name for the cluster formatted in screaming snake case (e.g, INCORRECT_PRICE_PARSING)")
    description: str = Field(description="What happens in which circumstances.")
