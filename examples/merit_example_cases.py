"""Examples for `Case` and `iter_cases`."""

from __future__ import annotations

from pydantic import BaseModel

import merit
from merit import Case, valididate_cases_for_sut


# =============================== Define SUT ===============================


def simple_chatbot(prompt: str) -> str:
    if "verbose" in prompt:
        answer = "What an excellent question! The answer is: "
    else:
        answer = ""
    if "France" in prompt:
        return answer + "Paris"
    if "Germany" in prompt:
        return answer + "Berlin"
    if "rock" in prompt:
        return answer + "Metallica"
    if "pop" in prompt:
        return answer + "Lady Gaga"
    raise ValueError(f"Unknown query: {prompt}")


# =============================== Prepare test cases ===============================


# Models for references are optional but helpful for static type checking
class ExampleReferences(BaseModel):
    expected: str
    max_len: int
    min_len: int


# Define all cases
all_cases = [
    Case[ExampleReferences](
        tags={"geography"},
        metadata={"verbose": False},
        references=ExampleReferences(expected="Paris", max_len=10, min_len=1),
        sut_input_values={"prompt": "What is the capital of France? Be concise."},
    ),
    Case[ExampleReferences](
        tags={"geography"},
        metadata={"verbose": True},
        references=ExampleReferences(expected="Berlin", max_len=10000, min_len=20),
        sut_input_values={"prompt": "What is the capital of Germany? Be verbose."},
    ),
    Case[ExampleReferences](
        tags={"music"},
        metadata={"verbose": True},
        references=ExampleReferences(expected="Metallica", max_len=10000, min_len=20),
        sut_input_values={"prompt": "What is the best rock band? Be verbose."},
    ),
    Case[ExampleReferences](
        tags={"music"},
        metadata={"verbose": False},
        references=ExampleReferences(expected="Lady Gaga", max_len=10, min_len=1),
        sut_input_values={"prompt": "What is the best pop band? Be concise."},
    ),
]


# =============================== Run tests ===============================


# Get output for each case input and assert against references
@merit.iter_cases(all_cases)
def merit_iter_cases_basic_usage(case: Case[ExampleReferences]):
    response = simple_chatbot(**case.sut_input_values)

    if case.references:
        assert case.references.expected in response
        assert len(response) <= case.references.max_len
        assert len(response) >= case.references.min_len


# Filter cases in code
@merit.iter_cases([c for c in all_cases if "geography" in c.tags])
def merit_iter_cases_only_geography(case: Case[ExampleReferences]):
    response = simple_chatbot(**case.sut_input_values)

    if case.references:
        assert case.references.expected in response
        assert len(response) <= case.references.max_len
        assert len(response) >= case.references.min_len


# Fail early if any case has invalid input
@merit.iter_cases(valididate_cases_for_sut(all_cases, simple_chatbot))
def merit_iter_cases_with_validation(case: Case[ExampleReferences]):
    response = simple_chatbot(**case.sut_input_values)

    if case.references:
        assert case.references.expected in response
        assert len(response) <= case.references.max_len
        assert len(response) >= case.references.min_len
