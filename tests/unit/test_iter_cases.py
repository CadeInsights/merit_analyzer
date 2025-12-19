import pytest
from pydantic_core import ValidationError
from uuid import UUID

from merit.testing.case import Case, iter_cases


def test_iter_cases_attaches_cases_as_decorator_and_copies_list():
    cases = [Case(tags={"a"}), Case(tags={"b"})]

    @iter_cases(cases)
    def merit_sample():
        return "ok"

    assert hasattr(merit_sample, "__merit_cases__")
    assert merit_sample.__merit_cases__ == cases
    assert merit_sample.__merit_cases__ is not cases

    # Changes to the original list should not affect the attached list (it is copied).
    cases.append(Case(tags={"c"}))
    assert len(cases) == 3
    assert len(merit_sample.__merit_cases__) == 2

    # Changes to the attached list should not affect the original list either.
    merit_sample.__merit_cases__.append(Case(tags={"d"}))
    assert len(merit_sample.__merit_cases__) == 3
    assert len(cases) == 3


def test_iter_cases_validates_sut_input_values_when_sut_provided():
    def sut(a: int, b: str = "x"):
        return a, b

    cases = [
        Case(sut_input_values={"a": 1, "b": "y"}),
        Case(sut_input_values={"a": 2}),  # default for b should be accepted
    ]

    @iter_cases(cases, sut_for_inputs_validation=sut)
    def merit_sample():
        return "ok"

    assert merit_sample.__merit_cases__ == cases


def test_iter_cases_validation_raises_for_invalid_inputs():
    def sut(a: int):
        return a

    # missing required 'a' (sut_input_values=None -> treated as {})
    with pytest.raises(ValidationError):
        iter_cases([Case()], sut_for_inputs_validation=sut)

    # wrong type for 'a'
    with pytest.raises(ValidationError):
        iter_cases([Case(sut_input_values={"a": "nope"})], sut_for_inputs_validation=sut)


def test_iter_cases_validation_raises_when_arguments_schema_missing():
    # `int` is callable, but it is not a function with an arguments schema.
    with pytest.raises(ValueError, match="No arguments schema found for the function"):
        iter_cases([Case(sut_input_values={"a": 1})], sut_for_inputs_validation=int)

