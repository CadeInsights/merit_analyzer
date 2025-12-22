from typing import Any
from pydantic import BaseModel, ValidationError
import pytest
from merit.testing.case import Case, valididate_cases_for_sut, iter_cases
from merit.testing.parametrize import get_parameter_sets


def test_case_generic_dict():
    """Test Case with dict as references."""
    case = Case[dict[str, Any]](
        references={"expected": "value"},
        sut_input_values={"input": "data"}
    )
    assert case.references == {"expected": "value"}
    assert case.sut_input_values == {"input": "data"}


def test_case_generic_basemodel():
    """Test Case with BaseModel as references."""
    class MyRefs(BaseModel):
        expected: str
        score: float

    case = Case[MyRefs](
        references=MyRefs(expected="value", score=1.0),
        sut_input_values={"input": "data"}
    )
    assert isinstance(case.references, MyRefs)
    assert case.references.expected == "value"
    assert case.references.score == 1.0


def test_valididate_cases_for_sut_valid():
    """Test valididate_cases_for_sut with valid inputs."""
    def my_sut(name: str, age: int, *args, **kwargs):
        pass

    cases = [
        Case(sut_input_values={"name": "Alice", "age": 30}),
        Case(sut_input_values={"name": "Bob", "age": 25})
    ]
    
    assert valididate_cases_for_sut(cases, my_sut) == cases


def test_valididate_cases_for_sut_invalid():
    """Test valididate_cases_for_sut with invalid inputs."""
    def my_sut(name: str, age: int):
        pass

    # age should be int, but we provide str
    cases = [
        Case(sut_input_values={"name": "Alice", "age": "not-an-int"})
    ]
    
    with pytest.raises(ValidationError):
        valididate_cases_for_sut(cases, my_sut)


def test_iter_cases_decorator():
    """Test iter_cases decorator attaches cases correctly."""
    cases = [
        Case(sut_input_values={"x": 1}),
        Case(sut_input_values={"x": 2})
    ]

    @iter_cases(cases)
    def my_test(case):
        pass

    param_sets = get_parameter_sets(my_test)
    assert len(param_sets) == 2
    assert param_sets[0].values["case"] == cases[0]
    assert param_sets[1].values["case"] == cases[1]
    
    # Check that IDs are correctly set from case IDs
    assert param_sets[0].id_suffix == str(cases[0].id)
    assert param_sets[1].id_suffix == str(cases[1].id)

