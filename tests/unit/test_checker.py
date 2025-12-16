import pytest
import json

from merit.checks.base import CheckerResult, CheckerMetadata

def simple_checker(actual: str, reference: str, context: str | None = None, strict: bool = True, metrics: list | None = None) -> CheckerResult:
    return CheckerResult(
        value=actual == reference,
        message=None,
        checker=CheckerMetadata(actual=actual, reference=reference, context=context, strict=strict),
        confidence=1.0,
    )
    

def test_checker_result_and_metadata_auto_filled():
    def merit_with_simple_checker():
        result = simple_checker("test", "test")

        # Basic properties
        assert result
        assert result.value is True
        assert result.confidence == 1.0

        checker_metadata = result.checker

        # Checker metadata
        assert checker_metadata.actual == "test"
        assert checker_metadata.reference == "test"

        # Auto-filled identifiers
        assert checker_metadata.checker_name is "simple_checker"
        assert checker_metadata.merit_name is "merit_with_simple_checker"

    merit_with_simple_checker()

def test_checker_metadata_auto_filled_no_merit_name():
    metadata = CheckerMetadata(actual="test", reference="test")

    # Auto-filled checker_name parsed test function name
    assert metadata.checker_name is "test_checker_metadata_auto_filled_no_merit_name"
    assert metadata.merit_name is None

def test_checker_actual_and_reference_truncated_in_repr():
    long_string_actual = "a" * 100
    long_string_reference = "b" * 100

    result = simple_checker(long_string_actual, long_string_reference)
    parsed_result_back_to_json = json.loads(repr(result))

    # Truncated actual and reference in repr
    assert parsed_result_back_to_json["checker"]["actual"] == long_string_actual[:50] + "..."
    assert parsed_result_back_to_json["checker"]["reference"] == long_string_reference[:50] + "..."

    # Original actual and reference are not truncated
    assert result.checker.actual == long_string_actual
    assert result.checker.reference == long_string_reference
