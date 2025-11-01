from __future__ import annotations

from pydantic import BaseModel, Field
from dataclasses import dataclass, asdict
from typing import Any, List, Literal, Tuple

from .test_case import TestCase

# Core objects

@dataclass
class AssertionState:
    test_case: TestCase
    return_value: Any
    passed: bool
    confidence: float
    failure_reason: StateFailureReason | None

@dataclass
class AssertionStateGroup:
    metadata: StateGroupMetadata
    assertion_states: List[AssertionState]
    grouped_by: Literal["failed", "passed"]
    
# Models for generating data with LLMs

class StateGroupMetadata(BaseModel):
    name: str = Field(description="name for the cluster formatted in screaming snake case (e.g, INCORRECT_PRICE_PARSING)")
    description: str = Field(description="What happens in which circumstances.")

class StateFailureReason(BaseModel):
    analysis: str = Field(description="Explain briefly why did AI system return values that don't match the expected one")