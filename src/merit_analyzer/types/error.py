from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field

from .code import CodeComponent


class ErrorDescription(BaseModel):
    """Error messages enreached with context"""

    errors: List[str] = Field(description="List all problems with this AI system output that led to failed assertions")


class FrameInfo(BaseModel):
    """Frame for building Traceback"""

    index: int = Field(description="What's the order index of this frame")
    summary: str = Field(description="In less then 30 words explain what happened in this frame")


class ErrorSolution(BaseModel):
    """Solution on how to fix specific errors"""

    type: Literal["code", "prompt", "config", "design"] = Field(
        ...,
        description="Type of fix",
    )
    title: str = Field(
        ...,
        description="Short action-oriented title (e.g., 'Add zero division check')",
    )
    description: str = Field(
        ...,
        description="Complete explanation of the fix and why it's needed",
    )
    file: str = Field(
        ...,
        description="File path where the fix should be applied",
    )
    line_number: str = Field(
        ...,
        description="Line number(s) to modify (e.g., '6' or '6-8')",
    )


class ErrorAnalysis(BaseModel):
    """Analysis for a group of errors"""

    involved_code_components: List[CodeComponent] = Field(
        ...,
        description="The root cause with file:line reference (e.g., 'calculator.py:35 - Missing zero check causes ZeroDivisionError')",
    )
    traceback: List[FrameInfo] = Field(
        ...,
        description="The exact code snippet that is causing the problem",
    )
    recommendations: List[ErrorSolution] = Field(
        min_length=1,
        max_length=3,
        description="List of 1-3 actionable recommendations with actual code fixes",
    )
