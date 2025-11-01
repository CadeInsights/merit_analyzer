"""Recommendation data models."""

from pydantic import BaseModel, Field  # type: ignore
from typing import Optional, List, Dict, Any
from enum import Enum


class PriorityLevel(str, Enum):
    """Priority levels for recommendations."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationType(str, Enum):
    """Types of recommendations."""
    CODE = "code"
    PROMPT = "prompt"
    ARCHITECTURE = "architecture"
    CONFIGURATION = "configuration"
    TESTING = "testing"


class Recommendation(BaseModel):
    """A single recommendation for fixing test failures."""

    priority: PriorityLevel = Field(..., description="Priority level")
    type: RecommendationType = Field(..., description="Type of recommendation")
    title: str = Field(..., description="Short title for the recommendation")
    description: str = Field(..., description="Detailed description of what to change")
    location: str = Field(..., description="File path and line numbers")
    implementation: Optional[str] = Field(None, description="Specific changes to make")
    expected_impact: Optional[str] = Field(None, description="Which tests this should fix")
    effort_estimate: str = Field(..., description="Time estimate (e.g., '5 minutes', '2 hours')")
    rationale: Optional[str] = Field(None, description="Why this will help")
    code_diff: Optional[str] = Field(None, description="Proposed code changes")
    before_after_examples: Optional[Dict[str, str]] = Field(
        None, description="Before/after examples"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="Other recommendations this depends on"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    def get_priority_score(self) -> int:
        """Get numeric priority score for sorting."""
        priority_scores = {
            PriorityLevel.HIGH: 3,
            PriorityLevel.MEDIUM: 2,
            PriorityLevel.LOW: 1,
        }
        return priority_scores.get(self.priority, 0)

    def to_markdown(self) -> str:
        """Convert recommendation to markdown format."""
        md = f"## {self.title}\n\n"
        md += f"**Priority:** {self.priority.value.title()}\n"
        md += f"**Type:** {self.type.value.title()}\n"
        md += f"**Location:** `{self.location}`\n"
        md += f"**Effort:** {self.effort_estimate}\n\n"
        md += f"### Description\n\n{self.description}\n\n"
        
        if self.rationale:
            md += f"### Rationale\n\n{self.rationale}\n\n"
        
        # Only show Implementation section if there's content
        if self.implementation:
            md += f"### Implementation\n\n{self.implementation}\n\n"
        
        if self.code_diff:
            md += f"### Code Changes\n\n```diff\n{self.code_diff}\n```\n\n"
        
        if self.before_after_examples:
            md += "### Before/After Examples\n\n"
            for key, value in self.before_after_examples.items():
                md += f"**{key}:**\n```\n{value}\n```\n\n"
        
        # Only show Expected Impact section if there's content
        if self.expected_impact:
            md += f"### Expected Impact\n\n{self.expected_impact}\n\n"
        
        if self.dependencies:
            md += f"### Dependencies\n\n- {chr(10).join(self.dependencies)}\n\n"
        
        return md
