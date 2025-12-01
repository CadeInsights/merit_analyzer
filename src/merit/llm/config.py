"""LLM configuration."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from merit.llm.defaults import MAX_AGENT_TURNS, FileAccessLevel, Tool


OutputT = TypeVar("OutputT", bound=BaseModel | str)


class LLMConfig(BaseSettings):
    """Configuration for LLM provider initialization.

    Loads from environment variables automatically:
        MODEL_VENDOR, INFERENCE_VENDOR, CLOUD_ML_REGION, GOOGLE_CLOUD_PROJECT

    Or pass values directly to build_client().
    """

    model_vendor: Literal["anthropic", "openai"] = Field(description="The model vendor (anthropic or openai)")
    inference_vendor: Literal["anthropic", "openai", "aws", "gcp"] = Field(
        description="Where to run inference (direct API or cloud provider)"
    )
    region: str | None = Field(
        default=None, validation_alias="CLOUD_ML_REGION", description="Cloud region (for GCP Vertex AI)"
    )
    project_id: str | None = Field(
        default=None, validation_alias="GOOGLE_CLOUD_PROJECT", description="Cloud project ID (for GCP Vertex AI)"
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        env_prefix="",
    )


SUPPORTED_COMBINATIONS = {
    "openai": ["openai"],
    "anthropic": ["anthropic", "gcp", "aws"],
}


@dataclass
class AgentConfig(Generic[OutputT]):
    """Configuration for an agent run.

    Subclass this to define reusable agent configurations:

        @dataclass
        class MyAnalyzer(AgentConfig[AnalysisResult]):
            system_prompt: str = "You are an analyzer..."
            tools: list[Tool] = field(default_factory=lambda: [Tool.READ, Tool.GREP])
            output_type: type[AnalysisResult] = AnalysisResult
    """

    system_prompt: str
    tools: list[Tool] = field(default_factory=list)
    extra_tools: list[Callable[..., Any]] = field(default_factory=list)
    file_access: FileAccessLevel = FileAccessLevel.READ_ONLY
    output_type: type[OutputT] = str  # type: ignore[assignment]
    cwd: Path | None = None
    max_turns: int = MAX_AGENT_TURNS
