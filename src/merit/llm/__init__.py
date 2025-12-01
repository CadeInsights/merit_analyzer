"""LLM utilities for AI agents."""

from merit.llm.base import LLMProvider
from merit.llm.client import build_client
from merit.llm.config import AgentConfig, LLMConfig
from merit.llm.defaults import MAX_AGENT_TURNS, MAX_JSON_PARSING_ATTEMPTS, Agent, FileAccessLevel, Tool


__all__ = [
    # Core
    "AgentConfig",
    "LLMProvider",
    "LLMConfig",
    "build_client",
    # Types
    "Agent",
    "Tool",
    "FileAccessLevel",
    "MAX_AGENT_TURNS",
    "MAX_JSON_PARSING_ATTEMPTS",
]
