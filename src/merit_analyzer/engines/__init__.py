"""Legacy engines module - deprecated in favor of core.llm_driver."""

from dotenv import load_dotenv

load_dotenv()

# Legacy import for backwards compatibility
# New code should use: from merit_analyzer.core.llm_driver import get_llm_client
from .llm_client import LLMOpenAI

# Don't initialize client at import time - breaks when API keys not set
llm_client = None

__all__ = ["llm_client", "LLMOpenAI"]