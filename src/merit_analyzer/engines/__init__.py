from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from .llm_client import LLMOpenAI

llm_client = LLMOpenAI(OpenAI())

__all__ = ["llm_client"]