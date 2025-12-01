"""Base classes for LLM providers."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from merit.llm.config import AgentConfig, OutputT


ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Implement this to add support for a new LLM vendor (Anthropic, OpenAI, etc.).
    """

    default_model: str
    default_small_model: str

    @abstractmethod
    async def run(self, agent: AgentConfig[OutputT], task: str) -> OutputT:
        """Run an agent with the given task.

        Args:
            agent: Agent configuration (system prompt, tools, output type, etc.)
            task: The task/prompt to send to the agent

        Returns:
            The agent's response, typed according to agent.output_type
        """

    @abstractmethod
    async def generate_structured(self, prompt: str, schema: type[ModelT], model: str | None = None) -> ModelT:
        """Generate a structured object from a prompt.

        Args:
            prompt: The prompt to send to the LLM
            schema: Pydantic model class for the expected output
            model: Optional model override

        Returns:
            Validated instance of the schema
        """

    @abstractmethod
    async def generate_embeddings(self, inputs: list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings for input texts.

        Args:
            inputs: List of texts to embed
            model: Optional model override

        Returns:
            List of embedding vectors
        """
