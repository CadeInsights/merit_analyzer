"""Handle LLM calls here"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from .policies import AGENT, FILE_ACCESS_POLICY, TOOL


ModelT = TypeVar("ModelT", bound=BaseModel)

# To add other LLM providers — inherit from the abstract LLM handler


class LLMAbstractHandler(ABC):
    """Abstract LLM handler"""

    compiled_agents: dict[AGENT, Any]
    default_small_model: str
    default_big_model: str
    default_embedding_model: str

    @abstractmethod
    async def generate_embeddings(self, input_values: list[str], model: str | None = None) -> list[list[float]]:
        pass

    @abstractmethod
    async def create_object(self, prompt: str, schema: type[ModelT], model: str | None = None) -> ModelT:
        pass

    @abstractmethod
    def compile_agent(
        self,
        agent_name: AGENT,
        system_prompt: str | None,
        model: str | None = None,
        file_access: FILE_ACCESS_POLICY = FILE_ACCESS_POLICY.READ_ONLY,
        standard_tools: list[TOOL] = [],
        extra_tools: list[Callable] = [],
        cwd: str | Path | None = None,
        output_type: type[ModelT] | type[str] = str,
    ):
        pass

    @overload
    async def run_agent(self, agent: AGENT, task: str, output_type: type[str]) -> str: ...

    @overload
    async def run_agent(self, agent: AGENT, task: str, output_type: type[ModelT]) -> ModelT: ...

    @abstractmethod
    async def run_agent(
        self, agent: AGENT, task: str, output_type: type[BaseModel] | type[str], max_turns: int | None = None
    ) -> BaseModel | str:
        pass
