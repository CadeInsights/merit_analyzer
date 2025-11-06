"""Handle LLM calls here"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type, TypeVar

from pydantic import BaseModel

from .policies import AGENT, TOOL, FILE_ACCESS_POLICY

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", str, BaseModel)

# To add other LLM providers — inherit from the abstract LLM handler

class LLMAbstractHandler(ABC):
    """Abstract LLM handler"""

    compiled_agents: Dict[AGENT, Any]
    default_small_model: str
    default_big_model: str
    default_embedding_model: str

    @abstractmethod
    async def generate_embeddings(self, input_values: List[str], model: str | None = None) -> List[List[float]]:
        pass

    @abstractmethod
    async def create_object(self, prompt: str, schema: Type[T], model: str | None = None) -> T:
        pass

    @abstractmethod
    def compile_agent(
        self,
        agent_name: AGENT,
        system_prompt: str | None,
        model: str | None = None,
        file_access: FILE_ACCESS_POLICY = FILE_ACCESS_POLICY.READ_ONLY,
        standard_tools: List[TOOL] = [],
        extra_tools: List[Callable] = [],
        output_type: type[U] = str):
        pass

    @abstractmethod
    async def run_agent(
        self,
        agent: AGENT,
        task: str,
        output_type: type[U]
        ) -> U:
        pass