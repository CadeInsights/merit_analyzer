"""Handle LLM calls here"""

from openai import OpenAI
from typing import List, Type, TypeVar, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)
U = TypeVar("U")

# To add other LLM providers — inherit from the abstract LLM handler

class LLMAbstractHandler(ABC):
    """Abstract LLM handler"""

    @abstractmethod
    async def generate_embeddings(self, model: str, input_values: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    async def create_object(self, model: str, prompt: str, schema: Type[T]) -> T:
        pass

    @abstractmethod
    async def run_agent(self, agent_name: str, agent_input: Any, output_type: U) -> U:
        pass



class LLMOpenAI(LLMAbstractHandler):
    """Handler for OpenAI models through Responses API"""

    def __init__(self, open_ai_client: OpenAI):
        self.client = open_ai_client

    async def generate_embeddings(self, model: str, input_values: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=model,
            input=input_values
        )
        return [item.embedding for item in response.data]
    
    async def create_object(self, model: str,  prompt: str, schema: Type[T]) -> T:
        response = self.client.responses.parse(
                model=model,
                input=prompt,
                text_format=schema
            )
        object = response.output_parsed

        if not object:
            raise ValueError("LLM didn't return any objects")

        return object
    
    async def run_agent(self, agent_name: str, agent_input: Any, output_type: U) -> U:
        raise ValueError("Not implemented yet")