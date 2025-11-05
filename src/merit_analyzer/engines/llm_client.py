"""Handle LLM calls here"""

from openai import OpenAI
from typing import List, Type, TypeVar, Callable, Dict, get_type_hints, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel, create_model
from enum import Enum
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    tool
)


T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", str, BaseModel)

class AGENT(Enum):
    CODING = "coding"

class TOOL(Enum):
    READ = "read"
    WRITE = "write"
    GREP = "grep"
    GLOB = "glob"

class FILE_ACCESS_POLICY(Enum):
    READ_ONLY = "read_only"
    READ_WRITE = ""


# To add other LLM providers — inherit from the abstract LLM handler

class LLMAbstractHandler(ABC):
    """Abstract LLM handler"""

    standard_tools: Dict[TOOL, Callable | str]
    file_access: Dict[FILE_ACCESS_POLICY, str]
    compiled_agents: Dict[AGENT, Any]

    @abstractmethod
    async def generate_embeddings(self, model: str, input_values: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    async def create_object(self, model: str, prompt: str, schema: Type[T]) -> T:
        pass

    @abstractmethod
    def compile_agent(
        self,
        agent_name: AGENT,
        system_prompt: str | None,
        permissions_config: dict, 
        standard_tools: List[TOOL] | None,
        extra_tools: List[Callable], 
        output_type: U = None
        ):
        pass

    @abstractmethod
    async def run_agent(self, agent: AGENT, task: str):
        pass

# Anthropic Handler

class LLMClaude(LLMAbstractHandler):
    standard_tools: Dict[TOOL, str] = {
        TOOL.READ: "Read",
        TOOL.WRITE: "Write",
        TOOL.GREP: "Grep",
        TOOL.GLOB: "Glob",
    }
    file_access = {
        FILE_ACCESS_POLICY.READ_ONLY: "acceptEdits"
    }
    agents: Dict[AGENT, ClaudeAgentOptions] = {}

    async def generate_embeddings(self, model: str, input_values: List[str]) -> List[List[float]]:
        raise ValueError("Not implemented yet")

    async def create_object(self, model: str, prompt: str, schema: Type[T]) -> T:
        raise ValueError("Not implemented yet")

    def compile_agent(
        self,
        agent_name: AGENT,
        system_prompt: str | None,
        model: str,
        file_access: FILE_ACCESS_POLICY,
        standard_tools: List[TOOL],
        extra_tools: List[Callable],
        output_type: U):
        
        allowed_tools = [self.standard_tools[tool] for tool in standard_tools]
        mcp_servers = None
        parsed_tools = []

        for extra_tool in extra_tools:
            name = extra_tool.__name__
            description = extra_tool.__doc__ or ""
            input_schema = create_model(
                f"InputSchema{name}", **{n: (tp, ...) for n, tp in get_type_hints(extra_tool)}) #type: ignore
            parsed_tool = tool(
                name=name, 
                description=description, 
                input_schema=input_schema)(extra_tool)
            parsed_tools.append(parsed_tool)
            allowed_tools.append(f"mcp__extra_tools__{name}")

        if parsed_tools:
            mcp_servers = {"extra_tools": create_sdk_mcp_server(name="extra_tools", tools=parsed_tools)}

        file_access_policy = self.file_access[file_access]

        self.agents[agent_name] = ClaudeAgentOptions(
            model=model,
            allowed_tools=allowed_tools,
            mcp_servers=mcp_servers, #type: ignore
            permission_mode=file_access_policy, #type: ignore
            system_prompt=system_prompt,
        )
        return
    
    async def run_agent[Y: BaseModel | str](
        self,
        agent: AGENT,
        task: str,
        output_type: type[Y]
        ) -> Y:
        options = self.agents[agent]
        client_response = None

        async with ClaudeSDKClient(options=options) as client:
            await client.query(task)
            async for message in client.receive_response():
                match message:
                    case AssistantMessage():
                        continue
                    case ResultMessage(result=res):
                        client_response = res
                    case _:
                        continue
                    
        match client_response, output_type:
            case (result, str) if isinstance(result, str):
                return result

            case (result, model_type) if issubclass(model_type, BaseModel):
                parsed = await self.create_object(
                    model="",
                    schema=model_type,
                    prompt="")
                return parsed

            case _:
                raise ValueError(f"Can't parse response into {output_type!r}")


class LLMOpenAI(LLMAbstractHandler):
    """Handler for OpenAI models through Responses API"""
    
    standard_tools = {}

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
    
    async def run_coding_agent(
        self,
        task: str, 
        system_prompt: str | None, 
        standard_tools: List[TOOL] | None,
        extra_tools: List[Callable], 
        output_type: U = None
        ) -> U:
        raise ValueError("Not implemented yet")
