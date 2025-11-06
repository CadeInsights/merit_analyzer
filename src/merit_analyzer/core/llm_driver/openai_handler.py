from typing import Any, Callable, Dict, List, Type, TypeVar, get_type_hints, cast

from agents import Agent, Runner, function_tool
from openai import OpenAI
from dotenv import load_dotenv

from .abstract_provider_handler import LLMAbstractHandler, U, T
from .local_tools import read, write, edit, grep, glob, ls, todo
from .policies import AGENT, TOOL, FILE_ACCESS_POLICY

load_dotenv()

class LLMOpenAI(LLMAbstractHandler):
    """Handler for OpenAI models through Responses API"""
    default_small_model = "gpt-5-mini"
    default_big_model = "gpt-5"
    default_embedding_model = "text-embedding-3-small"
    standard_tools_map = {
        TOOL.READ: read,
        TOOL.WRITE: write,
        TOOL.EDIT: edit,
        TOOL.GREP: grep,
        TOOL.GLOB: glob,
        TOOL.BASH: None,
        TOOL.WEB_FETCH: "WebFetch",
        TOOL.WEB_SEARCH: "WebSearch",
        TOOL.TODO_WRITE: todo,
        TOOL.BASH_OUTPUT: "BashOutput",
        TOOL.KILL_BASH: "KillBash",
        TOOL.LIST_MCP_RESOURCES: None,
        TOOL.READ_MCP_RESOURCE: None,
        TOOL.LS: ls,
        TOOL.TASK: None,
        TOOL.SLASH_COMMAND: None,
    }

    def __init__(self, open_ai_client: OpenAI):
        self.client = open_ai_client
        self.compiled_agents: Dict[AGENT, Agent] = {}

    async def generate_embeddings(
            self, 
            input_values: List[str], 
            model: str | None = None
            ) -> List[List[float]]:
        model = model or self.default_embedding_model
        response = self.client.embeddings.create(
            model=model, 
            input=input_values
            )
        return [item.embedding for item in response.data]

    async def create_object(
            self, 
            prompt: str, 
            schema: Type[T], 
            model: str | None = None
            ) -> T:
        model = model or self.default_big_model
        response = self.client.responses.parse(
            model=model, 
            input=prompt, 
            text_format=schema
            )
        parsed = response.output_parsed
        if not parsed:
            raise ValueError("LLM didn't return any objects")
        return parsed

    def compile_agent(
        self,
        agent_name: AGENT,
        system_prompt: str | None,
        model: str | None = None,
        file_access: FILE_ACCESS_POLICY = FILE_ACCESS_POLICY.READ_ONLY,
        standard_tools: List[TOOL] = [],
        extra_tools: List[Callable] = [],
        output_type: type[U] = str
        ):
        model = model or self.default_big_model 
        tools = []
        for standard_tool in standard_tools:
            if standard_tool not in file_access.value:
                raise ValueError(
                    f"""Tool {standard_tool.name} doesn't comply with access policy {file_access.name}.
                    Change file access policy, or remove the tool from the given tools."""
                    )
            if standard_tool.value is None:
                raise ValueError(
                    f"""Tool {standard_tool.name} has not been implemented for the OpenAI client yet.
                    Remove the tool from given arguments, or implement it for the OpenAI handler.
                    """
                )
            parsed_tool = self.standard_tools_map[standard_tool]
            tools.append(function_tool(parsed_tool))

        for extra_tool in extra_tools:
            tools.append(function_tool(extra_tool))

        agent = Agent(
            name=agent_name.value,
            instructions=system_prompt,
            model=model,
            tools=tools,
            output_type=output_type, #type: ignore
        )
        self.compiled_agents[agent_name] = agent
        return

    async def run_agent(
        self,
        agent: AGENT,
        task: str,
        output_type: type[U]
    ) -> U:
        compiled = self.compiled_agents[agent]
        result = await Runner.run(compiled, input=task)
        return result.final_output_as(output_type)
