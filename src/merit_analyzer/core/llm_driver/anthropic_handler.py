from typing import Any, Callable, Dict, List, Type, TypeVar, get_type_hints, cast

from dotenv import load_dotenv
from pydantic import BaseModel, create_model
from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    create_sdk_mcp_server,
    tool
)

from .abstract_provider_handler import LLMAbstractHandler, U, T
from .local_tools import read, write, edit, grep, glob, ls, todo
from .policies import AGENT, TOOL, FILE_ACCESS_POLICY

load_dotenv()

class LLMClaude(LLMAbstractHandler):
    default_small_model = "claude-haiku-4-5"
    default_big_model = "claude-sonnet-4-5"
    default_embedding_model = ""
    standard_tools_map = {
        TOOL.READ: "Read",
        TOOL.WRITE: "Write",
        TOOL.EDIT: "Edit",
        TOOL.GREP: "Grep",
        TOOL.GLOB: "Glob",
        TOOL.BASH: "Bash",
        TOOL.WEB_FETCH: "WebFetch",
        TOOL.WEB_SEARCH: "WebSearch",
        TOOL.TODO_WRITE: "TodoWrite",
        TOOL.BASH_OUTPUT: "BashOutput",
        TOOL.KILL_BASH: "KillBash",
        TOOL.LIST_MCP_RESOURCES: "ListMcpResources",
        TOOL.READ_MCP_RESOURCE: "ReadMcpResource",
        TOOL.LS: "LS",
        TOOL.TASK: "Task",
        TOOL.SLASH_COMMAND: "SlashCommand",
    }
    file_access_map = {
        FILE_ACCESS_POLICY.READ_ONLY: "default",
        FILE_ACCESS_POLICY.READ_AND_WRITE: "acceptEdits",
        FILE_ACCESS_POLICY.FULL_ACCESS: "bypassPermissions",
        FILE_ACCESS_POLICY.READ_AND_PLAN: "plan",
    }
    compiled_agents: Dict[AGENT, ClaudeAgentOptions] = {}

    def __init__(self, client: Anthropic | AnthropicBedrock | AnthropicVertex):
        self.client = client

    async def generate_embeddings(
            self, 
            input_values: List[str], 
            model: str | None = None
            ) -> List[List[float]]:
        model = model or self.default_big_model 
      
        raise ValueError("Not implemented yet")

    async def create_object(
            self, 
            prompt: str, 
            schema: Type[T], 
            model: str | None = None
            ) -> T:
        model = model or self.default_big_model 

        raise ValueError("Not implemented yet")

    def compile_agent(
        self,
        agent_name: AGENT,
        system_prompt: str | None,
        model: str | None = None,
        file_access: FILE_ACCESS_POLICY = FILE_ACCESS_POLICY.READ_ONLY,
        standard_tools: List[TOOL] = [],
        extra_tools: List[Callable] = [],
        output_type: type[U] = str):

        model = model or self.default_big_model 
        
        allowed_tools = [self.standard_tools_map[tool] for tool in standard_tools]
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

        file_access_policy = self.file_access_map[file_access]

        self.compiled_agents[agent_name] = ClaudeAgentOptions(
            model=model,
            allowed_tools=allowed_tools,
            mcp_servers=mcp_servers, #type: ignore
            permission_mode=file_access_policy, #type: ignore
            system_prompt=system_prompt,
        )
        return
    
    async def run_agent(
        self,
        agent: AGENT,
        task: str,
        output_type: type[U] = str
        ) -> U:

        options = self.compiled_agents[agent]
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

        if not client_response:
            raise ValueError
                    
        if isinstance(client_response, output_type):
            return cast(U, client_response)
        
        elif issubclass(output_type, BaseModel) and isinstance(client_response, str):
            parsed = await self.create_object(
                model="",
                schema=output_type,
                prompt="")
            return parsed
        
        else:
            raise TypeError(f"Client output can't be parsed as {output_type}")
        