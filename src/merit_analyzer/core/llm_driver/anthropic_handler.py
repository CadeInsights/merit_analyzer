from typing import Any, Callable, Dict, List, Type, TypeVar, get_type_hints, cast
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, create_model, ValidationError
from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex
from anthropic.types import ToolParam, MessageParam, ToolUseBlock
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    create_sdk_mcp_server,
    tool,
)

from .abstract_provider_handler import LLMAbstractHandler, ModelT
from .local_tools import read, write, edit, grep, glob, ls, todo
from .policies import AGENT, TOOL, FILE_ACCESS_POLICY

from ..local_models import local_embeddings_engine, MODEL_ID

load_dotenv()


class LLMClaude(LLMAbstractHandler):
    default_small_model = "claude-haiku-4-5"
    default_big_model = "claude-sonnet-4-5"
    default_embedding_model = MODEL_ID
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

    def __init__(self, client: Anthropic | AnthropicBedrock | AnthropicVertex):
        self.client = client
        self.compiled_agents: Dict[AGENT, ClaudeAgentOptions] = {}

    async def generate_embeddings(self, input_values: List[str], model: str | None = None) -> List[List[float]]:
        return await local_embeddings_engine.generate_embeddings(input_values=input_values)

    async def create_object(self, prompt: str, schema: Type[ModelT], model: str | None = None) -> ModelT:
        client = self.client
        tools: List[ToolParam] = [
            {
                "name": "emit_structured_result",
                "description": "Return the result strictly as JSON matching input_schema. No external effects.",
                "input_schema": schema.model_json_schema(),
            }
        ]
        msg = client.messages.create(
            model=model or self.default_big_model,
            temperature=0,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice={"type": "tool", "name": "emit_structured_result"},
        )
        tool_call = next(b for b in msg.content if isinstance(b, ToolUseBlock))
        try:
            # TODO: retry with same args if err (max 2 times)
            return schema.model_validate(tool_call.input)
        except ValidationError as e:
            raise

    def compile_agent(
        self,
        agent_name: AGENT,
        system_prompt: str | None,
        model: str | None = None,
        file_access: FILE_ACCESS_POLICY = FILE_ACCESS_POLICY.READ_ONLY,
        standard_tools: List[TOOL] = [],
        extra_tools: List[Callable] = [],
        output_type: type[ModelT] | type[str] = str,
        cwd: str | Path | None = None,
    ):
        agent_config = ClaudeAgentOptions(
            model=model or self.default_big_model,
            allowed_tools=[self.standard_tools_map[tool] for tool in standard_tools],
            permission_mode=self.file_access_map[file_access],  # type: ignore
            system_prompt=system_prompt,
            cwd=cwd,
        )

        parsed_tools = []

        for extra_tool in extra_tools:
            name = extra_tool.__name__
            description = extra_tool.__doc__ or ""
            input_schema = create_model(f"InputSchema{name}", **{n: (tp, ...) for n, tp in get_type_hints(extra_tool)})  # type: ignore
            parsed_tool = tool(name=name, description=description, input_schema=input_schema)(extra_tool)
            parsed_tools.append(parsed_tool)
            agent_config.allowed_tools.append(f"mcp__extra_tools__{name}")

        if parsed_tools:
            agent_config.mcp_servers = {"extra_tools": create_sdk_mcp_server(name="extra_tools", tools=parsed_tools)}

        self.compiled_agents[agent_name] = agent_config
        return

    async def run_agent(
        self, agent: AGENT, task: str, output_type: type[ModelT] | type[str] = str, max_turns: int | None = None
    ) -> ModelT | str:
        options = self.compiled_agents[agent]
        options.max_turns = max_turns
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
            return cast(ModelT, client_response)

        elif issubclass(output_type, BaseModel) and isinstance(client_response, str):
            prompt_template = f"""
                Your job is to transform the following text into a JSON and submit result
                using the 'emit_structured_result' tool. Be very careful with the JSON
                schema: read all field descriptions, check all required and optional types,
                and parse the data according to this schema.
                
                While parsing, you can rephrase / rewrite the original information to make it
                better align with the schema.

                <information_for_parsing>
                    {client_response}
                </information_for_parsing>

                <json_schema>
                    {output_type.model_json_schema()}
                </json_schema>
                """
            parsed = await self.create_object(
                model=self.default_small_model, schema=output_type, prompt=prompt_template
            )
            return parsed

        else:
            raise TypeError(f"Client output can't be parsed as {output_type}")
