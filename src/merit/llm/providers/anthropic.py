"""Anthropic LLM provider."""

from collections.abc import Callable
from typing import cast, get_type_hints

from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex
from anthropic.types import ToolParam, ToolUseBlock
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    create_sdk_mcp_server,
    tool,
)
from pydantic import BaseModel, ValidationError, create_model

from merit.llm.base import LLMProvider, ModelT
from merit.llm.config import AgentConfig, OutputT
from merit.llm.defaults import MAX_JSON_PARSING_ATTEMPTS, FileAccessLevel, Tool
from merit.llm.embeddings import MODEL_ID, local_embeddings


class AnthropicProvider(LLMProvider):
    """LLM provider for Anthropic Claude models."""

    default_embedding_model = MODEL_ID

    # Map our Tool enum to Claude SDK tool names
    _tool_name_map = {
        Tool.READ: "Read",
        Tool.WRITE: "Write",
        Tool.EDIT: "Edit",
        Tool.GREP: "Grep",
        Tool.GLOB: "Glob",
        Tool.BASH: "Bash",
        Tool.WEB_FETCH: "WebFetch",
        Tool.WEB_SEARCH: "WebSearch",
        Tool.TODO_WRITE: "TodoWrite",
        Tool.BASH_OUTPUT: "BashOutput",
        Tool.KILL_BASH: "KillBash",
        Tool.LIST_MCP_RESOURCES: "ListMcpResources",
        Tool.READ_MCP_RESOURCE: "ReadMcpResource",
        Tool.LS: "LS",
        Tool.TASK: "Task",
        Tool.SLASH_COMMAND: "SlashCommand",
    }

    _file_access_map = {
        FileAccessLevel.READ_ONLY: "default",
        FileAccessLevel.READ_AND_WRITE: "acceptEdits",
        FileAccessLevel.FULL_ACCESS: "bypassPermissions",
        FileAccessLevel.READ_AND_PLAN: "plan",
    }

    def __init__(
        self,
        client: Anthropic | AnthropicBedrock | AnthropicVertex,
        default_model: str = "claude-sonnet-4-5",
        default_small_model: str = "claude-haiku-4-5",
    ):
        self.client = client
        self.default_model = default_model
        self.default_small_model = default_small_model

    async def generate_embeddings(self, inputs: list[str], model: str | None = None) -> list[list[float]]:
        return await local_embeddings.generate_embeddings(inputs=inputs, model=model)

    async def generate_structured(self, prompt: str, schema: type[ModelT], model: str | None = None) -> ModelT:
        tools: list[ToolParam] = [
            {
                "name": "emit_structured_result",
                "description": "Return the result strictly as JSON matching input_schema.",
                "input_schema": schema.model_json_schema(),
            }
        ]

        last_error: ValidationError | None = None

        for _ in range(MAX_JSON_PARSING_ATTEMPTS):
            msg = self.client.messages.create(
                model=model or self.default_model,
                temperature=0,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                tool_choice={"type": "tool", "name": "emit_structured_result"},
            )
            tool_call = next(b for b in msg.content if isinstance(b, ToolUseBlock))

            try:
                return schema.model_validate(tool_call.input)
            except ValidationError as e:
                last_error = e
                continue

        raise last_error  # type: ignore[misc]

    async def run(self, agent: AgentConfig[OutputT], task: str) -> OutputT:
        options = self._build_agent_options(agent, extra_tools=agent.extra_tools or None)
        client_response = None

        async with ClaudeSDKClient(options=options) as sdk_client:
            await sdk_client.query(task)
            async for message in sdk_client.receive_response():
                match message:
                    case AssistantMessage():
                        continue
                    case ResultMessage(result=res):
                        client_response = res
                    case _:
                        continue

        if not client_response:
            raise ValueError("Agent returned no response")

        if isinstance(client_response, agent.output_type):
            return cast("OutputT", client_response)

        if issubclass(agent.output_type, BaseModel) and isinstance(client_response, str):
            return await self._parse_to_schema(client_response, agent.output_type)

        raise TypeError(f"Agent output cannot be parsed as {agent.output_type}")

    def _build_agent_options(
        self,
        agent: AgentConfig,
        extra_tools: list[Callable] | None = None,
    ) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from AgentConfig."""
        options = ClaudeAgentOptions(
            model=self.default_model,
            allowed_tools=[self._tool_name_map[t] for t in agent.tools],
            permission_mode=self._file_access_map[agent.file_access],  # type: ignore
            system_prompt=agent.system_prompt,
            cwd=agent.cwd,
            max_turns=agent.max_turns,
        )

        if extra_tools:
            parsed_tools = []
            for func in extra_tools:
                name = func.__name__
                description = func.__doc__ or ""
                input_schema = create_model(
                    f"InputSchema{name}",
                    **{n: (tp, ...) for n, tp in get_type_hints(func).items()},
                )
                parsed_tool = tool(name=name, description=description, input_schema=input_schema)(func)
                parsed_tools.append(parsed_tool)
                options.allowed_tools.append(f"mcp__extra_tools__{name}")

            options.mcp_servers = {"extra_tools": create_sdk_mcp_server(name="extra_tools", tools=parsed_tools)}

        return options

    async def _parse_to_schema(self, text: str, schema: type[ModelT]) -> ModelT:
        """Parse unstructured text into a Pydantic schema."""
        prompt = f"""
Transform the following text into JSON matching the schema.
Be careful with types and required fields.

<text>
{text}
</text>

<schema>
{schema.model_json_schema()}
</schema>
"""
        return await self.generate_structured(prompt=prompt, schema=schema, model=self.default_small_model)
