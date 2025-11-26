"""OpenAI LLM provider."""

from typing import cast

from agents import Agent, Runner, function_tool
from openai import OpenAI

from merit.llm.config import AgentConfig, OutputT
from merit.llm.base import LLMProvider, ModelT
from merit.llm.tools import filesystem
from merit.llm.defaults import Tool


class OpenAIProvider(LLMProvider):
    """LLM provider for OpenAI models."""

    default_model = "gpt-4o"
    default_small_model = "gpt-4o-mini"
    default_embedding_model = "text-embedding-3-small"

    # Map our Tool enum to tool functions
    _tool_function_map = {
        Tool.READ: filesystem.read,
        Tool.WRITE: filesystem.write,
        Tool.EDIT: filesystem.edit,
        Tool.GREP: filesystem.grep,
        Tool.GLOB: filesystem.glob,
        Tool.LS: filesystem.ls,
        Tool.TODO_WRITE: filesystem.todo,
        # Not implemented
        Tool.BASH: None,
        Tool.WEB_FETCH: None,
        Tool.WEB_SEARCH: None,
        Tool.BASH_OUTPUT: None,
        Tool.KILL_BASH: None,
        Tool.LIST_MCP_RESOURCES: None,
        Tool.READ_MCP_RESOURCE: None,
        Tool.TASK: None,
        Tool.SLASH_COMMAND: None,
    }

    def __init__(self, client: OpenAI):
        self.client = client

    async def generate_embeddings(self, inputs: list[str], model: str | None = None) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=model or self.default_embedding_model,
            input=inputs,
        )
        return [item.embedding for item in response.data]

    async def generate_structured(self, prompt: str, schema: type[ModelT], model: str | None = None) -> ModelT:
        response = self.client.beta.chat.completions.parse(
            model=model or self.default_model,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,
        )
        parsed = response.choices[0].message.parsed
        if not parsed:
            raise ValueError("LLM returned no structured output")
        return parsed

    async def run(self, agent: AgentConfig[OutputT], task: str) -> OutputT:
        sdk_agent = self._build_agent(agent)
        result = await Runner.run(sdk_agent, input=task, max_turns=agent.max_turns)
        return cast(OutputT, result.final_output_as(agent.output_type))

    def _build_agent(self, agent: AgentConfig) -> Agent:
        """Build OpenAI Agent from AgentConfig."""
        tools = []

        for tool_enum in agent.tools:
            if tool_enum not in agent.file_access.value:
                raise ValueError(
                    f"Tool {tool_enum.name} not permitted by {agent.file_access.name} policy"
                )

            tool_func = self._tool_function_map.get(tool_enum)
            if tool_func is None:
                raise NotImplementedError(
                    f"Tool {tool_enum.name} not implemented for OpenAI provider"
                )

            tools.append(function_tool(tool_func))

        for extra_tool in agent.extra_tools:
            tools.append(function_tool(extra_tool))

        return Agent(
            name="agent",
            instructions=agent.system_prompt,
            model=self.default_model,
            tools=tools,
            output_type=agent.output_type,
        )
