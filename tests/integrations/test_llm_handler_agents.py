import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from pydantic import BaseModel, Field

import pytest
from agents.tool_context import ToolContext

from merit_analyzer.core.llm_driver.openai_handler import LLMOpenAI
from merit_analyzer.core.llm_driver.policies import AGENT, FILE_ACCESS_POLICY, TOOL


class AgentResponse(BaseModel):
    read_file_name: str = Field(description="Exact file name with file extention without path")
    write_file_name: str = Field(description="Exact file name with file extention without path")
    code_file_name: str = Field(description="Exact file name with file extention without path")


@pytest.mark.asyncio
async def test_openai_handler_agent_writes_access_code(monkeypatch: pytest.MonkeyPatch) -> None:
    system_instructions = "TEST agent for exercising local READ/WRITE operations."
    task = """
        Find a local file that contains the word California and read the access code there.
        Then find a local file that contains the word Montana and write the access code there.
        Then in the same folder with the Montana file find there is another file with another code.
        Remember the name if this file.

        After you done locating and editing files, return report:
        - name of the file with the access code inside
        - name of the file that was edited to add the code
        - name of the file with another code in the filename
        """

    handler = LLMOpenAI(open_ai_client=MagicMock())
    handler.compile_agent(
        agent_name=AGENT.TEST,
        system_prompt=system_instructions,
        file_access=FILE_ACCESS_POLICY.READ_AND_WRITE,
        standard_tools=[TOOL.READ, TOOL.EDIT, TOOL.GLOB, TOOL.GREP, TOOL.LS],
        output_type=AgentResponse,
    )

    response = await handler.run_agent(AGENT.TEST, task, str)
    assert isinstance(response, AgentResponse)
    assert response.read_file_name == "read_test.md"
    assert response.write_file_name == "write_test.md"
    assert response.code_file_name == "code_is_Texas.md"
    write_lines = (Path("tests/sample_data") / "write_test.md").read_text().splitlines()
    assert len(write_lines) >= 3
    assert write_lines[2] == "Code: Yosemite"
