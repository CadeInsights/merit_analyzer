import json
from pathlib import Path

from .prompts import SYSTEM, TASK

from ...types import TestCaseGroup, ErrorAnalysis

from ...core import get_llm_client, AGENT, FILE_ACCESS_POLICY, TOOL, dataclass_to_xml


class ErrorAnalyzer:
    name = AGENT.ERROR_ANALYZER
    file_access = FILE_ACCESS_POLICY.READ_ONLY
    system_prompt = SYSTEM
    output_type = ErrorAnalysis
    standard_tools = [TOOL.GLOB, TOOL.GREP, TOOL.LS, TOOL.READ]

    async def run(self, failed_group: TestCaseGroup) -> ErrorAnalysis:
        """Analyze failed test groups and provide solutions for each"""

        client = await get_llm_client()

        if self.name not in client.compiled_agents:
            client.compile_agent(
                agent_name=self.name,
                file_access=self.file_access,
                output_type=self.output_type,
                standard_tools=self.standard_tools,
                system_prompt=self.system_prompt,
                cwd=Path.cwd(),
            )

        schema = json.dumps(ErrorAnalysis.model_json_schema(), indent=2)

        task = TASK.format(
            name=failed_group.metadata.name,
            description=failed_group.metadata.description,
            schema=schema,
            samples="\n".join([dataclass_to_xml(c) for c in failed_group.test_cases]),
        )

        error_analyses = await client.run_agent(self.name, task, self.output_type)

        return error_analyses
