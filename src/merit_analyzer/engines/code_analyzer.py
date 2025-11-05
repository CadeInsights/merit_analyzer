"""Code analysis engine using Claude Agent SDK for root cause analysis."""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field

from claude_agent_sdk import (  # type: ignore
    ClaudeSDKClient,
    ClaudeAgentOptions,
    create_sdk_mcp_server,
    tool,
)

from ..types import AssertionStateGroup

class Recommendation(BaseModel):
    type: Literal["code", "prompt", "config", "design"] = Field(
        description="Type of fix"
        )
    title: str = Field(
        description="Short action-oriented title (e.g., 'Add zero division check')"
        )
    description: str = Field(
        description="Complete explanation of the fix and why it's needed"
        )
    file: str = Field(
        description="File path where the fix should be applied"
        )
    line_number: str = Field(
        description="Line number(s) to modify (e.g., '6' or '6-8')"
        )
    current_code: str = Field(
        description="The current buggy code snippet"
        )
    fixed_code: str = Field(
        description="The corrected code that fixes the issue"
        )
    priority: Literal["high", "medium", "low"] = Field(
        description="Priority level"
        )
    effort: Literal["high", "medium", "low"] = Field(
        description="Implementation effort"
        )

class Diagnosis(BaseModel):
    root_cause: str = Field(
        description="The root cause with file:line reference (e.g., 'calculator.py:35 - Missing zero check causes ZeroDivisionError')"
        )
    problematic_code: str = Field(
        description="The exact code snippet that is causing the problem"
        )
    recommendations: List[Recommendation] = Field(
        min_length=1, 
        max_length=3, 
        description="List of 1-3 actionable recommendations with actual code fixes"
        )


@dataclass
class AnalysisResult:
    """Result of analyzing an error group."""
    group_name: str
    group_description: str
    root_cause: str
    problematic_code: str
    recommendations: List[Dict[str, Any]]
    relevant_tests: List[str]


class CodeAnalyzer:
    """
    Stateful engine for analyzing error groups using Claude Agent SDK.
    
    Takes clustered test failures and uses Claude to:
    1. Find the problematic code
    2. Determine root cause
    3. Generate fix recommendations
    """

    def __init__(self, project_path: str, api_key: str, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the code analyzer.

        Args:
            project_path: Path to the project to analyze
            api_key: Anthropic API key
            model: Claude model to use
        """
        self.project_path = Path(project_path)
        self.api_key = api_key
        self.model = model

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Claude's response to extract JSON."""
        # Try to extract JSON from the response
        try:
            # Look for JSON object
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
        except Exception:
            pass
        
        # Fallback if parsing fails
        return {
            'root_cause': 'Unable to determine root cause',
            'problematic_code': response_text[:500],
            'recommendations': []
        }

    async def analyze_multiple_groups(self, groups: List[AssertionStateGroup]) -> List[AnalysisResult]:
        """
        Analyze multiple error groups using ClaudeSDKClient with custom tools.
        
        Uses ClaudeSDKClient (not query()) to support custom tools via MCP server.

        Args:
            groups: List of error groups from clustering

        Returns:
            List of analysis results
        """
        # Define custom tool ONCE using @tool decorator (outside loop)
        # Use FULL JSON Schema for better Claude understanding
        schema = Diagnosis.model_json_schema()
        @tool(
            "submit_analysis",
            "REQUIRED: You MUST call this tool when you have completed your investigation and are ready to submit your final analysis of the test failure. This is the ONLY way to complete the task.",
            schema
        )
        async def submit_analysis_handler(args):
            """Handler for submit_analysis tool - returns confirmation."""
            return {
                "content": [{
                    "type": "text",
                    "text": f"✅ Analysis submitted successfully: {args.get('root_cause', 'N/A')}"
                }]
            }
        
        # Create SDK MCP server with the custom tool (once)
        analyzer_server = create_sdk_mcp_server(
            name="analyzer",
            version="1.0.0",
            tools=[submit_analysis_handler]
        )
        
        results = []
        
        # Define hook to remind Claude before stopping
        async def ensure_tool_call_hook(input_data, tool_use_id, context):
            """Remind Claude to call submit_analysis before completing."""
            return {
                'systemMessage': 'REMINDER: You must call mcp__analyzer__submit_analysis to submit your findings before ending.'
            }
        
        # Configure options with MCP server (ONCE for all clusters)
        options = ClaudeAgentOptions(
            cwd=str(self.project_path),
            mcp_servers={"analyzer": analyzer_server},
            allowed_tools=["Read", "Grep", "Glob", "mcp__analyzer__submit_analysis"],
            max_turns=8,
            model=self.model,
            stderr=lambda msg: None,  # Suppress errors
            hooks={
                'Stop': [{'hooks': [ensure_tool_call_hook]}]
            },
            system_prompt="""You are a code debugger. Your task is to:
1. Use Grep to search for relevant code
2. Use Read to examine the problematic files
3. Identify the root cause with file:line reference
4. Create fix recommendations with ACTUAL CODE CHANGES (before/after)
5. MANDATORY: Call mcp__analyzer__submit_analysis with your findings

For each recommendation, you MUST provide:
- The specific file and line number to change
- The current buggy code snippet
- The fixed code that resolves the issue

YOU MUST call mcp__analyzer__submit_analysis before the conversation ends. This is REQUIRED to complete the task."""
        )
        
        # Create ONE ClaudeSDKClient for all clusters to maintain context
        async with ClaudeSDKClient(options=options) as client:
            # Analyze each group with shared context
            for group in groups:
                try:
                    # Build minimal prompt
                    prompt = self._build_minimal_prompt(group)

                    # Use the same client - maintains context!
                    response_text = ""
                    tool_result = None
                    tool_was_called = False
                    await client.query(prompt)
                    
                    async for message in client.receive_response():
                        # Extract tool calls and text
                        if hasattr(message, 'content'):
                            for block in message.content:
                                if hasattr(block, 'text'):
                                    response_text += block.text
                                # Extract structured data from tool call - check by class name
                                elif type(block).__name__ == 'ToolUseBlock':
                                    # Check for ANY variation of submit_analysis
                                    if 'submit_analysis' in block.name.lower():
                                        tool_was_called = True
                                        tool_result = block.input
                    
                    # Safety net: Retry if tool wasn't called on first attempt
                    if not tool_was_called:
                        await client.query("Please call the mcp__analyzer__submit_analysis tool NOW with your findings.")
                        
                        async for message in client.receive_response():
                            if hasattr(message, 'content'):
                                for block in message.content:
                                    if hasattr(block, 'text'):
                                        response_text += block.text
                                    elif type(block).__name__ == 'ToolUseBlock':
                                        if 'submit_analysis' in block.name.lower():
                                            tool_was_called = True
                                            tool_result = block.input
                        
                    # Use tool result if available, otherwise parse text
                    if tool_result:
                        # With new schema, recommendations should already be an array
                        recs = tool_result.get('recommendations', [])
                        # Handle legacy string format if needed
                        if isinstance(recs, str):
                            try:
                                recs = json.loads(recs)
                            except:
                                recs = []
                        # Ensure it's a list
                        if not isinstance(recs, list):
                            recs = []
                        
                        analysis_data = {
                            'root_cause': tool_result.get('root_cause', 'Unknown'),
                            'problematic_code': tool_result.get('problematic_code', 'Not found'),
                            'recommendations': recs
                        }
                    else:
                        # Fallback to text parsing
                        analysis_data = self._parse_response(response_text)
                    
                    # Extract relevant test info
                    relevant_tests = [
                        f"Input: {state.test_case.input_value}, Expected: {state.test_case.expected}, Got: {state.return_value}"
                        for state in group.assertion_states[:3]
                    ]
                    
                    results.append(AnalysisResult(
                        group_name=group.metadata.name,
                        group_description=group.metadata.description,
                        root_cause=analysis_data.get('root_cause', 'Unknown'),
                        problematic_code=analysis_data.get('problematic_code', 'Not found'),
                        recommendations=analysis_data.get('recommendations', []),
                        relevant_tests=relevant_tests
                    ))
                    
                except Exception as e:
                    results.append(AnalysisResult(
                        group_name=group.metadata.name,
                        group_description=group.metadata.description,
                        root_cause=f'Error: {str(e)}',
                        problematic_code='Analysis failed',
                        recommendations=[],
                        relevant_tests=[]
                    ))
        
        return results
    
    def _build_minimal_prompt(self, group: AssertionStateGroup) -> str:
        """Build minimal prompt for this group to save tokens."""
        # Get sample error (just first one)
        sample_error = ""
        if group.assertion_states:
            state = group.assertion_states[0]
            error_msg = state.test_case.error_message or 'Test failed'
            if len(error_msg) > 150:
                error_msg = error_msg[:150] + "..."
            sample_error = error_msg
        
        prompt = f"""Analyze this test failure:

Pattern: {group.metadata.name}
Failed tests: {len(group.assertion_states)}
Error: {sample_error}

Steps:
1. Use Grep to find relevant code
2. Use Read to examine the buggy code
3. Identify the root cause

THEN YOU MUST call mcp__analyzer__submit_analysis with:
- root_cause: "file.py:line - specific issue description"
- problematic_code: "the exact code snippet causing the problem"
- recommendations: array of fix objects, where EACH recommendation MUST include:
  * type: "code" | "prompt" | "config" | "design"
  * title: short description
  * description: why this fix is needed
  * file: the file path to modify
  * line_number: which line(s) to change (e.g., "6" or "6-8")
  * current_code: the buggy code snippet
  * fixed_code: the corrected code that fixes the issue
  * priority: "high" | "medium" | "low"
  * effort: "high" | "medium" | "low"

CRITICAL: For code-type recommendations, you MUST provide the actual fixed code, not just a description.

DO NOT just describe the fix - you MUST call the tool to submit your analysis."""
        
        return prompt


def analyze_groups(
    groups: List[AssertionStateGroup],
    project_path: str,
    api_key: str,
    model: str = "claude-sonnet-4-20250514"
) -> List[AnalysisResult]:
    """
    Convenience function to analyze error groups synchronously.

    Args:
        groups: Error groups from clustering
        project_path: Path to project
        api_key: Anthropic API key
        model: Claude model to use

    Returns:
        List of analysis results
    """
    analyzer = CodeAnalyzer(project_path, api_key, model)
    
    # Run async analysis
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(analyzer.analyze_multiple_groups(groups))
    finally:
        loop.close()
