"""Code analysis engine using Claude Agent SDK for root cause analysis."""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from ..types import AssertionStateGroup


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
        
        # Lazy import Claude SDK components
        self._ClaudeSDKClient = None
        self._ClaudeAgentOptions = None
        self._tool = None
        self._create_sdk_mcp_server = None

    def _lazy_import_sdk(self):
        """Lazy import Claude Agent SDK to avoid hanging on module import."""
        if self._ClaudeSDKClient is None:
            from claude_agent_sdk import (  # type: ignore
                ClaudeSDKClient,
                ClaudeAgentOptions,
                tool,
                create_sdk_mcp_server
            )
            self._ClaudeSDKClient = ClaudeSDKClient
            self._ClaudeAgentOptions = ClaudeAgentOptions
            self._tool = tool
            self._create_sdk_mcp_server = create_sdk_mcp_server

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
        self._lazy_import_sdk()
        
        # Define custom tool ONCE using @tool decorator (outside loop)
        # Use simple type mapping per docs recommendation
        @self._tool(
            "submit_analysis",
            "REQUIRED: Call this tool to complete the analysis task and submit your findings about the test failure",
            {
                "root_cause": str,
                "problematic_code": str,
                "recommendations": str  # JSON string of array
            }
        )
        async def submit_analysis_handler(args):
            """Handler for submit_analysis tool - returns confirmation."""
            return {
                "content": [{
                    "type": "text",
                    "text": f"Analysis submitted: {args.get('root_cause', 'N/A')}"
                }]
            }
        
        # Create SDK MCP server with the custom tool (once)
        analyzer_server = self._create_sdk_mcp_server(
            name="analyzer",
            version="1.0.0",
            tools=[submit_analysis_handler]
        )
        
        
        results = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        # Analyze each group independently using ClaudeSDKClient
        for group in groups:
            try:
                # Build minimal prompt
                prompt = self._build_minimal_prompt(group)
                
                # Configure options with MCP server
                options = self._ClaudeAgentOptions(
                    cwd=str(self.project_path),
                    mcp_servers={"analyzer": analyzer_server},
                    allowed_tools=["Read", "Grep", "Glob", "mcp__analyzer__submit_analysis"],
                    max_turns=8,
                    model=self.model,
                    stderr=lambda msg: None,  # Suppress errors
                    system_prompt="""You analyze test failures. After investigating, call mcp__analyzer__submit_analysis to complete the task."""
                )
                
                print(f"\n🔍 Analyzing group: {group.metadata.name}")
                
                # Use ClaudeSDKClient (not query()) for custom tools support
                response_text = ""
                tool_result = None
                
                async with self._ClaudeSDKClient(options=options) as client:
                    await client.query(prompt)
                    
                    async for message in client.receive_response():
                        # Track tokens
                        if hasattr(message, 'usage'):
                            usage = message.usage
                            if isinstance(usage, dict):
                                total_input_tokens += usage.get('input_tokens', 0)
                                total_output_tokens += usage.get('output_tokens', 0)
                        
                        # Extract tool calls and text
                        if hasattr(message, 'content'):
                            for block in message.content:
                                if hasattr(block, 'text'):
                                    response_text += block.text
                                # Extract structured data from tool call - check by class name
                                elif type(block).__name__ == 'ToolUseBlock':
                                    # Check for ANY variation of submit_analysis
                                    if 'submit_analysis' in block.name.lower():
                                        tool_result = block.input
                
                # Use tool result if available, otherwise parse text
                if tool_result:
                    # Parse recommendations from JSON string if needed
                    recs = tool_result.get('recommendations', [])
                    if isinstance(recs, str):
                        try:
                            import json
                            recs = json.loads(recs)
                            print(f"✅ Parsed {len(recs)} recommendations from JSON")
                        except Exception as e:
                            print(f"⚠️ Failed to parse recommendations: {e}")
                            recs = []
                    elif isinstance(recs, list):
                        print(f"✅ Received {len(recs)} recommendations as list")
                    
                    analysis_data = {
                        'root_cause': tool_result.get('root_cause', 'Unknown'),
                        'problematic_code': tool_result.get('problematic_code', 'Not found'),
                        'recommendations': recs
                    }
                    print(f"📦 Final analysis_data has {len(analysis_data['recommendations'])} recommendations")
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
                import traceback
                print(f"\n⚠️  Error analyzing group '{group.metadata.name}': {e}")
                traceback.print_exc()
                results.append(AnalysisResult(
                    group_name=group.metadata.name,
                    group_description=group.metadata.description,
                    root_cause=f'Error: {str(e)}',
                    problematic_code='Analysis failed',
                    recommendations=[],
                    relevant_tests=[]
                ))
        
        # Calculate cost (Claude Sonnet 4 pricing: $3/M input, $15/M output)
        input_cost = (total_input_tokens / 1_000_000) * 3.0
        output_cost = (total_output_tokens / 1_000_000) * 15.0
        total_cost = input_cost + output_cost
        
        print(f"\n📊 Token Usage:")
        print(f"   Input:  {total_input_tokens:,} tokens (${input_cost:.4f})")
        print(f"   Output: {total_output_tokens:,} tokens (${output_cost:.4f})")
        print(f"   Total:  ${total_cost:.4f}")
        
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
        
        prompt = f"""Test failure pattern: {group.metadata.name}
Failed tests: {len(group.assertion_states)}
Error: {sample_error}

Find the root cause, then call mcp__analyzer__submit_analysis with:
- root_cause: "file.py:line - description"
- problematic_code: the code snippet
- recommendations: JSON string like '[{{"type":"code","title":"Fix it","description":"how to fix","priority":"high","effort":"low"}}]'"""
        
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

