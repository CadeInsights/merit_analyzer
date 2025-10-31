"""AI analysis engine integration for Merit Analyzer.

Architecture:
- Standard Anthropic API: schema discovery, architecture inference, pattern mapping (fast, scalable)
- Claude Agent SDK: pattern analysis ONLY (reads actual code for deep root cause analysis)
"""

import json
import re
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

# Anthropic API for fast standard LLM calls
from anthropic import Anthropic  # type: ignore

# Claude Agent SDK - imported lazily to avoid hanging on module import
# We only need it for pattern analysis, so we'll import it when needed
ClaudeSDKClient = None
ClaudeAgentOptions = None
CLINotFoundError = None
ProcessError = None
AssistantMessage = None

from ..models.test_result import TestResult
from ..core.config import MeritConfig


def _lazy_import_claude_sdk():
    """Lazy import of Claude Agent SDK to avoid hanging on module import."""
    global ClaudeSDKClient, ClaudeAgentOptions, CLINotFoundError, ProcessError, AssistantMessage
    if ClaudeSDKClient is None:
        try:
            from claude_agent_sdk import (  # type: ignore
                ClaudeSDKClient as _ClaudeSDKClient,
                ClaudeAgentOptions as _ClaudeAgentOptions,
                CLINotFoundError as _CLINotFoundError,
                ProcessError as _ProcessError,
                AssistantMessage as _AssistantMessage,
            )
            ClaudeSDKClient = _ClaudeSDKClient
            ClaudeAgentOptions = _ClaudeAgentOptions
            CLINotFoundError = _CLINotFoundError
            ProcessError = _ProcessError
            AssistantMessage = _AssistantMessage
        except Exception as e:
            raise RuntimeError(f"Failed to import Claude Agent SDK: {e}")


def _run_async(coro):
    """
    Run an async coroutine safely, avoiding nested event loop issues.
    
    Creates a new event loop to avoid conflicts.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(asyncio.sleep(0))  # Let pending tasks finish
        loop.close()


class MeritClaudeAgent:
    """
    Merit's AI analysis engine.
    
    Uses standard Anthropic API for fast inference tasks (schema, architecture, mapping).
    Uses Claude Agent SDK ONLY for pattern analysis (reading actual code files).
    """

    def __init__(self, config: MeritConfig):
        """
        Initialize Merit AI analysis engine.

        Args:
            config: Merit Analyzer configuration
        """
        self.config = config
        self.project_path = Path(config.project_path)
        self.api_key = config.api_key
        self.model = config.model
        self.provider = config.provider
        
        # Initialize Anthropic client for standard LLM calls
        # Used for: schema discovery, architecture discovery, pattern mapping
        self.anthropic_client = Anthropic(api_key=self.api_key)
        
        # Configure environment for Claude Agent SDK (used later for pattern analysis)
        if self.provider == "bedrock":
            os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
            aws_region = getattr(config, 'aws_region', None) or os.environ.get("AWS_REGION") or "us-east-1"
            os.environ["AWS_REGION"] = aws_region
        else:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                os.environ["ANTHROPIC_API_KEY"] = self.api_key
        
        # DON'T create ClaudeAgentOptions here - it hangs!
        # We'll create it lazily when we actually need pattern analysis
        
        # Cache for responses
        self._response_cache: Dict[str, str] = {}

    def _call_anthropic_direct(self, prompt: str, max_tokens: int = 4096) -> str:
        """
        Fast standard Anthropic API call (no agent, no tools).
        
        Used for: schema discovery, architecture inference, pattern mapping.
        These tasks don't need code access - just intelligent analysis.
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM response text
        """
        try:
            message = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract text from response
            response_text = ""
            for block in message.content:
                if hasattr(block, 'text'):
                    response_text += block.text
            
            return response_text
        except Exception as e:
            return f"Error: {str(e)}"

    # =======================
    # ARCHITECTURE DISCOVERY (Standard API)
    # =======================
    
    def discover_system_architecture(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discover system architecture using standard LLM inference.
        
        Analyzes project scan results (files, imports, frameworks) to infer
        the system architecture WITHOUT reading actual code files.

        Args:
            scan_results: Results from project scanning (file list, imports, entry points)

        Returns:
            Dict with agents, prompts, entry points, control flow
        """
        cache_key = f"architecture_{hash(str(scan_results))}"
        if cache_key in self._response_cache:
            return json.loads(self._response_cache[cache_key])
        
        # Build prompt for architecture inference
        prompt = f"""Analyze this AI system's project structure and infer its architecture.

PROJECT SCAN RESULTS:
{json.dumps(scan_results, indent=2)}

Based on the file structure, imports, and detected frameworks, provide your analysis in JSON format:

{{
    "system_type": "chatbot|rag|agent|code_generator|multi_modal|custom",
    "agents": [
        {{
            "name": "agent_name",
            "file": "path/to/file.py",
            "purpose": "inferred purpose",
            "key_methods": ["method1", "method2"]
        }}
    ],
    "prompts": [
        {{
            "name": "prompt_name",
            "location": "likely file or location",
            "purpose": "inferred purpose"
        }}
    ],
    "control_flow": {{
        "entry_points": ["main.py"],
        "flow": "description of likely data flow",
        "decision_points": ["point1", "point2"]
    }},
    "configuration": {{
        "api_keys": ["key1"],
        "models": ["model1"],
        "config_files": ["config.py"]
    }},
    "dependencies": [
        {{
            "name": "dependency",
            "type": "api|database|library"
        }}
    ]
}}

Return ONLY valid JSON, no explanation."""

        response = self._call_anthropic_direct(prompt)
        architecture = self._parse_architecture_response(response)
        
        # Cache the result
        self._response_cache[cache_key] = json.dumps(architecture)
        
        return architecture

    # =======================
    # PATTERN MAPPING (Standard API)
    # =======================
    
    def map_pattern_to_code(self, 
                           pattern_name: str,
                           test_examples: List[TestResult],
                           architecture: Dict[str, Any],
                           available_files: Optional[List[str]] = None) -> List[str]:
        """
        Map a failure pattern to relevant code locations using standard LLM inference.
        
        Intelligently selects which ACTUAL files from the codebase are relevant.

        Args:
            pattern_name: Name of the failure pattern
            test_examples: Example test cases
            architecture: System architecture information
            available_files: List of actual files in the project (from scan)

        Returns:
            List of file paths that likely relate to this pattern
        """
        cache_key = f"mapping_{pattern_name}_{len(test_examples)}"
        if cache_key in self._response_cache:
            return json.loads(self._response_cache[cache_key])
        
        # If no available files provided, return empty list (can't hallucinate files)
        if not available_files:
            return []
        
        # Build prompt for file mapping with ACTUAL file list
        examples = self._format_test_examples(test_examples[:2])
        files_list = "\n".join([f"  - {f}" for f in available_files[:50]])  # Limit to 50 files
        
        prompt = f"""Map this failure pattern to relevant code files from the ACTUAL project files.

PATTERN: {pattern_name}

FAILING TEST EXAMPLES:
{examples}

SYSTEM ARCHITECTURE:
{json.dumps(architecture, indent=2)}

ACTUAL PROJECT FILES (you MUST choose from this list):
{files_list}

Based on:
- The pattern name and failure characteristics
- The test inputs/outputs
- The known system architecture
- The ACTUAL files available in the project

Which 3-5 files from the ACTUAL PROJECT FILES list are most relevant to this failure?

Return ONLY the file paths from the list above, one per line.
Do NOT make up or hallucinate filenames.
Limit to 3-5 most relevant files."""

        response = self._call_anthropic_direct(prompt, max_tokens=500)
        
        # Parse and validate file paths (must be in available_files)
        suggested_paths = self._parse_file_list_response(response)
        valid_paths = [p for p in suggested_paths if p in available_files]
        
        # If LLM didn't return valid paths, do simple keyword matching
        if not valid_paths:
            valid_paths = self._fallback_file_matching(pattern_name, test_examples, available_files)
        
        # Cache the result
        self._response_cache[cache_key] = json.dumps(valid_paths[:5])
        
        return valid_paths[:5]
    
    def _fallback_file_matching(self, pattern_name: str, test_examples: List[TestResult], available_files: List[str]) -> List[str]:
        """Fallback: Simple keyword matching between pattern and filenames."""
        # Extract keywords from pattern name
        keywords = pattern_name.lower().replace('_', ' ').split()
        
        # Score each file based on keyword matches
        scored_files = []
        for file_path in available_files:
            filename = file_path.lower()
            score = sum(1 for keyword in keywords if keyword in filename)
            if score > 0:
                scored_files.append((score, file_path))
        
        # Return top scored files
        scored_files.sort(reverse=True, key=lambda x: x[0])
        return [f for _, f in scored_files[:5]]

    # =======================
    # PATTERN ANALYSIS (Claude Agent SDK - Code Reading)
    # =======================
    
    def analyze_pattern(self, 
                       pattern_name: str,
                       failing_tests: List[TestResult],
                       passing_tests: List[TestResult],
                       code_locations: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze a failure pattern using Claude Agent SDK to READ actual code.
        
        This is the ONLY method that uses Claude Agent SDK with code reading.
        Claude uses Read/Grep/Glob tools to navigate and analyze the codebase.

        Args:
            pattern_name: Name of the failure pattern
            failing_tests: Failed test cases
            passing_tests: Similar tests that are passing
            code_locations: Relevant code file paths (hints for Claude to explore)

        Returns:
            Dict with root cause and recommendations
        """
        cache_key = f"pattern_{pattern_name}_{len(failing_tests)}_{len(passing_tests)}"
        if cache_key in self._response_cache:
            return json.loads(self._response_cache[cache_key])
        
        # Use Claude Agent SDK async
        analysis = _run_async(self._analyze_pattern_async(
            pattern_name, failing_tests, passing_tests, code_locations
        ))
        
        # Cache the result
        self._response_cache[cache_key] = json.dumps(analysis)
        
        return analysis

    async def _analyze_pattern_async(self,
                                     pattern_name: str,
                                     failing_tests: List[TestResult],
                                     passing_tests: List[TestResult],
                                     code_locations: List[str]) -> Dict[str, Any]:
        """Async pattern analysis using Claude Agent SDK."""
        try:
            # Lazy import Claude SDK (only when we need it for code analysis)
            _lazy_import_claude_sdk()
            
            # Format test examples
            max_examples = 2
            fail_examples = self._format_test_examples(failing_tests[:max_examples])
            pass_examples = self._format_test_examples(passing_tests[:max_examples]) if passing_tests else "None"
            
            prompt = f"""You are an intelligent code analysis agent with direct access to this codebase via Read, Grep, and Glob tools.

PATTERN: {pattern_name}
FAILURE COUNT: {len(failing_tests)}
PASSING SIMILAR TESTS: {len(passing_tests)}

FAILING TEST EXAMPLES:
{fail_examples}

PASSING TEST EXAMPLES:
{pass_examples}

Your task: Find WHY these tests are failing by intelligently navigating the codebase.

**READ-ONLY MODE**: You can ONLY use Read, Grep, and Glob tools.

**STEP 1: Find Relevant Code**
- Use Grep to search for keywords from the test inputs/outputs (function names, class names, error messages)
- Use Glob to find files matching patterns (e.g., **/*agent*.py, **/*prompt*.py, **/*config*.py)
- Identify which files are most relevant to this failure pattern

**STEP 2: Analyze Root Cause**
- Use Read to examine the relevant files you found
- Trace the failure cascade through the code path
- Identify the exact code issue causing these failures

**STEP 3: Generate Recommendations**
Provide 2-3 specific, actionable fixes:
- Code changes (with exact file locations and line numbers if possible)
- Prompt changes (with exact text to modify)
- Configuration changes
- Design changes (new components/agents if needed)

Provide your analysis in JSON format:
{{
    "root_cause": "specific root cause from code analysis",
    "pattern_characteristics": {{
        "common_inputs": ["input1", "input2"],
        "common_failures": ["failure1", "failure2"]
    }},
    "code_issues": [
        {{
            "file": "path/to/file.py",
            "issue": "specific issue found in code",
            "line_number": 123
        }}
    ],
    "recommendations": [
        {{
            "type": "code|prompt|config|design",
            "title": "short title",
            "description": "detailed description",
            "file": "path/to/file",
            "priority": "high|medium|low",
            "effort": "high|medium|low"
        }}
    ]
}}

**Important**: Actually use Read/Grep tools to analyze the code. Don't guess."""
            
            # Create Claude Agent SDK options (lazily, only when needed)
            agent_options = ClaudeAgentOptions(
                cwd=str(self.project_path),
                allowed_tools=["Read", "Grep", "Glob"],
                disallowed_tools=["Write", "Edit", "Bash", "Task"],
                system_prompt=(
                    "You are an expert code analysis agent. "
                    "Use Read, Grep, and Glob tools to navigate the codebase and find root causes of test failures. "
                    "Trace failure cascades through code paths. Provide specific, actionable recommendations. "
                    "READ-ONLY MODE: Do NOT write, edit, or execute code."
                ),
                max_turns=15,
                model=self.model
            )
            
            # Create Claude Agent SDK client
            async with ClaudeSDKClient(options=agent_options) as client:
                await client.query(prompt)
                
                # Collect response
                full_response = ""
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage) and hasattr(message, 'content'):
                        for block in message.content:
                            if hasattr(block, 'text'):
                                full_response += block.text
                
                # Parse analysis
                analysis = self._parse_pattern_analysis_response(full_response)
                return analysis
                
        except Exception as e:
            return {
                "root_cause": f"Error analyzing pattern: {str(e)}",
                "pattern_characteristics": {"common_inputs": [], "common_failures": []},
                "code_issues": [],
                "recommendations": []
            }

    # =======================
    # HELPER METHODS
    # =======================

    def _format_test_examples(self, tests: List[TestResult]) -> str:
        """Format test examples for prompts."""
        formatted = []
        for i, test in enumerate(tests, 1):
            formatted.append(f"""
Example {i}:
  Input: {test.input}
  Expected: {test.expected_output or 'N/A'}
  Actual: {test.actual_output}
  Issue: {test.failure_reason or 'N/A'}
""")
        return "\n".join(formatted)

    def _parse_architecture_response(self, response: str) -> Dict[str, Any]:
        """Parse architecture analysis into structured format."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        
        # Fallback: return basic structure
        return {
            "system_type": "unknown",
            "agents": [],
            "prompts": [],
            "control_flow": {"entry_points": [], "flow": "", "decision_points": []},
            "configuration": {"api_keys": [], "models": [], "config_files": []},
            "dependencies": []
        }

    def _parse_pattern_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse pattern analysis into structured format."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        
        # Fallback: return basic structure
        return {
            "root_cause": "Unable to determine root cause",
            "pattern_characteristics": {"common_inputs": [], "common_failures": []},
            "code_issues": [],
            "recommendations": []
        }

    def _parse_file_list_response(self, response: str) -> List[str]:
        """Parse file paths from response."""
        lines = response.strip().split('\n')
        file_paths = []
        
        for line in lines:
            line = line.strip()
            # Look for file paths
            if line and ('.py' in line or '.txt' in line or '.md' in line or '.yaml' in line or '.json' in line):
                # Clean up the path
                path = line.split()[0] if ' ' in line else line
                path = path.strip('"\'')
                file_paths.append(path)
        
        return file_paths[:8]  # Limit to 8 files

    def clear_cache(self):
        """Clear the response cache."""
        self._response_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cached_responses": len(self._response_cache),
            "cache_size_mb": sum(len(v) for v in self._response_cache.values()) / (1024 * 1024)
        }
