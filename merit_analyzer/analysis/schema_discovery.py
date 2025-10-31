"""Schema discovery for universal AI system analysis."""

import json
from typing import List, Dict, Any, Optional
from ..models.test_result import TestResult
from ..analysis.claude_agent import MeritClaudeAgent


class SchemaDiscovery:
    """Discovers system schema and patterns from test data."""

    def __init__(self, claude_agent: MeritClaudeAgent):
        """
        Initialize schema discovery.
        
        Args:
            claude_agent: Claude agent for AI-powered analysis
        """
        self.claude_agent = claude_agent

    def discover_system_schema(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """
        Discover the system's data schema and patterns.
        
        Args:
            test_results: List of test results to analyze
            
        Returns:
            Schema information including system type, key fields, and patterns
        """
        if not test_results:
            return self._empty_schema()

        # Sample tests for analysis (mix of passed and failed)
        sample_tests = self._select_sample_tests(test_results)
        
        if not self.claude_agent:
            return self._fallback_schema_discovery(sample_tests)

        try:
            schema_info = self._ai_schema_discovery(sample_tests)
            return schema_info
        except Exception as e:
            print(f"Warning: AI schema discovery failed: {e}")
            return self._fallback_schema_discovery(sample_tests)

    def _select_sample_tests(self, test_results: List[TestResult], max_samples: int = 10) -> List[TestResult]:
        """Select representative sample of tests for schema analysis."""
        # Get mix of passed and failed tests
        passed_tests = [t for t in test_results if t.status == "passed"]
        failed_tests = [t for t in test_results if t.status == "failed"]
        
        # Sample from each category
        sample_size = min(max_samples, len(test_results))
        passed_sample = passed_tests[:sample_size // 2] if passed_tests else []
        failed_sample = failed_tests[:sample_size - len(passed_sample)] if failed_tests else []
        
        return passed_sample + failed_sample

    def _ai_schema_discovery(self, sample_tests: List[TestResult]) -> Dict[str, Any]:
        """Use AI to discover system schema and patterns."""
        prompt = f"""
Analyze these test results to understand the AI system's architecture and data patterns.

TEST RESULTS:
{self._format_tests_for_schema_analysis(sample_tests)}

Please provide a comprehensive analysis of:

1. SYSTEM TYPE: What kind of AI system is this? (chatbot, agent, RAG, code generator, image analyzer, etc.)

2. DATA SCHEMA: 
   - What are the key fields in inputs that matter?
   - What are the key fields in outputs that matter?
   - What data types and structures are used?

3. VALIDATION PATTERNS:
   - What patterns do you see in expected vs actual outputs?
   - What validation rules seem to be in place?
   - What are common success criteria?

4. FAILURE PATTERNS:
   - What types of failures are occurring?
   - What fields are commonly missing or incorrect?
   - What are the root causes of failures?

5. SYSTEM ARCHITECTURE HINTS:
   - What external services/APIs might be involved?
   - What are the main processing steps?
   - Where might the decision points be?

Provide a structured analysis that will help with pattern detection and root cause analysis.
Focus on what matters for understanding and fixing test failures.
"""

        response = self.claude_agent._call_anthropic_direct(prompt)
        return self._parse_schema_response(response)

    def _format_tests_for_schema_analysis(self, tests: List[TestResult]) -> str:
        """Format tests for schema analysis."""
        formatted = []
        for i, test in enumerate(tests, 1):
            formatted.append(f"""
Test {i} (ID: {test.test_id}, Status: {test.status}):
  Input: {json.dumps(test.input, indent=4)}
  Expected: {json.dumps(test.expected_output, indent=4) if test.expected_output else 'None'}
  Actual: {json.dumps(test.actual_output, indent=4)}
  Category: {test.category or 'None'}
  Tags: {test.tags or []}
""")
        return "\n".join(formatted)

    def _parse_schema_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response for schema information."""
        schema_info = {
            "system_type": "unknown",
            "key_input_fields": [],
            "key_output_fields": [],
            "validation_patterns": [],
            "failure_types": [],
            "data_types": {},
            "architecture_hints": [],
            "success_criteria": []
        }
        
        # Extract system type
        response_lower = response.lower()
        if "chatbot" in response_lower or "conversational" in response_lower:
            schema_info["system_type"] = "chatbot"
        elif "rag" in response_lower or "retrieval" in response_lower:
            schema_info["system_type"] = "rag"
        elif "code" in response_lower and "generator" in response_lower:
            schema_info["system_type"] = "code_generator"
        elif "agent" in response_lower:
            schema_info["system_type"] = "agent"
        elif "image" in response_lower or "vision" in response_lower:
            schema_info["system_type"] = "image_analyzer"
        elif "nlp" in response_lower or "language" in response_lower:
            schema_info["system_type"] = "nlp_processor"
        
        # Extract key fields (simplified parsing)
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            if "key fields" in line.lower() or "input fields" in line.lower():
                current_section = "input_fields"
            elif "output fields" in line.lower() or "response fields" in line.lower():
                current_section = "output_fields"
            elif "validation" in line.lower():
                current_section = "validation"
            elif "failure" in line.lower():
                current_section = "failure"
            elif "architecture" in line.lower():
                current_section = "architecture"
            
            # Extract field names (simple heuristic)
            elif current_section == "input_fields" and line.startswith('-'):
                field = line.replace('-', '').strip().split(':')[0].strip()
                if field and field not in schema_info["key_input_fields"]:
                    schema_info["key_input_fields"].append(field)
            elif current_section == "output_fields" and line.startswith('-'):
                field = line.replace('-', '').strip().split(':')[0].strip()
                if field and field not in schema_info["key_output_fields"]:
                    schema_info["key_output_fields"].append(field)
            elif current_section == "validation" and line.startswith('-'):
                pattern = line.replace('-', '').strip()
                if pattern and pattern not in schema_info["validation_patterns"]:
                    schema_info["validation_patterns"].append(pattern)
            elif current_section == "failure" and line.startswith('-'):
                failure_type = line.replace('-', '').strip()
                if failure_type and failure_type not in schema_info["failure_types"]:
                    schema_info["failure_types"].append(failure_type)
            elif current_section == "architecture" and line.startswith('-'):
                hint = line.replace('-', '').strip()
                if hint and hint not in schema_info["architecture_hints"]:
                    schema_info["architecture_hints"].append(hint)
        
        return schema_info

    def _fallback_schema_discovery(self, sample_tests: List[TestResult]) -> Dict[str, Any]:
        """Fallback schema discovery using heuristics."""
        schema_info = {
            "system_type": "unknown",
            "key_input_fields": [],
            "key_output_fields": [],
            "validation_patterns": [],
            "failure_types": [],
            "data_types": {},
            "architecture_hints": [],
            "success_criteria": []
        }
        
        if not sample_tests:
            return schema_info
        
        # Analyze input fields
        input_fields = set()
        output_fields = set()
        
        for test in sample_tests:
            if isinstance(test.input, dict):
                input_fields.update(test.input.keys())
            if isinstance(test.actual_output, dict):
                output_fields.update(test.actual_output.keys())
            if isinstance(test.expected_output, dict):
                output_fields.update(test.expected_output.keys())
        
        schema_info["key_input_fields"] = list(input_fields)
        schema_info["key_output_fields"] = list(output_fields)
        
        # Analyze failure patterns
        failure_types = set()
        for test in sample_tests:
            if test.status == "failed" and test.expected_output and test.actual_output:
                if isinstance(test.expected_output, dict) and isinstance(test.actual_output, dict):
                    expected_keys = set(test.expected_output.keys())
                    actual_keys = set(test.actual_output.keys())
                    
                    if expected_keys - actual_keys:
                        failure_types.add("missing_fields")
                    if actual_keys - expected_keys:
                        failure_types.add("unexpected_fields")
                    
                    # Check for value mismatches
                    for key in expected_keys.intersection(actual_keys):
                        if test.expected_output[key] != test.actual_output[key]:
                            failure_types.add("value_mismatch")
        
        schema_info["failure_types"] = list(failure_types)
        
        return schema_info

    def _empty_schema(self) -> Dict[str, Any]:
        """Return empty schema for no test results."""
        return {
            "system_type": "unknown",
            "key_input_fields": [],
            "key_output_fields": [],
            "validation_patterns": [],
            "failure_types": [],
            "data_types": {},
            "architecture_hints": [],
            "success_criteria": []
        }

    def generate_failure_context(self, test: TestResult, schema_info: Dict[str, Any]) -> str:
        """
        Generate failure context for a test using discovered schema.
        
        Args:
            test: Test result to analyze
            schema_info: Discovered schema information
            
        Returns:
            Generated failure context string
        """
        if test.failure_reason:
            return test.failure_reason
        
        context_parts = []
        
        # Add status information
        context_parts.append(f"status: {test.status}")
        
        # Analyze input/output comparison
        if test.expected_output and test.actual_output:
            delta = self._compute_delta(test.expected_output, test.actual_output)
            
            if delta.get("missing_keys"):
                context_parts.append(f"missing_fields: {', '.join(delta['missing_keys'])}")
            if delta.get("wrong_values"):
                context_parts.append(f"wrong_values: {', '.join(delta['wrong_values'])}")
            if delta.get("unexpected_keys"):
                context_parts.append(f"unexpected_fields: {', '.join(delta['unexpected_keys'])}")
        
        # Add category and tags
        if test.category:
            context_parts.append(f"category: {test.category}")
        if test.tags:
            context_parts.append(f"tags: {', '.join(test.tags)}")
        
        # Add system-specific context based on discovered schema
        if schema_info.get("system_type") == "chatbot":
            context_parts.extend(self._analyze_chatbot_context(test))
        elif schema_info.get("system_type") == "rag":
            context_parts.extend(self._analyze_rag_context(test))
        elif schema_info.get("system_type") == "code_generator":
            context_parts.extend(self._analyze_codegen_context(test))
        
        return " ".join(context_parts) if context_parts else "unknown_failure"

    def _compute_delta(self, expected: Any, actual: Any) -> Dict[str, Any]:
        """Compute difference between expected and actual output."""
        delta = {
            "missing_keys": set(),
            "wrong_values": set(),
            "unexpected_keys": set()
        }
        
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return delta
        
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        
        delta["missing_keys"] = expected_keys - actual_keys
        delta["unexpected_keys"] = actual_keys - expected_keys
        
        for key in expected_keys.intersection(actual_keys):
            if expected[key] != actual[key]:
                delta["wrong_values"].add(key)
        
        return delta

    def _analyze_chatbot_context(self, test: TestResult) -> List[str]:
        """Analyze chatbot-specific context."""
        context = []
        
        if isinstance(test.actual_output, dict):
            output = test.actual_output
            if "response" in output:
                response_text = str(output["response"]).lower()
                if any(word in response_text for word in ["sorry", "cannot", "unable"]):
                    context.append("apology_response")
                if any(word in response_text for word in ["hello", "hi", "greeting"]):
                    context.append("contains_greeting")
                if any(word in response_text for word in ["goodbye", "bye", "thanks"]):
                    context.append("contains_signoff")
        
        return context

    def _analyze_rag_context(self, test: TestResult) -> List[str]:
        """Analyze RAG-specific context."""
        context = []
        
        if isinstance(test.actual_output, dict):
            output = test.actual_output
            if "answer" in output and "citation" in output:
                if not output.get("citation"):
                    context.append("missing_citation")
            if "confidence" in output:
                conf = output["confidence"]
                if isinstance(conf, (int, float)) and conf < 0.7:
                    context.append("low_confidence")
        
        return context

    def _analyze_codegen_context(self, test: TestResult) -> List[str]:
        """Analyze code generation-specific context."""
        context = []
        
        if isinstance(test.actual_output, dict):
            output = test.actual_output
            if "code" in output:
                code = str(output["code"])
                if "def " in code and "return" not in code:
                    context.append("missing_return")
                if "import" not in code and "from " not in code:
                    context.append("missing_imports")
            if "runs_successfully" in output and not output["runs_successfully"]:
                context.append("code_doesnt_run")
        
        return context