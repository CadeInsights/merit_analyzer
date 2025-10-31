"""Root cause analysis for test failures."""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict

from ..models.test_result import TestResult
from ..models.pattern import Pattern


class RootCauseAnalyzer:
    """Analyze root causes of test failure patterns."""

    def __init__(self):
        """Initialize root cause analyzer."""
        self.common_causes = {
            "validation_error": "Input validation is too strict or missing",
            "prompt_issue": "Prompt template has issues or is unclear",
            "model_config": "Model configuration is incorrect",
            "api_error": "External API call is failing",
            "timeout": "Request is timing out",
            "data_format": "Data format is incorrect",
            "logic_error": "Business logic has a bug",
            "edge_case": "Edge case is not handled",
            "permission": "Permission or authentication issue",
            "resource_limit": "Resource limit exceeded",
        }

    def analyze_root_cause(self, 
                          pattern: Pattern,
                          code_context: Dict[str, str],
                          architecture: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze root cause of a failure pattern.

        Args:
            pattern: The failure pattern to analyze
            code_context: Relevant code snippets
            architecture: System architecture information

        Returns:
            Dictionary with root cause analysis
        """
        # Verbose output suppressed - progress shown via Rich in main analyzer
        
        # Analyze test failures
        failure_analysis = self._analyze_failures(pattern.test_results)
        # Add pattern name to failure analysis for root cause detection
        failure_analysis["pattern_name"] = pattern.name.lower()
        
        # Analyze code context
        code_analysis = self._analyze_code_context(code_context)
        
        # Analyze architecture
        arch_analysis = self._analyze_architecture(architecture, pattern)
        
        # Determine most likely root cause
        root_cause = self._determine_root_cause(failure_analysis, code_analysis, arch_analysis)
        
        # Generate confidence score
        confidence = self._calculate_confidence(failure_analysis, code_analysis, arch_analysis)
        
        return {
            "root_cause": root_cause,
            "confidence": confidence,
            "evidence": {
                "failure_analysis": failure_analysis,
                "code_analysis": code_analysis,
                "architecture_analysis": arch_analysis,
            },
            "recommended_fixes": self._suggest_fixes(root_cause, pattern),
        }

    def _analyze_failures(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Analyze failure patterns in test results."""
        analysis = {
            "common_failure_reasons": [],
            "input_patterns": [],
            "output_patterns": [],
            "timing_issues": False,
            "error_types": [],
        }
        
        if not test_results:
            return analysis
        
        # Analyze failure reasons
        failure_reasons = [t.failure_reason for t in test_results if t.failure_reason]
        if failure_reasons:
            reason_counts = Counter(failure_reasons)
            analysis["common_failure_reasons"] = [
                {"reason": reason, "count": count, "percentage": count / len(failure_reasons) * 100}
                for reason, count in reason_counts.most_common(5)
            ]
        
        # Analyze input patterns
        inputs = [t.input for t in test_results if t.input]
        if inputs:
            analysis["input_patterns"] = self._extract_input_patterns(inputs)
        
        # Analyze output patterns
        outputs = [t.actual_output for t in test_results if t.actual_output]
        if outputs:
            analysis["output_patterns"] = self._extract_output_patterns(outputs)
        
        # Check for timing issues
        execution_times = [t.execution_time_ms for t in test_results if t.execution_time_ms is not None]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            analysis["timing_issues"] = avg_time > 10000  # More than 10 seconds
        
        # Analyze error types
        analysis["error_types"] = self._classify_error_types(failure_reasons)
        
        return analysis

    def _extract_input_patterns(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """Extract patterns from test inputs."""
        patterns = []
        
        # Length patterns
        lengths = [len(inp) for inp in inputs]
        if lengths:
            patterns.append({
                "type": "length",
                "min": min(lengths),
                "max": max(lengths),
                "avg": sum(lengths) / len(lengths),
            })
        
        # Common words
        all_words = []
        for inp in inputs:
            # Convert to string if it's a dict
            if isinstance(inp, dict):
                inp = json.dumps(inp, sort_keys=True)
            elif not isinstance(inp, str):
                inp = str(inp)
            
            words = re.findall(r'\b[a-z]{3,}\b', inp.lower())
            all_words.extend(words)
        
        if all_words:
            word_counts = Counter(all_words)
            patterns.append({
                "type": "common_words",
                "words": [{"word": word, "count": count} for word, count in word_counts.most_common(10)]
            })
        
        # Special characters
        special_chars = set()
        for inp in inputs:
            special_chars.update(re.findall(r'[^\w\s]', inp))
        
        if special_chars:
            patterns.append({
                "type": "special_characters",
                "chars": list(special_chars)
            })
        
        return patterns

    def _extract_output_patterns(self, outputs: List[str]) -> List[Dict[str, Any]]:
        """Extract patterns from test outputs."""
        patterns = []
        
        # Error message patterns
        error_patterns = [
            r'error|exception|failed|invalid|unexpected',
            r'timeout|timed out|time out',
            r'permission|unauthorized|forbidden',
            r'not found|missing|undefined',
            r'validation|invalid input|bad request',
        ]
        
        for pattern in error_patterns:
            matches = sum(1 for output in outputs if re.search(pattern, output.lower()))
            if matches > 0:
                patterns.append({
                    "type": "error_pattern",
                    "pattern": pattern,
                    "matches": matches,
                    "percentage": matches / len(outputs) * 100
                })
        
        # Length patterns
        lengths = [len(output) for output in outputs]
        if lengths:
            patterns.append({
                "type": "length",
                "min": min(lengths),
                "max": max(lengths),
                "avg": sum(lengths) / len(lengths),
            })
        
        return patterns

    def _classify_error_types(self, failure_reasons: List[str]) -> List[str]:
        """Classify error types from failure reasons."""
        error_types = []
        
        for reason in failure_reasons:
            reason_lower = reason.lower()
            
            if any(word in reason_lower for word in ['timeout', 'timed out', 'time out']):
                error_types.append('timeout')
            elif any(word in reason_lower for word in ['permission', 'unauthorized', 'forbidden']):
                error_types.append('permission')
            elif any(word in reason_lower for word in ['validation', 'invalid', 'bad request']):
                error_types.append('validation')
            elif any(word in reason_lower for word in ['not found', 'missing', 'undefined']):
                error_types.append('not_found')
            elif any(word in reason_lower for word in ['error', 'exception', 'failed']):
                error_types.append('general_error')
            else:
                error_types.append('unknown')
        
        return list(set(error_types))

    def _analyze_code_context(self, code_context: Dict[str, str]) -> Dict[str, Any]:
        """Analyze code context for potential issues."""
        analysis = {
            "potential_issues": [],
            "code_quality": {},
            "error_handling": {},
        }
        
        for file_path, code in code_context.items():
            file_analysis = self._analyze_single_file(file_path, code)
            analysis["potential_issues"].extend(file_analysis["issues"])
            analysis["code_quality"][file_path] = file_analysis["quality"]
            analysis["error_handling"][file_path] = file_analysis["error_handling"]
        
        return analysis

    def _analyze_single_file(self, file_path: str, code: str) -> Dict[str, Any]:
        """Analyze a single file for issues."""
        issues = []
        quality = {"score": 0, "issues": []}
        error_handling = {"has_try_catch": False, "has_validation": False, "has_logging": False}
        
        lines = code.split('\n')
        
        # Check for common issues
        for i, line in enumerate(lines, 1):
            line_lower = line.lower().strip()
            
            # Check for hardcoded values
            if any(word in line_lower for word in ['localhost', '127.0.0.1', 'api_key', 'password']):
                issues.append({
                    "type": "hardcoded_value",
                    "file": file_path,
                    "line": i,
                    "description": "Hardcoded value found",
                    "severity": "medium"
                })
            
            # Check for missing error handling
            if 'api' in line_lower and 'requests' in line_lower and 'try' not in code[:code.find(line)]:
                issues.append({
                    "type": "missing_error_handling",
                    "file": file_path,
                    "line": i,
                    "description": "API call without error handling",
                    "severity": "high"
                })
            
            # Check for validation issues
            if 'input' in line_lower and 'validate' not in line_lower and 'check' not in line_lower:
                issues.append({
                    "type": "missing_validation",
                    "file": file_path,
                    "line": i,
                    "description": "Input without validation",
                    "severity": "medium"
                })
        
        # Check for error handling patterns
        error_handling["has_try_catch"] = 'try:' in code
        error_handling["has_validation"] = any(word in code.lower() for word in ['validate', 'check', 'assert'])
        error_handling["has_logging"] = any(word in code.lower() for word in ['log', 'logger', 'print'])
        
        # Calculate quality score
        quality_score = 100
        if not error_handling["has_try_catch"]:
            quality_score -= 20
        if not error_handling["has_validation"]:
            quality_score -= 15
        if not error_handling["has_logging"]:
            quality_score -= 10
        
        quality["score"] = max(quality_score, 0)
        
        return {
            "issues": issues,
            "quality": quality,
            "error_handling": error_handling,
        }

    def _analyze_architecture(self, architecture: Dict[str, Any], pattern: Pattern) -> Dict[str, Any]:
        """Analyze architecture for potential issues."""
        analysis = {
            "component_issues": [],
            "integration_issues": [],
            "configuration_issues": [],
        }
        
        # Check for missing components
        if not architecture.get("agents"):
            analysis["component_issues"].append({
                "type": "missing_agents",
                "description": "No AI agents identified in architecture",
                "severity": "high"
            })
        
        # Check for prompt issues
        prompts = architecture.get("prompts", [])
        if not prompts:
            analysis["component_issues"].append({
                "type": "missing_prompts",
                "description": "No prompt templates identified",
                "severity": "medium"
            })
        
        # Check for configuration issues
        config = architecture.get("configuration", {})
        if not config.get("api_keys"):
            analysis["configuration_issues"].append({
                "type": "missing_api_keys",
                "description": "No API keys configured",
                "severity": "high"
            })
        
        return analysis

    def _determine_root_cause(self, 
                            failure_analysis: Dict[str, Any],
                            code_analysis: Dict[str, Any],
                            arch_analysis: Dict[str, Any]) -> str:
        """Determine the most likely root cause."""
        # Score different potential causes
        cause_scores = defaultdict(float)
        
        # Analyze pattern name for hints (often very descriptive)
        pattern_name = failure_analysis.get("pattern_name", "").lower()
        
        if pattern_name:
            # Pattern names often contain the root cause
            if any(term in pattern_name for term in ["hallucinat", "hallucin"]):
                cause_scores["hallucination"] += 1.0  # High confidence from pattern name
            if any(term in pattern_name for term in ["missing", "no data", "empty", "insufficient"]):
                cause_scores["missing_data"] += 1.0
            if any(term in pattern_name for term in ["truncat", "cut", "incomplete"]):
                cause_scores["content_truncation"] += 1.0
            if any(term in pattern_name for term in ["search", "query", "retrieval", "rag"]):
                cause_scores["search_failure"] += 0.9
            if any(term in pattern_name for term in ["tone", "format", "style", "quality"]):
                cause_scores["output_format"] += 0.8
        
        # First, analyze actual failure reasons from tests (more specific than generic error types)
        failure_reasons = failure_analysis.get("common_failure_reasons", [])
        if failure_reasons:
            # Look for RAG-specific failure patterns
            reasons_text = " ".join([r.get("reason", "").lower() for r in failure_reasons[:3]])
            
            if any(term in reasons_text for term in ["missing", "not found", "no data", "empty result"]):
                cause_scores["missing_data"] += 0.9
            if any(term in reasons_text for term in ["hallucinat", "incorrect", "wrong", "fabricat"]):
                cause_scores["hallucination"] += 0.9
            if any(term in reasons_text for term in ["truncat", "cut off", "incomplete"]):
                cause_scores["content_truncation"] += 0.8
            if any(term in reasons_text for term in ["search", "query", "retrieval"]):
                cause_scores["search_failure"] += 0.8
            if any(term in reasons_text for term in ["tone", "format", "style"]):
                cause_scores["output_format"] += 0.7
        
        # Analyze output patterns for RAG-specific issues
        output_patterns = failure_analysis.get("output_patterns", [])
        for pattern in output_patterns:
            if "missing_fields" in str(pattern).lower():
                cause_scores["missing_data"] += 0.5
            if "wrong_values" in str(pattern).lower():
                cause_scores["hallucination"] += 0.5
        
        # Analyze failure patterns (generic error types)
        error_types = failure_analysis.get("error_types", [])
        if "timeout" in error_types:
            cause_scores["timeout"] += 0.8
        if "permission" in error_types:
            cause_scores["permission"] += 0.8
        if "validation" in error_types:
            cause_scores["validation_error"] += 0.4  # Lower weight - too generic
        if "not_found" in error_types:
            cause_scores["missing_data"] += 0.6  # More specific than data_format
        
        # Analyze code issues
        for issue in code_analysis.get("potential_issues", []):
            if issue["type"] == "missing_error_handling":
                cause_scores["api_error"] += 0.6
            elif issue["type"] == "missing_validation":
                # Only add validation_error if we don't have more specific causes
                if not any(c in cause_scores for c in ["missing_data", "hallucination", "search_failure"]):
                    cause_scores["validation_error"] += 0.3  # Lower weight
            elif issue["type"] == "hardcoded_value":
                cause_scores["model_config"] += 0.3
        
        # Analyze architecture issues
        for issue in arch_analysis.get("component_issues", []):
            if issue["type"] == "missing_agents":
                cause_scores["logic_error"] += 0.7
            elif issue["type"] == "missing_prompts":
                cause_scores["prompt_issue"] += 0.6
        
        # Return the cause with highest score, preferring specific causes over generic ones
        if cause_scores:
            # Separate specific causes from generic ones
            specific_causes = ["missing_data", "hallucination", "content_truncation", "search_failure", 
                             "output_format", "prompt_issue", "timeout", "permission", "api_error"]
            generic_causes = ["validation_error", "model_config", "logic_error", "data_format", "edge_case"]
            
            # Filter out generic causes if we have specific ones
            specific_scores = {c: s for c, s in cause_scores.items() if c in specific_causes}
            if specific_scores:
                # Return the best specific cause
                return max(specific_scores.items(), key=lambda x: x[1])[0]
            
            # Only use generic causes if no specific ones found
            best_cause, best_score = max(cause_scores.items(), key=lambda x: x[1])
            return best_cause
        else:
            return "unknown"

    def _calculate_confidence(self,
                            failure_analysis: Dict[str, Any],
                            code_analysis: Dict[str, Any],
                            arch_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for root cause analysis."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on evidence
        if failure_analysis.get("common_failure_reasons"):
            confidence += 0.2
        
        if code_analysis.get("potential_issues"):
            confidence += 0.2
        
        if arch_analysis.get("component_issues"):
            confidence += 0.1
        
        return min(confidence, 1.0)

    def _suggest_fixes(self, root_cause: str, pattern: Pattern) -> List[str]:
        """Suggest fixes based on root cause."""
        fixes = []
        
        if root_cause == "validation_error":
            fixes.extend([
                "Add input validation for edge cases",
                "Review validation logic for false positives",
                "Add better error messages for validation failures"
            ])
        elif root_cause == "prompt_issue":
            fixes.extend([
                "Review and improve prompt templates",
                "Add examples to prompts",
                "Test prompts with different inputs"
            ])
        elif root_cause == "model_config":
            fixes.extend([
                "Check model configuration settings",
                "Verify API endpoints and parameters",
                "Review model selection logic"
            ])
        elif root_cause == "api_error":
            fixes.extend([
                "Add proper error handling for API calls",
                "Implement retry logic",
                "Add timeout handling"
            ])
        elif root_cause == "timeout":
            fixes.extend([
                "Increase timeout values",
                "Optimize query performance",
                "Add timeout handling"
            ])
        elif root_cause == "logic_error":
            fixes.extend([
                "Review business logic implementation",
                "Add unit tests for edge cases",
                "Debug step by step execution"
            ])
        else:
            fixes.append("Investigate the specific error patterns")
        
        return fixes
