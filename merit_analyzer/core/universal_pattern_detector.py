"""Universal pattern detector for any AI system."""

import json
import re
import numpy as np  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.cluster import DBSCAN  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ..models.test_result import TestResult
from ..models.pattern import Pattern
from ..analysis.claude_agent import MeritClaudeAgent


class UniversalPatternDetector:
    """Detects failure patterns in ANY AI system's test results."""

    def __init__(self, 
                 min_cluster_size: int = 2,
                 similarity_threshold: float = 0.3,
                 claude_agent: Optional[MeritClaudeAgent] = None):
        """
        Initialize universal pattern detector.
        
        Args:
            min_cluster_size: Minimum cluster size for patterns
            similarity_threshold: Similarity threshold for clustering
            claude_agent: Optional Claude agent for AI-powered analysis
        """
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.claude_agent = claude_agent

    def detect_patterns(self, test_results: List[TestResult]) -> Dict[str, List[TestResult]]:
        """
        Detect failure patterns using hierarchical clustering.
        
        Args:
            test_results: List of test results to analyze
            
        Returns:
            Dictionary mapping pattern names to lists of failing tests
        """
        failures = [t for t in test_results if t.status == "failed"]
        
        if not failures:
            return {}

        # For large datasets, sample before clustering to improve performance
        # Sample size: max 500 or 30% of failures (whichever is smaller)
        max_samples = 500
        if len(failures) > max_samples:
            # Use intelligent sampling to maintain diversity
            sampled_failures = self._sample_representative_tests(failures, max_samples)
        else:
            sampled_failures = failures

        # Step 1: Discover system schema and patterns
        schema_info = self._discover_system_schema(sampled_failures)
        
        # Step 2: Hierarchical clustering on sample
        sample_patterns = self._hierarchical_clustering(sampled_failures, schema_info)
        
        # Step 3: Assign all failures (not just sample) to discovered patterns
        if len(failures) > max_samples:
            patterns = self._assign_all_failures_to_patterns(failures, sample_patterns, schema_info)
        else:
            patterns = sample_patterns
        
        # Step 4: Generate intelligent pattern names (use fast heuristic naming)
        named_patterns = self._generate_pattern_names_fast(patterns, schema_info)
        
        return named_patterns

    def _discover_system_schema(self, failures: List[TestResult]) -> Dict[str, Any]:
        """
        Use LLM to discover the system's data schema and patterns.
        
        Args:
            failures: List of failed test results
            
        Returns:
            Schema information including field importance and patterns
        """
        if not self.claude_agent or len(failures) < 2:
            return self._fallback_schema_discovery(failures)

        # Sample representative tests for schema analysis
        # Use smart sampling to capture diversity while staying within token limits
        sample_tests = self._sample_representative_tests(failures, max_samples=10)
        
        prompt = f"""
Analyze these test results to understand the AI system's data schema and patterns.

TEST RESULTS (showing {len(sample_tests)} of {len(failures)} total failures):
{self._format_tests_for_schema_analysis(sample_tests)}

As an intelligent agent, analyze the test data to discover:

1. SYSTEM TYPE: What kind of AI system is this? 
   - Analyze the input/output patterns, not keywords
   - Options: chatbot, rag, agent, code_generator, image_analysis, custom, etc.
   - Be specific based on the actual test data structure

2. DATA SCHEMA: What are the key fields in input/output that matter for this system?

3. VALIDATION PATTERNS: What patterns do you see in expected vs actual outputs?

4. FAILURE TYPES: What types of failures are occurring? (missing fields, wrong values, format issues, etc.)

5. IMPORTANT FIELDS: Which fields are most critical for success/failure?

Provide your analysis in JSON format (NO hardcoded detection - analyze the actual data):
{{
    "system_type": "specific system type based on analysis",
    "key_fields": ["field1", "field2"],
    "failure_types": ["type1", "type2"],
    "validation_patterns": ["pattern1", "pattern2"]
}}
"""

        try:
            response = self.claude_agent._call_anthropic_direct(prompt)
            return self._parse_schema_response(response)
        except Exception as e:
            print(f"Warning: Schema discovery failed: {e}")
            return self._fallback_schema_discovery(failures)

    def _sample_representative_tests(self, failures: List[TestResult], max_samples: int = 10) -> List[TestResult]:
        """
        Sample representative tests to capture diversity.
        
        Strategy:
        1. If few tests, use all
        2. Otherwise, sample to ensure:
           - Different categories are represented
           - Different input types/structures
           - Different failure types
        """
        if len(failures) <= max_samples:
            return failures
        
        # Group by category for diversity
        by_category = defaultdict(list)
        for test in failures:
            category = test.category or "uncategorized"
            by_category[category].append(test)
        
        # Sample from each category proportionally
        samples = []
        categories = list(by_category.keys())
        
        # Distribute samples across categories
        samples_per_category = max(1, max_samples // len(categories))
        remaining = max_samples
        
        for category in categories:
            category_tests = by_category[category]
            if remaining <= 0:
                break
            
            # Take samples from this category
            num_from_category = min(samples_per_category, len(category_tests), remaining)
            
            # Sample evenly spaced tests from this category
            if len(category_tests) <= num_from_category:
                samples.extend(category_tests)
            else:
                # Take evenly spaced samples
                step = len(category_tests) / num_from_category
                indices = [int(i * step) for i in range(num_from_category)]
                samples.extend([category_tests[i] for i in indices])
            
            remaining -= num_from_category
        
        # Ensure we don't exceed max_samples
        return samples[:max_samples]

    def _fallback_schema_discovery(self, failures: List[TestResult]) -> Dict[str, Any]:
        """Fallback schema discovery using heuristics."""
        schema_info = {
            "system_type": "unknown",
            "key_fields": [],
            "failure_types": [],
            "validation_patterns": []
        }
        
        # Analyze input/output structure
        input_fields = set()
        output_fields = set()
        
        for test in failures:
            if isinstance(test.input, dict):
                input_fields.update(test.input.keys())
            if isinstance(test.actual_output, dict):
                output_fields.update(test.actual_output.keys())
            if isinstance(test.expected_output, dict):
                output_fields.update(test.expected_output.keys())
        
        schema_info["key_fields"] = list(input_fields.union(output_fields))
        
        return schema_info

    def _assign_all_failures_to_patterns(self, 
                                       all_failures: List[TestResult],
                                       sample_patterns: Dict[str, List[TestResult]],
                                       schema_info: Dict[str, Any]) -> Dict[str, List[TestResult]]:
        """
        Assign all failures to patterns discovered from sample.
        
        This allows us to cluster on a sample but still map all failures to patterns.
        """
        # Extract pattern centroids from sample
        pattern_centroids = {}
        for pattern_name, sample_tests in sample_patterns.items():
            if sample_tests:
                # Use first test as representative
                pattern_centroids[pattern_name] = sample_tests[0]
        
        # Assign each failure to closest pattern
        assigned_patterns = defaultdict(list)
        
        # Get already-assigned failures from sample
        sample_failures = set()
        for tests in sample_patterns.values():
            sample_failures.update(t.test_id for t in tests)
        
        # Assign remaining failures
        for failure in all_failures:
            if failure.test_id in sample_failures:
                # Already assigned in sample patterns
                continue
            
            # Find closest pattern by input similarity
            best_pattern = None
            best_similarity = -1
            
            failure_input = self._extract_input_features(failure).lower()
            
            for pattern_name, centroid_test in pattern_centroids.items():
                centroid_input = self._extract_input_features(centroid_test).lower()
                
                # Simple similarity check
                failure_words = set(failure_input.split())
                centroid_words = set(centroid_input.split())
                
                if failure_words and centroid_words:
                    similarity = len(failure_words & centroid_words) / len(failure_words | centroid_words)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_pattern = pattern_name
            
            # Assign to best matching pattern if similarity > threshold, else create new
            if best_pattern and best_similarity > self.similarity_threshold:
                assigned_patterns[best_pattern].append(failure)
            else:
                # Create new pattern for this failure
                assigned_patterns[f"single_failure_{failure.test_id[:8]}"].append(failure)
        
        # Merge with sample patterns
        result = dict(sample_patterns)
        for pattern_name, failures in assigned_patterns.items():
            if pattern_name in result:
                result[pattern_name].extend(failures)
            else:
                result[pattern_name] = failures
        
        return result

    def _format_tests_for_schema_analysis(self, tests: List[TestResult]) -> str:
        """Format tests for schema analysis."""
        formatted = []
        for i, test in enumerate(tests, 1):
            formatted.append(f"""
Test {i} (ID: {test.test_id}):
  Input: {json.dumps(test.input, indent=2)}
  Expected: {json.dumps(test.expected_output, indent=2) if test.expected_output else 'None'}
  Actual: {json.dumps(test.actual_output, indent=2)}
  Status: {test.status}
""")
        return "\n".join(formatted)

    def _parse_schema_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response for schema information intelligently.
        
        NO HARDCODED DETECTION - Let the LLM provide structured output.
        """
        schema_info = {
            "system_type": "unknown",
            "key_fields": [],
            "failure_types": [],
            "validation_patterns": []
        }
        
        try:
            # Try to extract JSON from LLM response (agent should return structured data)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                schema_info["system_type"] = parsed.get("system_type", "unknown")
                schema_info["key_fields"] = parsed.get("key_fields", [])
                schema_info["failure_types"] = parsed.get("failure_types", [])
                schema_info["validation_patterns"] = parsed.get("validation_patterns", [])
            else:
                # Fallback: ask LLM to provide system_type explicitly
                # Look for explicit system_type declaration in response
                system_type_match = re.search(r'system[_\s]*type[:\s]+([a-z_]+)', response.lower())
                if system_type_match:
                    schema_info["system_type"] = system_type_match.group(1)
                else:
                    # No hardcoded detection - use what LLM says
                    schema_info["system_type"] = "unknown"
        except Exception as e:
            # If parsing fails, don't default to hardcoded values
            # Keep as unknown and let downstream LLM analysis figure it out
            schema_info["system_type"] = "unknown"
        
        return schema_info

    def _hierarchical_clustering(self, failures: List[TestResult], schema_info: Dict[str, Any]) -> Dict[str, List[TestResult]]:
        """
        Perform hierarchical clustering on failures.
        
        Strategy:
        1. Cluster by input similarity
        2. Within each input cluster, sub-cluster by output patterns
        3. Within output clusters, group by delta types
        """
        if len(failures) < self.min_cluster_size:
            return {"single_failure": failures}

        # Step 1: Input similarity clustering
        input_clusters = self._cluster_by_input_similarity(failures)
        
        # Step 2: Output pattern clustering within input clusters
        output_clusters = self._cluster_by_output_patterns(input_clusters)
        
        # Step 3: Delta analysis clustering
        delta_clusters = self._cluster_by_deltas(output_clusters)
        
        return delta_clusters

    def _cluster_by_input_similarity(self, failures: List[TestResult]) -> Dict[str, List[TestResult]]:
        """Cluster failures by input similarity."""
        if len(failures) < 2:
            return {"input_cluster_0": failures}

        # Convert inputs to text for similarity analysis
        input_texts = []
        for test in failures:
            input_text = self._extract_input_features(test)
            input_texts.append(input_text)

        # Use TF-IDF for text similarity
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(input_texts)
            
            # Use DBSCAN for clustering
            clustering = DBSCAN(
                eps=self.similarity_threshold,
                min_samples=self.min_cluster_size,
                metric='cosine'
            ).fit(tfidf_matrix)
            
            # Group by cluster labels
            clusters = defaultdict(list)
            for i, label in enumerate(clustering.labels_):
                cluster_name = f"input_cluster_{label}" if label != -1 else f"input_noise_{i}"
                clusters[cluster_name].append(failures[i])
            
            return dict(clusters)
            
        except Exception as e:
            print(f"Warning: Input clustering failed: {e}")
            return {"input_cluster_0": failures}

    def _cluster_by_output_patterns(self, input_clusters: Dict[str, List[TestResult]]) -> Dict[str, List[TestResult]]:
        """Cluster by output patterns within input clusters."""
        output_clusters = {}
        
        for cluster_name, tests in input_clusters.items():
            if len(tests) < 2:
                output_clusters[f"{cluster_name}_output"] = tests
                continue

            # Analyze output patterns
            output_patterns = self._analyze_output_patterns(tests)
            
            # Group by similar patterns
            pattern_groups = defaultdict(list)
            for i, test in enumerate(tests):
                pattern_key = self._get_output_pattern_key(test, output_patterns)
                pattern_groups[pattern_key].append(test)
            
            # Create new cluster names
            for j, (pattern_key, group_tests) in enumerate(pattern_groups.items()):
                new_name = f"{cluster_name}_output_{j}"
                output_clusters[new_name] = group_tests
        
        return output_clusters

    def _cluster_by_deltas(self, output_clusters: Dict[str, List[TestResult]]) -> Dict[str, List[TestResult]]:
        """Cluster by expected vs actual deltas."""
        delta_clusters = {}
        
        for cluster_name, tests in output_clusters.items():
            if len(tests) < 2:
                delta_clusters[f"{cluster_name}_delta"] = tests
                continue

            # Analyze deltas
            delta_groups = defaultdict(list)
            for test in tests:
                delta_type = self._classify_delta(test)
                delta_groups[delta_type].append(test)
            
            # Create final cluster names
            for j, (delta_type, group_tests) in enumerate(delta_groups.items()):
                new_name = f"{cluster_name}_delta_{j}"
                delta_clusters[new_name] = group_tests
        
        return delta_clusters

    def _extract_input_features(self, test: TestResult) -> str:
        """Extract features from test input for clustering."""
        features = []
        
        if isinstance(test.input, dict):
            # Extract key-value pairs
            for key, value in test.input.items():
                if isinstance(value, (str, int, float, bool)):
                    features.append(f"{key}:{value}")
                else:
                    features.append(f"{key}:{str(value)[:100]}")
        else:
            features.append(str(test.input)[:200])
        
        # Add test metadata
        if test.category:
            features.append(f"category:{test.category}")
        if test.tags:
            features.extend([f"tag:{tag}" for tag in test.tags])
        
        return " ".join(features)

    def _analyze_output_patterns(self, tests: List[TestResult]) -> Dict[str, Any]:
        """Analyze common patterns in test outputs."""
        patterns = {
            "missing_fields": set(),
            "wrong_values": set(),
            "format_issues": set(),
            "length_issues": set()
        }
        
        for test in tests:
            if test.expected_output and test.actual_output:
                delta = self._compute_delta(test.expected_output, test.actual_output)
                patterns["missing_fields"].update(delta.get("missing_keys", []))
                patterns["wrong_values"].update(delta.get("wrong_values", []))
                patterns["format_issues"].update(delta.get("format_issues", []))
        
        return patterns

    def _get_output_pattern_key(self, test: TestResult, patterns: Dict[str, Any]) -> str:
        """Get a key representing the output pattern for this test."""
        if not test.expected_output or not test.actual_output:
            return "no_expected_output"
        
        delta = self._compute_delta(test.expected_output, test.actual_output)
        
        # Create pattern key based on delta
        key_parts = []
        if delta.get("missing_keys"):
            key_parts.append(f"missing_{len(delta['missing_keys'])}")
        if delta.get("wrong_values"):
            key_parts.append(f"wrong_{len(delta['wrong_values'])}")
        if delta.get("format_issues"):
            key_parts.append("format_issue")
        
        return "_".join(key_parts) if key_parts else "exact_match"

    def _classify_delta(self, test: TestResult) -> str:
        """Classify the type of delta between expected and actual output."""
        if not test.expected_output or not test.actual_output:
            return "no_expected_output"
        
        delta = self._compute_delta(test.expected_output, test.actual_output)
        
        if delta.get("missing_keys"):
            return "missing_fields"
        elif delta.get("wrong_values"):
            return "wrong_values"
        elif delta.get("format_issues"):
            return "format_issues"
        elif delta.get("length_issues"):
            return "length_issues"
        else:
            return "unknown_delta"

    def _compute_delta(self, expected: Any, actual: Any) -> Dict[str, Any]:
        """Compute difference between expected and actual output."""
        delta = {
            "missing_keys": set(),
            "wrong_values": set(),
            "format_issues": set(),
            "length_issues": set()
        }
        
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            # Handle non-dict outputs
            if str(expected) != str(actual):
                delta["wrong_values"].add("value_mismatch")
            return delta
        
        # Check for missing keys
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        delta["missing_keys"] = expected_keys - actual_keys
        delta["unexpected_keys"] = actual_keys - expected_keys
        
        # Check for wrong values
        for key in expected_keys.intersection(actual_keys):
            if expected[key] != actual[key]:
                delta["wrong_values"].add(key)
        
        # Check for format issues
        if isinstance(expected, str) and isinstance(actual, str):
            if len(expected) > 0 and len(actual) > 0:
                if abs(len(expected) - len(actual)) / len(expected) > 0.5:
                    delta["length_issues"].add("significant_length_difference")
        
        return delta

    def _generate_pattern_names_fast(self, patterns: Dict[str, List[TestResult]], schema_info: Dict[str, Any]) -> Dict[str, List[TestResult]]:
        """Generate pattern names using fast heuristic approach (no AI calls)."""
        named_patterns = {}
        
        for pattern_name, tests in patterns.items():
            # Use fast heuristic naming
            heuristic_name = self._generate_heuristic_pattern_name(tests, pattern_name)
            named_patterns[heuristic_name] = tests
        
        return named_patterns
    
    def _generate_pattern_names(self, patterns: Dict[str, List[TestResult]], schema_info: Dict[str, Any]) -> Dict[str, List[TestResult]]:
        """Generate intelligent pattern names using LLM or heuristics."""
        named_patterns = {}
        
        for idx, (pattern_name, tests) in enumerate(patterns.items(), 1):
            # Always try AI naming if available (works for single tests too)
            if self.claude_agent:
                try:
                    intelligent_name = self._generate_ai_pattern_name(tests, schema_info)
                    named_patterns[intelligent_name] = tests
                except Exception as e:
                    # Fallback to heuristic if AI fails
                    heuristic_name = self._generate_heuristic_pattern_name(tests, pattern_name)
                    named_patterns[heuristic_name] = tests
            else:
                # Use heuristic naming
                heuristic_name = self._generate_heuristic_pattern_name(tests, pattern_name)
                named_patterns[heuristic_name] = tests
        
        return named_patterns

    def _generate_ai_pattern_name(self, tests: List[TestResult], schema_info: Dict[str, Any]) -> str:
        """Generate pattern name using AI analysis."""
        if not self.claude_agent:
            return self._generate_heuristic_pattern_name(tests, "ai_pattern")
        
        # Sample a few tests for analysis
        sample_tests = tests[:min(3, len(tests))]
        
        prompt = f"""
Analyze these failing tests to generate a descriptive pattern name.

SYSTEM TYPE: {schema_info.get('system_type', 'Unknown')}

FAILING TESTS ({len(tests)} total):
{self._format_tests_for_pattern_analysis(sample_tests)}

Generate a concise, descriptive pattern name that captures the common failure theme.
The name should be:
- Specific and actionable
- Not generic (avoid "uncategorized", "general", etc.)
- Focus on the root cause or missing element
- Use snake_case format

Examples of good pattern names:
- missing_greeting_signature
- pricing_information_incomplete
- code_validation_failed
- response_format_incorrect

Pattern name:
"""

        try:
            response = self.claude_agent._call_anthropic_direct(prompt, max_tokens=200)
            # Extract the pattern name from response
            lines = response.strip().split('\n')
            
            # Remove code blocks first
            response_cleaned = response
            if '```' in response:
                # Remove everything between triple backticks
                import re
                response_cleaned = re.sub(r'```[^`]*```', '', response, flags=re.DOTALL)
                lines = response_cleaned.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                # Skip empty lines, comments, markdown headers, and instruction text
                if (not line or 
                    line.startswith('#') or 
                    line.startswith('*') or 
                    line.startswith('##') or
                    line.lower().startswith('pattern name:') and ':' not in line[12:] or
                    'based on' in line.lower() or
                    'failing test' in line.lower() and 'analysis' in line.lower() or
                    'this captures' in line.lower() or
                    'the root cause' in line.lower() and 'where' in line.lower()):
                    continue
                    
                # Clean up the response
                name = line.replace('Pattern name:', '').replace('Pattern:', '').strip()
                # Remove list markers (dashes, numbers, bullets)
                name = name.lstrip('-. •*0123456789)').strip()
                # Remove any leading/trailing punctuation
                name = name.strip('.,;:!?').strip()
                
                # Skip if it's clearly instruction text or empty
                if (not name or
                    name.startswith('Based on') or
                    name.startswith('This captures') or
                    name.startswith('-') or
                    name == '```' or
                    '```' in name or
                    len(name) < 3):
                    continue
                
                # Validate and return (increase limit to 60 for descriptive names)
                if len(name) <= 60:
                    return name
            
            # Fallback if parsing fails
            return self._generate_heuristic_pattern_name(tests, "ai_generated")
            
        except Exception as e:
            print(f"Warning: AI pattern naming failed: {e}")
            return self._generate_heuristic_pattern_name(tests, "ai_fallback")

    def _generate_heuristic_pattern_name(self, tests: List[TestResult], base_name: str) -> str:
        """Generate pattern name using heuristics."""
        if len(tests) == 1:
            test = tests[0]
            # Try to use test_name if available
            if test.test_name:
                # Clean and format test name
                name = test.test_name.lower().replace(' ', '_').replace('-', '_')
                # Remove common prefixes/suffixes
                for prefix in ['test_', 'test', 'testcase_', 'testcase', 'should_']:
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                # If still too long, truncate but try to keep it meaningful (cut at word boundary if possible)
                if len(name) > 40:
                    # Try to cut at underscore
                    if '_' in name:
                        parts = name.split('_')
                        result = []
                        for part in parts:
                            if len('_'.join(result + [part])) <= 40:
                                result.append(part)
                            else:
                                break
                        name = '_'.join(result) if result else name[:40]
                    else:
                        name = name[:40]
                return name
            
            # Try to extract from failure_reason
            if test.failure_reason:
                # Extract key words from failure reason
                words = test.failure_reason.lower().split()
                # Filter out common words
                stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 
                             'to', 'for', 'of', 'in', 'on', 'at', 'by', 'with', 'from', 'failed', 
                             'error', 'exception', 'test', 'tests', 'testcase'}
                keywords = [w for w in words if w not in stop_words and len(w) > 3][:3]
                if keywords:
                    return '_'.join(keywords)[:30]
            
            # Fallback to test_id (truncated)
            return test.test_id.replace('-', '_').replace(' ', '_')[:20]
        
        # Analyze common elements
        common_elements = []
        
        # Check for common input patterns
        if all(isinstance(t.input, dict) for t in tests):
            input_keys = set()
            for test in tests:
                input_keys.update(test.input.keys())
            if len(input_keys) == 1:
                common_elements.append(f"input_{list(input_keys)[0]}")
        
        # Check for common output issues
        missing_fields = set()
        for test in tests:
            if test.expected_output and test.actual_output:
                delta = self._compute_delta(test.expected_output, test.actual_output)
                missing_fields.update(delta.get("missing_keys", []))
        
        if missing_fields:
            common_elements.append(f"missing_{list(missing_fields)[0]}")
        
        # Check for common categories
        categories = [t.category for t in tests if t.category]
        if categories and len(set(categories)) == 1:
            common_elements.append(categories[0])
        
        if common_elements:
            return "_".join(common_elements[:2])  # Max 2 elements
        else:
            return f"pattern_{base_name}"

    def _format_tests_for_pattern_analysis(self, tests: List[TestResult]) -> str:
        """Format tests for pattern analysis."""
        formatted = []
        for i, test in enumerate(tests, 1):
            formatted.append(f"""
Test {i}:
  Input: {json.dumps(test.input, indent=2)}
  Expected: {json.dumps(test.expected_output, indent=2) if test.expected_output else 'None'}
  Actual: {json.dumps(test.actual_output, indent=2)}
""")
        return "\n".join(formatted)