"""Pattern merging for scalability."""

from typing import List, Dict, Any
from ..models.test_result import TestResult
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import numpy as np  # type: ignore


class PatternMerger:
    """Merge similar patterns to reduce analysis overhead."""

    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize pattern merger.
        
        Args:
            similarity_threshold: Threshold for merging similar patterns (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold

    def merge_similar_patterns(self, patterns: Dict[str, List[TestResult]]) -> Dict[str, List[TestResult]]:
        """
        Merge similar patterns to reduce redundancy.
        
        Args:
            patterns: Dictionary mapping pattern names to test lists
            
        Returns:
            Merged patterns dictionary
        """
        if len(patterns) <= 1:
            return patterns

        # Extract pattern features for similarity comparison
        pattern_features = self._extract_pattern_features(patterns)
        
        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(pattern_features)
        
        # Merge similar patterns
        merged_patterns = self._merge_by_similarity(patterns, similarity_matrix)
        
        return merged_patterns

    def _extract_pattern_features(self, patterns: Dict[str, List[TestResult]]) -> Dict[str, str]:
        """Extract features from patterns for similarity comparison."""
        features = {}
        
        for pattern_name, tests in patterns.items():
            # Extract key features from pattern
            feature_parts = []
            
            # Add pattern name
            feature_parts.append(pattern_name.lower())
            
            # Add test input features
            input_features = []
            for test in tests:
                input_text = self._extract_input_text(test)
                input_features.append(input_text.lower()[:100])  # Truncate for efficiency
            feature_parts.append(" ".join(input_features[:3]))  # Use first 3 tests
            
            # Add failure characteristics
            if tests:
                first_test = tests[0]
                if first_test.expected_output and first_test.actual_output:
                    delta_type = self._classify_delta_type(first_test)
                    feature_parts.append(delta_type)
            
            # Add category info
            categories = set(t.category for t in tests if t.category)
            if categories:
                feature_parts.append(" ".join(categories))
            
            features[pattern_name] = " ".join(feature_parts)
        
        return features

    def _extract_input_text(self, test: TestResult) -> str:
        """Extract text representation of test input."""
        if isinstance(test.input, str):
            return test.input
        elif isinstance(test.input, dict):
            # Convert dict to string (key-value pairs)
            parts = [f"{k}:{str(v)[:50]}" for k, v in test.input.items()]
            return " ".join(parts)
        else:
            return str(test.input)

    def _classify_delta_type(self, test: TestResult) -> str:
        """Classify the type of delta between expected and actual."""
        if not test.expected_output or not test.actual_output:
            return "no_expected_output"
        
        if isinstance(test.expected_output, dict) and isinstance(test.actual_output, dict):
            expected_keys = set(test.expected_output.keys())
            actual_keys = set(test.actual_output.keys())
            
            if expected_keys - actual_keys:
                return "missing_fields"
            if actual_keys - expected_keys:
                return "unexpected_fields"
            if any(test.expected_output.get(k) != test.actual_output.get(k) for k in expected_keys.intersection(actual_keys)):
                return "wrong_values"
        
        return "value_mismatch"

    def _calculate_similarity_matrix(self, pattern_features: Dict[str, str]) -> np.ndarray:
        """Calculate similarity matrix between patterns."""
        pattern_names = list(pattern_features.keys())
        feature_texts = [pattern_features[name] for name in pattern_names]
        
        # Use TF-IDF for similarity
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(feature_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
        except Exception:
            # Fallback to identity matrix (no merging)
            return np.eye(len(pattern_names))

    def _merge_by_similarity(self, patterns: Dict[str, List[TestResult]], 
                            similarity_matrix: np.ndarray) -> Dict[str, List[TestResult]]:
        """Merge patterns based on similarity matrix."""
        pattern_names = list(patterns.keys())
        merged = {}
        merged_indices = set()
        
        for i, pattern_name in enumerate(pattern_names):
            if i in merged_indices:
                continue
            
            # Find similar patterns
            similar_patterns = [pattern_name]
            similar_tests = patterns[pattern_name].copy()
            
            for j in range(i + 1, len(pattern_names)):
                if j in merged_indices:
                    continue
                
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    similar_pattern = pattern_names[j]
                    similar_patterns.append(similar_pattern)
                    similar_tests.extend(patterns[similar_pattern])
                    merged_indices.add(j)
            
            # Create merged pattern name
            if len(similar_patterns) > 1:
                # Use the most descriptive name or create a combined name
                merged_name = self._create_merged_name(similar_patterns)
            else:
                merged_name = pattern_name
            
            merged[merged_name] = similar_tests
            merged_indices.add(i)
        
        return merged

    def _create_merged_name(self, pattern_names: List[str]) -> str:
        """Create a name for merged patterns."""
        # Use the shortest, most descriptive name
        if len(pattern_names) == 1:
            return pattern_names[0]
        
        # Find common words in pattern names
        name_words = [name.lower().replace("_", " ").split() for name in pattern_names]
        common_words = set(name_words[0])
        for words in name_words[1:]:
            common_words &= set(words)
        
        if common_words:
            # Use common words + count
            base_name = "_".join(sorted(common_words))
            return f"{base_name}_merged_{len(pattern_names)}"
        else:
            # Use first pattern name + merged suffix
            return f"{pattern_names[0]}_merged_{len(pattern_names)}"

