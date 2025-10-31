"""Generate specific recommendations for fixing test failures."""

import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from ..models.test_result import TestResult
from ..models.pattern import Pattern
from ..models.recommendation import Recommendation, PriorityLevel, RecommendationType
from ..analysis.claude_agent import MeritClaudeAgent


class RecommendationGenerator:
    """Generate specific, actionable recommendations for fixing test failures."""

    def __init__(self, claude_agent: MeritClaudeAgent):
        """
        Initialize recommendation generator.

        Args:
            claude_agent: Claude agent for generating recommendations
        """
        self.claude_agent = claude_agent
        self.template_recommendations = self._load_template_recommendations()

    def generate_recommendations(self,
                               pattern: Pattern,
                               root_cause: str,
                               code_context: Dict[str, str],
                               architecture: Dict[str, Any]) -> List[Recommendation]:
        """
        Generate recommendations for a specific pattern.

        Args:
            pattern: The failure pattern
            root_cause: Identified root cause
            code_context: Relevant code snippets
            architecture: System architecture

        Returns:
            List of recommendations
        """
        # Recommendations are generated - don't print progress here
        
        # Primary: Generate recommendations using LLM (most intelligent and specific)
        claude_recommendations = self._generate_claude_recommendations(
            pattern, root_cause, code_context
        )
        
        # Only use templates as fallback if LLM didn't generate enough recommendations
        # Minimum threshold: at least 2 recommendations, or templates as supplement
        MIN_RECOMMENDATIONS = 2
        all_recommendations = claude_recommendations.copy()
        
        if len(claude_recommendations) < MIN_RECOMMENDATIONS:
            # LLM generated too few recommendations - supplement with templates
            template_recommendations = self._generate_template_recommendations(
                pattern, root_cause, code_context
            )
            # Only add templates that don't duplicate LLM recommendations
            for template_rec in template_recommendations:
                # Check if similar to existing recommendations
                is_duplicate = False
                for existing_rec in all_recommendations:
                    # Simple similarity check: title similarity
                    title_similarity = len(set(template_rec.title.lower().split()) & 
                                         set(existing_rec.title.lower().split()))
                    if title_similarity >= 2:  # At least 2 words in common
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    all_recommendations.append(template_rec)
        
        # Deduplicate recommendations
        unique_recommendations = self._deduplicate_recommendations(all_recommendations)
        
        # Enhance recommendations with additional context
        enhanced_recommendations = self._enhance_recommendations(
            unique_recommendations, pattern, code_context
        )
        
        return enhanced_recommendations

    def _generate_claude_recommendations(self,
                                       pattern: Pattern,
                                       root_cause: str,
                                       code_context: Dict[str, str]) -> List[Recommendation]:
        """
        Generate recommendations using Claude.
        
        NOTE: This is now a fallback. Pattern analysis should return recommendations directly.
        If this is called, it means pattern analysis failed to provide recommendations.
        """
        try:
            # Pattern analysis should provide recommendations directly now
            # This is just a fallback that returns empty list
            # Template recommendations will be used instead
            return []
            
            # Old code that called non-existent method:
            # claude_recs = self.claude_agent.generate_recommendations(...)
            recommendations = []
            for rec_data in []:
                try:
                    recommendation = Recommendation(
                        priority=self._parse_priority(rec_data.get("priority", "medium")),
                        type=self._parse_type(rec_data.get("type", "code")),
                        title=rec_data.get("title", "Fix issue"),
                        description=rec_data.get("description", ""),
                        location=rec_data.get("location", ""),
                        implementation=rec_data.get("implementation", ""),
                        expected_impact=rec_data.get("expected_impact", ""),
                        effort_estimate=rec_data.get("effort_estimate", "Unknown"),
                        rationale=rec_data.get("rationale"),
                        code_diff=rec_data.get("code_diff"),
                        before_after_examples=rec_data.get("before_after_examples"),
                        dependencies=rec_data.get("dependencies", []),
                        tags=rec_data.get("tags", [])
                    )
                    recommendations.append(recommendation)
                except Exception as e:
                    print(f"    ⚠️  Error parsing Claude recommendation: {e}")
                    continue
            
            return recommendations
        except Exception as e:
            print(f"    ⚠️  Error generating Claude recommendations: {e}")
            return []

    def _generate_template_recommendations(self,
                                         pattern: Pattern,
                                         root_cause: str,
                                         code_context: Dict[str, str]) -> List[Recommendation]:
        """Generate recommendations based on templates."""
        recommendations = []
        
        # Get template recommendations for the root cause
        if root_cause in self.template_recommendations:
            templates = self.template_recommendations[root_cause]
            
            for template in templates:
                try:
                    # Customize template for this pattern
                    recommendation = self._customize_template(template, pattern, code_context)
                    if recommendation:
                        recommendations.append(recommendation)
                except Exception as e:
                    print(f"    ⚠️  Error customizing template: {e}")
                    continue
        
        return recommendations

    def _load_template_recommendations(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load minimal template recommendations as fallback only.
        
        These are generic fallbacks used only when LLM generation fails
        or produces too few recommendations. The LLM should handle most cases.
        """
        # Minimal template set - only used as last resort fallback
        return {
            # Keep templates minimal - LLM should handle most cases
            # Only provide very generic fallbacks for common root causes
        }

    def _customize_template(self,
                          template: Dict[str, Any],
                          pattern: Pattern,
                          code_context: Dict[str, str]) -> Optional[Recommendation]:
        """Customize a template recommendation for a specific pattern."""
        try:
            # Customize based on pattern characteristics
            title = template["title"]
            description = template["description"]
            
            # Add pattern-specific context
            if pattern.keywords:
                description += f" (Pattern keywords: {', '.join(pattern.keywords[:3])})"
            
            # Customize location if possible
            location = template.get("location", "")
            if not location and code_context:
                # Use the first file in context
                location = list(code_context.keys())[0]
            
            # Customize expected impact
            expected_impact = template.get("expected_impact", "")
            if pattern.failure_count > 1:
                expected_impact += f" (Should fix {pattern.failure_count} tests)"
            
            return Recommendation(
                priority=self._parse_priority(template.get("priority", "medium")),
                type=self._parse_type(template.get("type", "code")),
                title=title,
                description=description,
                location=location,
                implementation=template.get("implementation", ""),
                expected_impact=expected_impact,
                effort_estimate=template.get("effort_estimate", "Unknown"),
                rationale=f"Based on pattern analysis of {pattern.name}",
                tags=["template", pattern.name]
            )
        except Exception as e:
            print(f"    ⚠️  Error customizing template: {e}")
            return None

    def _deduplicate_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Remove duplicate recommendations."""
        seen = set()
        unique = []
        
        for rec in recommendations:
            # Create a key based on title and location
            key = (rec.title.lower(), rec.location.lower())
            if key not in seen:
                seen.add(key)
                unique.append(rec)
        
        return unique

    def _enhance_recommendations(self,
                               recommendations: List[Recommendation],
                               pattern: Pattern,
                               code_context: Dict[str, str]) -> List[Recommendation]:
        """Enhance recommendations with additional context."""
        enhanced = []
        
        for rec in recommendations:
            # Add pattern-specific tags
            if not rec.tags:
                rec.tags = []
            rec.tags.extend([pattern.name, "enhanced"])
            
            # Enhance implementation with code context
            if rec.implementation and code_context:
                rec.implementation = self._enhance_implementation(rec, code_context)
            
            # Add before/after examples if not present
            if not rec.before_after_examples and pattern.test_results:
                rec.before_after_examples = self._generate_before_after_examples(pattern)
            
            enhanced.append(rec)
        
        return enhanced

    def _enhance_implementation(self, 
                              recommendation: Recommendation,
                              code_context: Dict[str, str]) -> str:
        """Enhance implementation with specific code context."""
        implementation = recommendation.implementation
        
        # Add file-specific context
        if recommendation.location and recommendation.location in code_context:
            file_content = code_context[recommendation.location]
            lines = file_content.split('\n')
            
            # Find relevant lines
            relevant_lines = []
            for i, line in enumerate(lines, 1):
                if any(keyword in line.lower() for keyword in recommendation.title.lower().split()):
                    relevant_lines.append(f"Line {i}: {line.strip()}")
            
            if relevant_lines:
                implementation += f"\n\nRelevant code in {recommendation.location}:\n" + "\n".join(relevant_lines[:5])
        
        return implementation

    def _generate_before_after_examples(self, pattern: Pattern) -> Dict[str, str]:
        """Generate before/after examples from pattern tests."""
        if not pattern.test_results:
            return {}
        
        # Use first failing test as example
        failing_test = pattern.test_results[0]
        
        return {
            "Before (Failing)": f"Input: {failing_test.input}\nOutput: {failing_test.actual_output}",
            "After (Expected)": f"Input: {failing_test.input}\nOutput: {failing_test.expected_output or 'Fixed output'}"
        }

    def _parse_priority(self, priority_str: str) -> PriorityLevel:
        """Parse priority string to PriorityLevel enum."""
        priority_map = {
            "high": PriorityLevel.HIGH,
            "medium": PriorityLevel.MEDIUM,
            "low": PriorityLevel.LOW,
        }
        return priority_map.get(priority_str.lower(), PriorityLevel.MEDIUM)

    def _parse_type(self, type_str: str) -> RecommendationType:
        """Parse type string to RecommendationType enum."""
        type_map = {
            "code": RecommendationType.CODE,
            "prompt": RecommendationType.PROMPT,
            "architecture": RecommendationType.ARCHITECTURE,
            "configuration": RecommendationType.CONFIGURATION,
            "testing": RecommendationType.TESTING,
        }
        return type_map.get(type_str.lower(), RecommendationType.CODE)

    def generate_quick_fixes(self, pattern: Pattern) -> List[Recommendation]:
        """Generate quick fixes for common issues."""
        quick_fixes = []
        
        # Check for common quick fixes
        if any("timeout" in test.failure_reason.lower() for test in pattern.test_results if test.failure_reason):
            quick_fixes.append(Recommendation(
                priority=PriorityLevel.HIGH,
                type=RecommendationType.CONFIGURATION,
                title="Increase timeout values",
                description="Quick fix for timeout issues",
                location="Configuration file",
                implementation="Increase timeout values in configuration",
                expected_impact="Fixes timeout-related failures",
                effort_estimate="5 minutes",
                tags=["quick_fix", "timeout"]
            ))
        
        if any("validation" in test.failure_reason.lower() for test in pattern.test_results if test.failure_reason):
            quick_fixes.append(Recommendation(
                priority=PriorityLevel.HIGH,
                type=RecommendationType.CODE,
                title="Add input validation",
                description="Quick fix for validation issues",
                location="Input validation code",
                implementation="Add basic input validation",
                expected_impact="Prevents validation failures",
                effort_estimate="15 minutes",
                tags=["quick_fix", "validation"]
            ))
        
        return quick_fixes

    def generate_preventive_measures(self, pattern: Pattern) -> List[Recommendation]:
        """Generate preventive measures to avoid similar issues."""
        preventive = []
        
        preventive.append(Recommendation(
            priority=PriorityLevel.MEDIUM,
            type=RecommendationType.TESTING,
            title="Add regression tests",
            description="Add tests to prevent regression of this issue",
            location="Test files",
            implementation="Write tests that cover the fixed scenarios",
            expected_impact="Prevents future regressions",
            effort_estimate="1 hour",
            tags=["preventive", "testing"]
        ))
        
        preventive.append(Recommendation(
            priority=PriorityLevel.LOW,
            type=RecommendationType.CODE,
            title="Add monitoring",
            description="Add monitoring for this type of failure",
            location="Monitoring code",
            implementation="Add logging and alerts for this failure pattern",
            expected_impact="Early detection of similar issues",
            effort_estimate="30 minutes",
            tags=["preventive", "monitoring"]
        ))
        
        return preventive
