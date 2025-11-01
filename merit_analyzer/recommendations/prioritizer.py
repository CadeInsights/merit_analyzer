"""Prioritize recommendations based on impact and effort."""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import re

from ..models.recommendation import Recommendation, PriorityLevel
from ..models.pattern import Pattern


class RecommendationPrioritizer:
    """Prioritize recommendations based on various factors."""

    def __init__(self):
        """Initialize prioritizer."""
        self.priority_weights = {
            "impact": 0.4,
            "effort": 0.3,
            "urgency": 0.2,
            "dependencies": 0.1,
        }

    def prioritize_recommendations(self,
                                 recommendations: List[Recommendation],
                                 patterns: Dict[str, Pattern]) -> List[Recommendation]:
        """
        Prioritize a list of recommendations.

        Args:
            recommendations: List of recommendations to prioritize
            patterns: Dictionary of patterns for context

        Returns:
            List of recommendations sorted by priority
        """
        if not recommendations:
            return []
        
        print(f"  📊 Prioritizing {len(recommendations)} recommendations...")
        
        # Deduplicate recommendations first
        unique_recommendations = self._deduplicate_recommendations(recommendations)
        print(f"  🔄 Removed {len(recommendations) - len(unique_recommendations)} duplicate recommendations")
        
        # Calculate priority scores
        scored_recommendations = []
        for rec in unique_recommendations:
            score = self._calculate_priority_score(rec, patterns, unique_recommendations)
            scored_recommendations.append((rec, score))
        
        # Sort by score (higher is better)
        scored_recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Update priority levels based on ranking
        prioritized = []
        total = len(scored_recommendations)
        
        for i, (rec, score) in enumerate(scored_recommendations):
            # Assign priority based on ranking
            if i < total * 0.2:  # Top 20%
                rec.priority = PriorityLevel.HIGH
            elif i < total * 0.6:  # Next 40%
                rec.priority = PriorityLevel.MEDIUM
            else:  # Bottom 40%
                rec.priority = PriorityLevel.LOW
            
            prioritized.append(rec)
        
        return prioritized

    def _deduplicate_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Remove duplicate recommendations based on title and description similarity."""
        if not recommendations:
            return []
        
        unique_recs = []
        seen_titles = set()
        
        for rec in recommendations:
            # Create a normalized title for comparison
            normalized_title = self._normalize_text(rec.title)
            
            # Check if we've seen a similar recommendation
            is_duplicate = False
            for seen_title in seen_titles:
                if self._are_similar(normalized_title, seen_title):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_recs.append(rec)
                seen_titles.add(normalized_title)
        
        return unique_recs

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Remove common words that don't add meaning
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in normalized.split() if word not in stop_words]
        
        return ' '.join(words)

    def _are_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar enough to be considered duplicates."""
        if not text1 or not text2:
            return False
        
        # Simple similarity check - if 80% of words match
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0
        return similarity > 0.8

    def _calculate_priority_score(self, 
                                recommendation: Recommendation,
                                patterns: Dict[str, Pattern],
                                all_recommendations: List[Recommendation]) -> float:
        """Calculate priority score for a recommendation."""
        score = 0.0
        
        # Impact score (0-1)
        impact_score = self._calculate_impact_score(recommendation, patterns)
        score += impact_score * self.priority_weights["impact"]
        
        # Effort score (0-1, inverted - lower effort = higher score)
        effort_score = self._calculate_effort_score(recommendation)
        score += effort_score * self.priority_weights["effort"]
        
        # Urgency score (0-1)
        urgency_score = self._calculate_urgency_score(recommendation)
        score += urgency_score * self.priority_weights["urgency"]
        
        # Dependencies score (0-1)
        dependencies_score = self._calculate_dependencies_score(recommendation, all_recommendations)
        score += dependencies_score * self.priority_weights["dependencies"]
        
        return min(score, 1.0)

    def _calculate_impact_score(self, 
                              recommendation: Recommendation,
                              patterns: Dict[str, Pattern]) -> float:
        """Calculate impact score based on expected test fixes."""
        score = 0.0
        
        # Handle None expected_impact (now optional field)
        if not recommendation.expected_impact:
            return 0.0
        
        # Base score from expected impact text
        expected_impact = recommendation.expected_impact.lower()
        
        # Look for number of tests mentioned
        test_count_match = re.search(r'(\d+)\s*tests?', expected_impact)
        if test_count_match:
            test_count = int(test_count_match.group(1))
            score += min(test_count / 10.0, 0.5)  # Cap at 0.5 for test count
        
        # Look for impact keywords
        high_impact_keywords = ['fixes', 'resolves', 'prevents', 'eliminates', 'solves']
        medium_impact_keywords = ['improves', 'enhances', 'reduces', 'minimizes']
        low_impact_keywords = ['helps', 'assists', 'supports', 'aids']
        
        if any(keyword in expected_impact for keyword in high_impact_keywords):
            score += 0.4
        elif any(keyword in expected_impact for keyword in medium_impact_keywords):
            score += 0.2
        elif any(keyword in expected_impact for keyword in low_impact_keywords):
            score += 0.1
        
        # Boost score for high-priority types
        if recommendation.type.value in ['code', 'architecture']:
            score += 0.2
        elif recommendation.type.value in ['prompt', 'configuration']:
            score += 0.1
        
        return min(score, 1.0)

    def _calculate_effort_score(self, recommendation: Recommendation) -> float:
        """Calculate effort score (inverted - lower effort = higher score)."""
        effort_text = recommendation.effort_estimate.lower()
        
        # Parse effort estimates
        if 'minute' in effort_text:
            minutes_match = re.search(r'(\d+)\s*minute', effort_text)
            if minutes_match:
                minutes = int(minutes_match.group(1))
                if minutes <= 15:
                    return 1.0
                elif minutes <= 30:
                    return 0.8
                elif minutes <= 60:
                    return 0.6
                else:
                    return 0.4
        
        elif 'hour' in effort_text:
            hours_match = re.search(r'(\d+)\s*hour', effort_text)
            if hours_match:
                hours = int(hours_match.group(1))
                if hours <= 1:
                    return 0.6
                elif hours <= 2:
                    return 0.4
                elif hours <= 4:
                    return 0.2
                else:
                    return 0.1
        
        elif 'day' in effort_text:
            return 0.1
        
        # Default scores based on text
        if 'quick' in effort_text or 'easy' in effort_text:
            return 0.9
        elif 'simple' in effort_text:
            return 0.7
        elif 'complex' in effort_text or 'difficult' in effort_text:
            return 0.2
        elif 'unknown' in effort_text:
            return 0.5
        
        return 0.5

    def _calculate_urgency_score(self, recommendation: Recommendation) -> float:
        """Calculate urgency score based on recommendation characteristics."""
        score = 0.0
        
        # High urgency indicators
        title_lower = recommendation.title.lower()
        description_lower = recommendation.description.lower()
        
        urgent_keywords = ['critical', 'urgent', 'immediate', 'asap', 'fix', 'error', 'bug', 'crash']
        if any(keyword in title_lower or keyword in description_lower for keyword in urgent_keywords):
            score += 0.4
        
        # Type-based urgency
        if recommendation.type.value == 'code':
            score += 0.3
        elif recommendation.type.value == 'configuration':
            score += 0.2
        elif recommendation.type.value == 'prompt':
            score += 0.1
        
        # Priority-based urgency
        if recommendation.priority == PriorityLevel.HIGH:
            score += 0.3
        elif recommendation.priority == PriorityLevel.MEDIUM:
            score += 0.1
        
        return min(score, 1.0)

    def _calculate_dependencies_score(self, 
                                    recommendation: Recommendation,
                                    all_recommendations: List[Recommendation]) -> float:
        """Calculate dependencies score."""
        if not recommendation.dependencies:
            return 1.0  # No dependencies = high score
        
        # Check if dependencies are satisfied
        satisfied_deps = 0
        for dep in recommendation.dependencies:
            # Look for dependency in other recommendations
            for other_rec in all_recommendations:
                if other_rec.title.lower() in dep.lower() or dep.lower() in other_rec.title.lower():
                    satisfied_deps += 1
                    break
        
        if satisfied_deps == len(recommendation.dependencies):
            return 1.0  # All dependencies satisfied
        elif satisfied_deps > 0:
            return 0.5  # Some dependencies satisfied
        else:
            return 0.1  # No dependencies satisfied

    def group_recommendations_by_type(self, 
                                    recommendations: List[Recommendation]) -> Dict[str, List[Recommendation]]:
        """Group recommendations by type."""
        grouped = defaultdict(list)
        
        for rec in recommendations:
            grouped[rec.type.value].append(rec)
        
        return dict(grouped)

    def group_recommendations_by_effort(self, 
                                      recommendations: List[Recommendation]) -> Dict[str, List[Recommendation]]:
        """Group recommendations by effort level."""
        grouped = {
            "quick": [],      # < 30 minutes
            "medium": [],     # 30 minutes - 2 hours
            "long": []        # > 2 hours
        }
        
        for rec in recommendations:
            effort_text = rec.effort_estimate.lower()
            
            if 'minute' in effort_text:
                minutes_match = re.search(r'(\d+)\s*minute', effort_text)
                if minutes_match:
                    minutes = int(minutes_match.group(1))
                    if minutes < 30:
                        grouped["quick"].append(rec)
                    else:
                        grouped["medium"].append(rec)
                else:
                    grouped["medium"].append(rec)
            elif 'hour' in effort_text:
                hours_match = re.search(r'(\d+)\s*hour', effort_text)
                if hours_match:
                    hours = int(hours_match.group(1))
                    if hours <= 2:
                        grouped["medium"].append(rec)
                    else:
                        grouped["long"].append(rec)
                else:
                    grouped["medium"].append(rec)
            else:
                grouped["medium"].append(rec)
        
        return grouped

    def create_implementation_plan(self, 
                                 recommendations: List[Recommendation]) -> List[Dict[str, Any]]:
        """Create an implementation plan with phases."""
        # Group by effort
        effort_groups = self.group_recommendations_by_effort(recommendations)
        
        plan = []
        
        # Phase 1: Quick wins
        if effort_groups["quick"]:
            plan.append({
                "phase": 1,
                "name": "Quick Wins",
                "description": "Low-effort, high-impact fixes",
                "recommendations": effort_groups["quick"][:5],  # Limit to 5
                "estimated_time": "30 minutes - 1 hour",
                "expected_impact": "Immediate improvement"
            })
        
        # Phase 2: Medium effort
        if effort_groups["medium"]:
            plan.append({
                "phase": 2,
                "name": "Core Fixes",
                "description": "Medium-effort fixes for core issues",
                "recommendations": effort_groups["medium"][:8],  # Limit to 8
                "estimated_time": "2-4 hours",
                "expected_impact": "Significant improvement"
            })
        
        # Phase 3: Long-term improvements
        if effort_groups["long"]:
            plan.append({
                "phase": 3,
                "name": "Long-term Improvements",
                "description": "High-effort improvements for long-term stability",
                "recommendations": effort_groups["long"][:5],  # Limit to 5
                "estimated_time": "1-2 days",
                "expected_impact": "Long-term stability and performance"
            })
        
        return plan

    def calculate_roi(self, recommendation: Recommendation) -> float:
        """Calculate return on investment for a recommendation."""
        # Simple ROI calculation: impact / effort
        impact_score = self._calculate_impact_score(recommendation, {})
        effort_score = self._calculate_effort_score(recommendation)
        
        if effort_score == 0:
            return 0.0
        
        # ROI = impact / (1 - effort_score) to invert effort
        roi = impact_score / (1 - effort_score + 0.1)  # Add small constant to avoid division by zero
        return min(roi, 10.0)  # Cap at 10

    def get_top_recommendations(self, 
                              recommendations: List[Recommendation],
                              count: int = 5) -> List[Recommendation]:
        """Get top N recommendations by priority score."""
        if not recommendations:
            return []
        
        # Calculate scores and sort
        scored = [(rec, self._calculate_priority_score(rec, {}, recommendations)) for rec in recommendations]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [rec for rec, score in scored[:count]]

    def get_quick_wins(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Get recommendations that are quick wins (low effort, high impact)."""
        quick_wins = []
        
        for rec in recommendations:
            impact_score = self._calculate_impact_score(rec, {})
            effort_score = self._calculate_effort_score(rec)
            
            # Quick win: high impact (>= 0.6) and low effort (>= 0.7)
            if impact_score >= 0.6 and effort_score >= 0.7:
                quick_wins.append(rec)
        
        return quick_wins
