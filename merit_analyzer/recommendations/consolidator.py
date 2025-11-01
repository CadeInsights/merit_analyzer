"""Consolidate similar recommendations to reduce redundancy."""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
from ..models.recommendation import Recommendation


class RecommendationConsolidator:
    """Consolidates similar recommendations into high-impact fixes."""
    
    def __init__(self):
        self.similarity_threshold = 0.7
    
    def consolidate_recommendations(
        self,
        recommendations: List[Recommendation],
        patterns: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        """
        Consolidate similar recommendations and calculate impact.
        
        Returns:
            - List of consolidated recommendations with impact scores
            - Mapping of recommendation groups to affected patterns
        """
        # Group recommendations by similarity
        groups = self._group_similar_recommendations(recommendations)
        
        # Create consolidated recommendations with impact scores
        consolidated = []
        impact_map = {}
        
        for group_id, group_recs in groups.items():
            # Calculate which patterns are affected by this group
            affected_patterns = self._get_affected_patterns(group_recs, patterns)
            
            # Create consolidated recommendation
            primary = group_recs[0]  # Use first as template
            
            # Get all unique locations
            all_locations = self._get_all_locations(group_recs)
            primary_location = self._consolidate_locations(group_recs)
            
            consolidated_rec = {
                'id': f'consolidated_{group_id}',
                'title': self._consolidate_titles(group_recs),
                'description': self._consolidate_descriptions(group_recs),
                'priority': self._get_highest_priority(group_recs),
                'type': primary.type.value,
                'location': primary_location,  # Primary location for backward compatibility
                'locations': all_locations,  # All locations where changes are needed
                'effort_estimate': self._consolidate_effort(group_recs),
                'expected_impact': f"Fixes {len(affected_patterns)} patterns ({len(group_recs)} related issues)",
                'impact_score': len(affected_patterns),  # Number of patterns fixed
                'patterns_fixed': affected_patterns,
                'implementation': self._merge_implementations(group_recs),
                'rationale': self._merge_rationales(group_recs),
                'original_count': len(group_recs),
                'category': self._categorize_recommendation(primary)
            }
            
            consolidated.append(consolidated_rec)
            impact_map[group_id] = affected_patterns
        
        # Merge any consolidated recs with identical titles (post-processing)
        consolidated = self._merge_duplicate_titles(consolidated)
        
        # Sort by impact score (descending)
        consolidated.sort(key=lambda x: (x['impact_score'], x['priority']), reverse=True)
        
        return consolidated, impact_map
    
    def _group_similar_recommendations(self, recommendations: List[Recommendation]) -> Dict[int, List[Recommendation]]:
        """Group similar recommendations together."""
        groups = defaultdict(list)
        group_id = 0
        processed = set()
        
        for i, rec in enumerate(recommendations):
            if i in processed:
                continue
            
            # Start new group
            groups[group_id].append(rec)
            processed.add(i)
            
            # Find similar recommendations
            for j, other_rec in enumerate(recommendations[i+1:], start=i+1):
                if j in processed:
                    continue
                
                if self._are_similar(rec, other_rec):
                    groups[group_id].append(other_rec)
                    processed.add(j)
            
            group_id += 1
        
        return groups
    
    def _are_similar(self, rec1: Recommendation, rec2: Recommendation) -> bool:
        """Check if two recommendations are similar enough to consolidate."""
        # Same type and priority
        if rec1.type != rec2.type:
            return False
        
        # Check title similarity
        title1_words = set(rec1.title.lower().split())
        title2_words = set(rec2.title.lower().split())
        
        if not title1_words or not title2_words:
            return False
        
        overlap = len(title1_words & title2_words) / max(len(title1_words), len(title2_words))
        
        if overlap >= self.similarity_threshold:
            return True
        
        # Check description similarity for key phrases
        desc1_lower = rec1.description.lower()
        desc2_lower = rec2.description.lower()
        
        # Common patterns
        common_patterns = [
            'character limit', 'truncation', 'content length',
            'empty result', 'no results', 'fallback',
            'hallucination', 'fabricate', 'validation',
            'accessibility', 'jargon', 'technical',
            'greeting', 'sign-off', 'formatting',
            'search', 'query', 'search strategy'
        ]
        
        rec1_patterns = [p for p in common_patterns if p in desc1_lower]
        rec2_patterns = [p for p in common_patterns if p in desc2_lower]
        
        if rec1_patterns and rec2_patterns:
            pattern_overlap = len(set(rec1_patterns) & set(rec2_patterns)) / max(len(rec1_patterns), len(rec2_patterns))
            if pattern_overlap >= 0.5:
                return True
        
        return False
    
    def _get_affected_patterns(self, recommendations: List[Recommendation], patterns: Dict[str, Any]) -> List[str]:
        """Get list of pattern names affected by these recommendations."""
        affected = set()
        
        for rec in recommendations:
            # Extract pattern name from rationale
            if rec.rationale:
                for pattern_name in patterns.keys():
                    if pattern_name in rec.rationale.lower():
                        affected.add(pattern_name)
        
        return sorted(list(affected))
    
    def _consolidate_titles(self, recommendations: List[Recommendation]) -> str:
        """Create a consolidated title."""
        if len(recommendations) == 1:
            return recommendations[0].title
        
        # Find common theme
        first_title = recommendations[0].title.lower()
        
        # Common consolidation patterns
        if any(term in first_title for term in ['character limit', 'truncation', 'content length']):
            return "Increase content character limits across search tools"
        elif any(term in first_title for term in ['empty result', 'no results', 'fallback']):
            return "Add result validation and fallback search strategies"
        elif any(term in first_title for term in ['hallucination', 'fabricate', 'validation']):
            return "Add anti-hallucination guardrails and data validation"
        elif any(term in first_title for term in ['accessibility', 'jargon', 'technical']):
            return "Improve content accessibility for non-technical readers"
        elif any(term in first_title for term in ['greeting', 'sign-off', 'formatting']):
            return "Enforce formatting requirements (greetings, sign-offs)"
        else:
            # Use first title
            return recommendations[0].title
    
    def _consolidate_descriptions(self, recommendations: List[Recommendation]) -> str:
        """Create a consolidated description."""
        if len(recommendations) == 1:
            return recommendations[0].description
        
        # Create summary of all descriptions
        summary = f"This fix addresses {len(recommendations)} related issues:\n\n"
        
        # Group by location
        by_location = defaultdict(list)
        for rec in recommendations:
            by_location[rec.location].append(rec)
        
        for location, recs in by_location.items():
            if location:
                summary += f"**{location}:**\n"
            for rec in recs[:3]:  # Limit to 3 per location
                summary += f"- {rec.description.split('.')[0]}.\n"
            if len(recs) > 3:
                summary += f"- ...and {len(recs) - 3} more issues\n"
            summary += "\n"
        
        return summary.strip()
    
    def _get_all_locations(self, recommendations: List[Recommendation]) -> List[str]:
        """Get all unique locations from recommendations."""
        locations = [rec.location for rec in recommendations if rec.location]
        # Remove duplicates while preserving order
        seen = set()
        unique_locations = []
        for loc in locations:
            if loc not in seen:
                seen.add(loc)
                unique_locations.append(loc)
        return unique_locations
    
    def _consolidate_locations(self, recommendations: List[Recommendation]) -> str:
        """
        Consolidate file locations into a single string for backward compatibility.
        Returns the most common location or a summary if multiple.
        """
        locations = [rec.location for rec in recommendations if rec.location]
        
        if not locations:
            return ""
        
        unique_locations = list(set(locations))
        
        if len(unique_locations) == 1:
            return unique_locations[0]
        
        # If multiple locations, show summary with count
        files = set()
        for loc in unique_locations:
            if ':' in loc:
                files.add(loc.split(':')[0])
            else:
                files.add(loc)
        
        if len(files) == 1:
            # Multiple locations in same file
            return f"{list(files)[0]} (multiple locations)"
        
        return f"Multiple files: {', '.join(sorted(files)[:3])}" + (" ..." if len(files) > 3 else "")
    
    def _consolidate_effort(self, recommendations: List[Recommendation]) -> str:
        """Consolidate effort estimates."""
        efforts = [rec.effort_estimate for rec in recommendations]
        
        # Count effort levels
        low_count = sum(1 for e in efforts if 'low' in e.lower())
        medium_count = sum(1 for e in efforts if 'medium' in e.lower())
        high_count = sum(1 for e in efforts if 'high' in e.lower())
        
        if len(recommendations) == 1:
            return efforts[0]
        
        # Return highest effort level
        if high_count > 0:
            return "medium"  # Multiple changes, even if individual is low
        elif medium_count > len(recommendations) / 2:
            return "medium"
        else:
            return "low"
    
    def _get_highest_priority(self, recommendations: List[Recommendation]) -> str:
        """Get the highest priority from the group."""
        priorities = [rec.priority.value for rec in recommendations]
        
        if "high" in priorities:
            return "high"
        elif "medium" in priorities:
            return "medium"
        else:
            return "low"
    
    def _merge_implementations(self, recommendations: List[Recommendation]) -> str:
        """Merge implementation steps."""
        if len(recommendations) == 1:
            return recommendations[0].implementation or ""
        
        impl = "**Consolidated implementation steps:**\n\n"
        
        # Get unique implementations
        seen = set()
        step_num = 1
        
        for rec in recommendations[:5]:  # Limit to top 5
            if not rec.implementation:
                continue
            impl_clean = rec.implementation.strip()
            if impl_clean and impl_clean not in seen:
                impl += f"{step_num}. {impl_clean.split(chr(10))[0]}\n"
                seen.add(impl_clean)
                step_num += 1
        
        if len(recommendations) > 5:
            impl += f"\n...plus {len(recommendations) - 5} more related implementations.\n"
        
        return impl
    
    def _merge_rationales(self, recommendations: List[Recommendation]) -> str:
        """Merge rationales."""
        if len(recommendations) == 1:
            return recommendations[0].rationale or ""
        
        # Extract unique pattern names mentioned
        patterns = set()
        for rec in recommendations:
            if rec.rationale:
                # Extract pattern names from rationale
                words = rec.rationale.lower().split()
                for i, word in enumerate(words):
                    if word == "analysis" and i > 0:
                        # Pattern name likely before "analysis"
                        pattern = words[i-1].replace('_', ' ')
                        patterns.add(pattern)
        
        if patterns:
            return f"Addresses failures in: {', '.join(sorted(patterns)[:5])}"
        
        return f"Consolidates {len(recommendations)} similar recommendations"
    
    def _merge_duplicate_titles(self, consolidated: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge consolidated recommendations that have identical titles."""
        title_groups = defaultdict(list)
        
        # Group by exact title match
        for rec in consolidated:
            title_groups[rec['title']].append(rec)
        
        # Merge groups with same title
        merged = []
        for title, recs in title_groups.items():
            if len(recs) == 1:
                merged.append(recs[0])
            else:
                # Merge multiple recommendations with same title
                primary = recs[0]
                
                # Combine patterns_fixed (remove duplicates)
                all_patterns = set()
                for rec in recs:
                    all_patterns.update(rec['patterns_fixed'])
                
                # Combine implementation and rationale (filter out empty ones)
                implementations = [r['implementation'] for r in recs if r.get('implementation')]
                rationales = [r['rationale'] for r in recs if r.get('rationale')]
                
                # Merge all locations from all recommendations
                all_merged_locations = []
                for rec in recs:
                    if rec.get('locations'):
                        all_merged_locations.extend(rec['locations'])
                # Remove duplicates while preserving order
                seen_locs = set()
                unique_merged_locations = []
                for loc in all_merged_locations:
                    if loc and loc not in seen_locs:
                        seen_locs.add(loc)
                        unique_merged_locations.append(loc)
                
                merged_rec = {
                    'id': primary['id'],
                    'title': title,
                    'description': self._merge_descriptions(recs),
                    'priority': self._get_highest_priority_from_dicts(recs),
                    'type': primary['type'],
                    'location': self._merge_locations(recs),
                    'locations': unique_merged_locations,  # All unique locations
                    'effort_estimate': primary['effort_estimate'],
                    'expected_impact': f"Fixes {len(all_patterns)} patterns ({sum(r['original_count'] for r in recs)} related issues)",
                    'impact_score': len(all_patterns),
                    'patterns_fixed': sorted(list(all_patterns)),
                    'implementation': '\n\n'.join(implementations) if implementations else '',
                    'rationale': ' | '.join(rationales) if rationales else '',
                    'original_count': sum(r['original_count'] for r in recs),
                    'category': primary['category']
                }
                merged.append(merged_rec)
        
        return merged
    
    def _merge_descriptions(self, recs: List[Dict[str, Any]]) -> str:
        """Merge descriptions from multiple consolidated recommendations."""
        # Combine unique description points
        combined = f"This fix addresses {sum(r['original_count'] for r in recs)} related issues:\n\n"
        
        seen_points = set()
        for rec in recs:
            # Extract key points from description
            lines = rec['description'].split('\n')
            for line in lines:
                line_clean = line.strip()
                if line_clean and line_clean not in seen_points and not line_clean.startswith('This fix'):
                    combined += f"{line_clean}\n"
                    seen_points.add(line_clean)
        
        return combined.strip()
    
    def _merge_locations(self, recs: List[Dict[str, Any]]) -> str:
        """Merge locations from multiple consolidated recommendations."""
        all_locations = []
        for rec in recs:
            if rec['location']:
                all_locations.append(rec['location'])
        
        if not all_locations:
            return ""
        
        # Extract unique files
        files = set()
        for loc in all_locations:
            if 'Multiple files:' in loc:
                # Already a multi-file location
                parts = loc.replace('Multiple files:', '').strip().split(',')
                files.update(p.strip() for p in parts if p.strip() and p.strip() != '...')
            elif loc:
                files.add(loc.split(':')[0] if ':' in loc else loc)
        
        if len(files) == 1:
            return list(files)[0]
        
        return f"Multiple files: {', '.join(sorted(files)[:3])}" + (" ..." if len(files) > 3 else "")
    
    def _get_highest_priority_from_dicts(self, recs: List[Dict[str, Any]]) -> str:
        """Get the highest priority from consolidated recommendation dicts."""
        priorities = [rec['priority'] for rec in recs]
        
        if "high" in priorities:
            return "high"
        elif "medium" in priorities:
            return "medium"
        else:
            return "low"
    
    def _categorize_recommendation(self, rec: Recommendation) -> str:
        """Categorize the recommendation for grouping."""
        title_lower = rec.title.lower()
        desc_lower = rec.description.lower()
        
        categories = {
            'Data Truncation': ['character limit', 'truncation', 'content length', 'truncate'],
            'Search & Retrieval': ['search', 'query', 'exa', 'fallback', 'result validation'],
            'Data Quality': ['hallucination', 'validation', 'empty result', 'fabricate', 'verify'],
            'Content Accessibility': ['accessibility', 'jargon', 'technical', 'plain language'],
            'Output Formatting': ['greeting', 'sign-off', 'formatting', 'structure'],
            'Agent Instructions': ['backstory', 'task description', 'instructions', 'agent'],
            'Error Handling': ['error', 'exception', 'fallback', 'handle']
        }
        
        for category, keywords in categories.items():
            if any(kw in title_lower or kw in desc_lower for kw in keywords):
                return category
        
        return 'Other'

