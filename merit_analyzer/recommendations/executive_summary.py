"""Generate executive summary for analysis reports."""

from typing import List, Dict, Any
from collections import defaultdict


class ExecutiveSummaryGenerator:
    """Generates executive summaries with strategic recommendations."""
    
    def generate_summary(
        self,
        consolidated_recs: List[Dict[str, Any]],
        patterns: Dict[str, Any],
        summary_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate executive summary with top recommendations.
        
        Returns dict with:
        - overview: High-level stats
        - top_fixes: Top 5-10 highest-impact recommendations
        - quick_wins: Low-effort, high-impact fixes
        - implementation_order: Suggested order to implement fixes
        - categories: Breakdown by category
        """
        # Sort by impact score
        sorted_recs = sorted(
            consolidated_recs,
            key=lambda x: (x['impact_score'], x['priority'] == 'high', x['priority'] == 'medium'),
            reverse=True
        )
        
        # Get top recommendations
        top_fixes = sorted_recs[:10]
        
        # Identify quick wins (low effort, high impact)
        quick_wins = [
            rec for rec in sorted_recs
            if rec['effort_estimate'] == 'low' and rec['impact_score'] >= 2
        ][:5]
        
        # Calculate coverage
        total_patterns = len(patterns)
        patterns_addressed = set()
        for rec in top_fixes:
            patterns_addressed.update(rec.get('patterns_fixed', []))
        
        coverage_pct = (len(patterns_addressed) / total_patterns * 100) if total_patterns > 0 else 0
        
        # Group by category
        by_category = defaultdict(list)
        for rec in consolidated_recs:
            by_category[rec.get('category', 'Other')].append(rec)
        
        # Determine implementation order
        implementation_order = self._generate_implementation_order(consolidated_recs, by_category)
        
        return {
            'overview': {
                'total_tests': summary_stats.get('total_tests', 0),
                'passed': summary_stats.get('passed', 0),
                'total_failures': summary_stats.get('failed', 0),
                'pass_rate': f"{summary_stats.get('passed', 0) / summary_stats.get('total_tests', 1) * 100:.1f}%",
                'analysis_time': f"{summary_stats.get('analysis_duration_seconds', 0):.1f}s",
                'patterns_identified': total_patterns,
                'total_recommendations': len(consolidated_recs),
                'top_fixes_coverage': f"{coverage_pct:.0f}%",
                'patterns_addressed_by_top_fixes': len(patterns_addressed),
                'quick_wins_available': len(quick_wins)
            },
            'top_fixes': top_fixes,
            'quick_wins': quick_wins,
            'implementation_order': implementation_order,
            'categories': {
                category: {
                    'count': len(recs),
                    'total_impact': sum(r['impact_score'] for r in recs),
                    'top_recommendation': recs[0]['title'] if recs else None
                }
                for category, recs in by_category.items()
            }
        }
    
    def _generate_implementation_order(
        self,
        consolidated_recs: List[Dict[str, Any]],
        by_category: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Generate suggested implementation order.
        
        Strategy:
        1. Infrastructure fixes (data quality, search) first
        2. High-impact, low-effort fixes
        3. Content/formatting fixes
        4. Pattern-specific fixes
        """
        order = []
        
        # Phase 1: Infrastructure (Data quality, Search)
        infra_categories = ['Data Truncation', 'Search & Retrieval', 'Data Quality']
        infra_recs = []
        for cat in infra_categories:
            infra_recs.extend(by_category.get(cat, []))
        
        if infra_recs:
            # Sort by impact
            infra_recs.sort(key=lambda x: x['impact_score'], reverse=True)
            order.append({
                'phase': 1,
                'name': 'Infrastructure & Data Quality',
                'description': 'Fix foundational issues with data retrieval and validation',
                'recommendations': [r['id'] for r in infra_recs[:5]],
                'impact': f"~{sum(r['impact_score'] for r in infra_recs[:5])} patterns"
            })
        
        # Phase 2: Agent Configuration (Instructions, prompts)
        agent_categories = ['Agent Instructions', 'Content Accessibility', 'Output Formatting']
        agent_recs = []
        for cat in agent_categories:
            agent_recs.extend(by_category.get(cat, []))
        
        if agent_recs:
            agent_recs.sort(key=lambda x: x['impact_score'], reverse=True)
            order.append({
                'phase': 2,
                'name': 'Agent Instructions & Output Quality',
                'description': 'Improve agent prompts, instructions, and output formatting',
                'recommendations': [r['id'] for r in agent_recs[:5]],
                'impact': f"~{sum(r['impact_score'] for r in agent_recs[:5])} patterns"
            })
        
        # Phase 3: Error Handling & Edge Cases
        error_recs = by_category.get('Error Handling', [])
        other_recs = by_category.get('Other', [])
        remaining_recs = error_recs + other_recs
        
        if remaining_recs:
            remaining_recs.sort(key=lambda x: x['impact_score'], reverse=True)
            order.append({
                'phase': 3,
                'name': 'Error Handling & Edge Cases',
                'description': 'Handle edge cases and improve error handling',
                'recommendations': [r['id'] for r in remaining_recs[:5]],
                'impact': f"~{sum(r['impact_score'] for r in remaining_recs[:5])} patterns"
            })
        
        return order
    
    def format_executive_summary_markdown(self, summary: Dict[str, Any]) -> str:
        """Format executive summary as markdown."""
        md = "# Executive Summary\n\n"
        
        # Overview
        overview = summary['overview']
        md += "## At a Glance\n\n"
        md += f"- **Total Failures:** {overview['total_failures']} tests\n"
        md += f"- **Failure Patterns:** {overview['patterns_identified']} distinct patterns identified\n"
        md += f"- **Recommendations Generated:** {overview['total_recommendations']} (consolidated from similar issues)\n"
        md += f"- **Top {len(summary['top_fixes'])} Fixes Cover:** {overview['top_fixes_coverage']} of patterns ({overview['patterns_addressed_by_top_fixes']}/{overview['patterns_identified']})\n"
        
        if overview['quick_wins_available'] > 0:
            md += f"- **Quick Wins Available:** {overview['quick_wins_available']} low-effort, high-impact fixes\n"
        
        md += "\n---\n\n"
        
        # Top Fixes
        md += f"## Top {len(summary['top_fixes'])} Recommended Fixes\n\n"
        md += "*These fixes address the most patterns with the highest impact:*\n\n"
        
        for i, fix in enumerate(summary['top_fixes'], 1):
            priority_emoji = "🔴" if fix['priority'] == 'high' else "🟡" if fix['priority'] == 'medium' else "🟢"
            md += f"### {i}. {fix['title']}\n\n"
            md += f"{priority_emoji} **Priority:** {fix['priority'].title()} | "
            md += f"**Effort:** {fix['effort_estimate']} | "
            md += f"**Impact:** Fixes {fix['impact_score']} patterns\n\n"
            
            # Show locations where changes are needed
            if fix.get('locations') and len(fix['locations']) > 0:
                if len(fix['locations']) == 1:
                    md += f"**Location:** `{fix['locations'][0]}`\n\n"
                else:
                    md += f"**Locations:** (changes needed in {len(fix['locations'])} places)\n"
                    for loc in fix['locations'][:5]:  # Show up to 5 locations
                        md += f"  - `{loc}`\n"
                    if len(fix['locations']) > 5:
                        md += f"  - ...and {len(fix['locations']) - 5} more\n"
                    md += "\n"
            
            if fix.get('patterns_fixed'):
                md += f"**Patterns Fixed:** {', '.join(fix['patterns_fixed'][:3])}"
                if len(fix['patterns_fixed']) > 3:
                    md += f" (+{len(fix['patterns_fixed']) - 3} more)"
                md += "\n\n"
            
            # Brief description
            desc_lines = fix['description'].split('\n')[:3]
            md += f"{' '.join(desc_lines)}\n\n"
            
            if i < len(summary['top_fixes']):
                md += "---\n\n"
        
        # Quick Wins
        if summary['quick_wins']:
            md += "\n## ⚡ Quick Wins\n\n"
            md += "*Low-effort fixes with high impact - start here:*\n\n"
            
            for qw in summary['quick_wins']:
                md += f"- **{qw['title']}** (Fixes {qw['impact_score']} patterns)\n"
            
            md += "\n"
        
        # Implementation Order
        if summary['implementation_order']:
            md += "## 📋 Suggested Implementation Order\n\n"
            
            for phase in summary['implementation_order']:
                md += f"### Phase {phase['phase']}: {phase['name']}\n\n"
                md += f"{phase['description']}\n\n"
                md += f"**Expected Impact:** {phase['impact']}\n\n"
                md += f"**Recommendations:** {len(phase['recommendations'])} fixes in this phase\n\n"
        
        # Category Breakdown
        if summary['categories']:
            md += "\n## 📊 Issue Categories\n\n"
            
            sorted_cats = sorted(
                summary['categories'].items(),
                key=lambda x: x[1]['total_impact'],
                reverse=True
            )
            
            for category, info in sorted_cats:
                if info['count'] > 0:
                    md += f"- **{category}:** {info['count']} recommendations "
                    md += f"(~{info['total_impact']} patterns affected)\n"
        
        return md
    
    def format_executive_summary_cli(self, summary: Dict[str, Any]) -> str:
        """Format executive summary for CLI display."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        
        console = Console()
        output = []
        
        # Overview panel with full analysis stats
        overview = summary['overview']
        
        # Create stats table for left side
        stats_lines = [
            f"[cyan]Total Tests:[/cyan]     {overview.get('total_tests', 0)}",
            f"[cyan]Passed:[/cyan]         [green]{overview.get('passed', 0)}[/green]",
            f"[cyan]Failed:[/cyan]         [red]{overview['total_failures']}[/red]",
            f"[cyan]Pass Rate:[/cyan]      {overview.get('pass_rate', '0')}",
            "",
            f"[cyan]Patterns:[/cyan]       {overview['patterns_identified']}",
            f"[cyan]Consolidated:[/cyan]   {overview['total_recommendations']} fixes",
            f"[cyan]Analysis Time:[/cyan]  {overview.get('analysis_time', '0s')}",
        ]
        
        # Create summary for right side
        summary_lines = [
            f"[bold green]✓ Top {len(summary['top_fixes'])} fixes address {overview['top_fixes_coverage']} of patterns[/bold green]",
            "",
        ]
        
        if overview['quick_wins_available'] > 0:
            summary_lines.append(f"[bold yellow]⚡ {overview['quick_wins_available']} quick wins available![/bold yellow]")
        
        # Combine into two columns
        overview_text = "\n".join(stats_lines) + "\n\n" + "\n".join(summary_lines)
        
        output.append(Panel(overview_text, title="📊 Executive Summary", border_style="cyan"))
        
        # Top Fixes table
        table = Table(title=f"🎯 Top {len(summary['top_fixes'])} Recommended Fixes", box=box.ROUNDED)
        table.add_column("#", style="dim", width=3)
        table.add_column("Fix", style="cyan", width=50)
        table.add_column("Impact", justify="right", style="green")
        table.add_column("Effort", justify="center", style="yellow")
        table.add_column("Priority", justify="center")
        
        for i, fix in enumerate(summary['top_fixes'], 1):
            priority_style = "red" if fix['priority'] == 'high' else "yellow" if fix['priority'] == 'medium' else "green"
            
            title = fix['title']
            if len(title) > 47:
                title = title[:44] + "..."
            
            table.add_row(
                str(i),
                title,
                f"{fix['impact_score']} patterns",
                fix['effort_estimate'],
                f"[{priority_style}]{fix['priority'].upper()}[/{priority_style}]"
            )
        
        output.append(table)
        
        # Quick wins
        if summary['quick_wins']:
            qw_text = "\n".join([
                f"  • {qw['title']} (fixes {qw['impact_score']} patterns)"
                for qw in summary['quick_wins']
            ])
            output.append(Panel(qw_text, title="⚡ Quick Wins - Start Here!", border_style="yellow"))
        
        return output

