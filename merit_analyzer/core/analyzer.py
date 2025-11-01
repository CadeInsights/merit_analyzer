"""Main MeritAnalyzer class - the public API for Merit Analyzer."""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

try:
    from rich.console import Console  # type: ignore
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn  # type: ignore
    from rich.status import Status  # type: ignore
    from rich.panel import Panel  # type: ignore
    from rich.table import Table  # type: ignore
    from rich import box  # type: ignore
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..models.test_result import TestResult, TestResultBatch
from ..models.report import AnalysisReport, ReportSummary, PatternSummary
from ..models.pattern import Pattern
from ..models.recommendation import Recommendation
from ..core.config import MeritConfig
from ..core.test_parser import TestParser
from ..core.universal_pattern_detector import UniversalPatternDetector
from ..core.pattern_merger import PatternMerger
from ..analysis.schema_discovery import SchemaDiscovery
from ..discovery.project_scanner import ProjectScanner
from ..discovery.framework_detector import FrameworkDetector
from ..discovery.code_mapper import CodeMapper
from ..analysis.claude_agent import MeritClaudeAgent
# NOTE: Removed hardcoded RootCauseAnalyzer - using LLM agent for intelligent analysis
from ..analysis.comparative import ComparativeAnalyzer
from ..recommendations.generator import RecommendationGenerator
from ..recommendations.prioritizer import RecommendationPrioritizer
from ..recommendations.formatter import RecommendationFormatter
from ..recommendations.consolidator import RecommendationConsolidator
from ..recommendations.executive_summary import ExecutiveSummaryGenerator


class MeritAnalyzer:
    """
    Main analyzer class - the public API for Merit Analyzer.
    
    This class orchestrates the entire analysis pipeline from test results
    to actionable recommendations.
    """

    def __init__(self, 
                 project_path: str,
                 api_key: str,
                 provider: str = "anthropic",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Merit Analyzer.

        Args:
            project_path: Path to the AI system codebase
            api_key: Anthropic API key or AWS credentials
            provider: "anthropic" or "bedrock"
            config: Optional configuration overrides
        """
        self.project_path = Path(project_path)
        
        # Create configuration
        config_dict = config or {}
        config_dict.update({
            "project_path": str(project_path),
            "api_key": api_key,
            "provider": provider
        })
        self.config = MeritConfig(**config_dict)
        
        # Initialize components
        self.test_parser = TestParser()
        self.claude_agent = MeritClaudeAgent(self.config)
        
        # Initialize schema discovery
        self.schema_discovery = SchemaDiscovery(self.claude_agent)
        
        # Initialize universal pattern detector
        self.pattern_detector = UniversalPatternDetector(
            min_cluster_size=self.config.min_cluster_size,
            similarity_threshold=self.config.similarity_threshold,
            claude_agent=self.claude_agent
        )
        
        # Initialize pattern merger
        self.pattern_merger = PatternMerger(similarity_threshold=0.7)
        
        self.project_scanner = ProjectScanner(project_path)
        self.framework_detector = FrameworkDetector()
        self.code_mapper = CodeMapper(project_path)
        # NOTE: No hardcoded root_cause_analyzer - using LLM agent for intelligent analysis
        self.comparative_analyzer = ComparativeAnalyzer()
        self.recommendation_generator = RecommendationGenerator(self.claude_agent)
        self.recommendation_prioritizer = RecommendationPrioritizer()
        self.recommendation_formatter = RecommendationFormatter()
        self.recommendation_consolidator = RecommendationConsolidator()
        self.executive_summary_generator = ExecutiveSummaryGenerator()
        
        # Cache for discovered architecture
        self._architecture_cache: Optional[Dict[str, Any]] = None
        self._scan_results_cache: Optional[Dict[str, Any]] = None
        
        # Cost tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def analyze_universal(self, test_results: Union[List[TestResult], List[Dict], TestResultBatch]) -> AnalysisReport:
        """
        Universal analysis method for ANY AI system.
        
        This is an alias for analyze() - both methods now use the same universal approach.
        Uses schema discovery, hierarchical clustering, and AI-powered failure analysis
        to work with any type of AI system without domain-specific assumptions.

        Args:
            test_results: List of TestResult objects, dicts, or TestResultBatch

        Returns:
            AnalysisReport with patterns, recommendations, and action plan
        """
        # Delegate to main analyze() method which uses parallel execution
        return self.analyze(test_results)

    def analyze(self, test_results: Union[List[TestResult], List[Dict], TestResultBatch]) -> AnalysisReport:
        """
        Main analysis method.

        Args:
            test_results: List of TestResult objects, dicts, or TestResultBatch

        Returns:
            AnalysisReport with patterns, recommendations, and action plan
        """
        start_time = time.time()
        step_start = start_time
        
        def log_step_time(step_name: str):
            """Helper to log elapsed time for a step."""
            nonlocal step_start
            elapsed = time.time() - step_start
            step_start = time.time()
            if RICH_AVAILABLE:
                console.print(f"[dim]   ⏱ {elapsed:.2f}s[/dim]")
            else:
                print(f"   ⏱ {elapsed:.2f}s")
        
        if RICH_AVAILABLE:
            console = Console()
            console.print(Panel.fit(
                "[bold blue]🚀 Merit Analyzer SDK[/bold blue]\n"
                "[dim]Analyzing AI system test failures and generating actionable recommendations[/dim]",
                border_style="blue"
            ))
        else:
            print("🔍 Merit Analyzer - Starting analysis...")
        
        try:
            # Step 1: Validate and parse test results
            if RICH_AVAILABLE:
                with Status("[bold cyan]Parsing test results...", console=console):
                    parsed_tests = self._parse_test_results(test_results)
                failures = sum(1 for t in parsed_tests if t.status == 'failed')
                console.print(f"[green]✓[/green] Found [bold]{len(parsed_tests)}[/bold] tests ([bold]{failures}[/bold] failures)")
                log_step_time("Parse test results")
            else:
                print("\n📋 Parsing test results...")
                parsed_tests = self._parse_test_results(test_results)
                print(f"   Found {len(parsed_tests)} tests ({sum(1 for t in parsed_tests if t.status == 'failed')} failures)")
            
            if not parsed_tests:
                raise ValueError("No test results provided")
            
            # Step 2: Quick project scan
            if RICH_AVAILABLE:
                console.print()  # Blank line for spacing
                with Status("[bold cyan]Scanning project structure...", console=console):
                    scan_results = self._get_or_scan_project()
                frameworks = list(scan_results.get('frameworks', {}).keys())
                console.print(f"[green]✓[/green] Found [bold]{scan_results['file_count']}[/bold] Python files", end="")
                if frameworks:
                    console.print(f" • Frameworks: [bold]{', '.join(frameworks)}[/bold]")
                else:
                    console.print()
                log_step_time("Project scan")
            else:
                print("\n🗂️  Scanning project structure...")
                scan_results = self._get_or_scan_project()
                print(f"   Found {scan_results['file_count']} Python files")
                print(f"   Detected frameworks: {', '.join(scan_results.get('frameworks', {}).keys()) or 'None'}")
            
            # Step 3: Discover system schema
            if RICH_AVAILABLE:
                console.print()  # Blank line for spacing
                with Status("[bold cyan]Discovering system schema...", console=console):
                    schema_info = self.schema_discovery.discover_system_schema(parsed_tests)
                
                # Track tokens from schema discovery
                if '_token_usage' in schema_info:
                    self._total_input_tokens += schema_info['_token_usage'].get('input_tokens', 0)
                    self._total_output_tokens += schema_info['_token_usage'].get('output_tokens', 0)
                
                system_type = schema_info.get('system_type', 'unknown')
                if system_type != 'unknown':
                    # Format system type nicely (RAG stays uppercase, others are title case)
                    display_type = system_type.replace('_', ' ').title()
                    if system_type == 'rag':
                        display_type = 'RAG'
                    console.print(f"[green]✓[/green] System schema discovered ([bold]{display_type}[/bold])")
                else:
                    console.print(f"[green]✓[/green] System schema discovered")
                log_step_time("Schema discovery")
            else:
                print("\n🧠 Discovering system schema...")
                schema_info = self.schema_discovery.discover_system_schema(parsed_tests)
                
                # Track tokens from schema discovery
                if '_token_usage' in schema_info:
                    self._total_input_tokens += schema_info['_token_usage'].get('input_tokens', 0)
                    self._total_output_tokens += schema_info['_token_usage'].get('output_tokens', 0)
                
                system_type = schema_info.get('system_type', 'unknown')
                if system_type != 'unknown':
                    # Format system type nicely (RAG stays uppercase, others are title case)
                    display_type = system_type.replace('_', ' ').title()
                    if system_type == 'rag':
                        display_type = 'RAG'
                    print(f"   System schema discovered ({display_type})")
                else:
                    print(f"   System schema discovered")

            # Step 4: Detect failure patterns
            if RICH_AVAILABLE:
                console.print()  # Blank line for spacing
                with Status("[bold cyan]Detecting failure patterns...", console=console):
                    raw_patterns = self.pattern_detector.detect_patterns(parsed_tests)
                    if raw_patterns:
                        patterns = self.pattern_merger.merge_similar_patterns(raw_patterns)
                        # Cap patterns for large datasets to control cost
                        # Analyze top N patterns by failure count
                        # Dynamic cap: scale with dataset size (min 20, max 50)
                        total_failures = sum(len(tests) for tests in patterns.values())
                        if total_failures > 500:
                            max_patterns = min(50, max(20, len(patterns) // 2))
                        else:
                            max_patterns = 50
                        
                        if len(patterns) > max_patterns:
                            sorted_patterns = sorted(patterns.items(), 
                                                    key=lambda x: len(x[1]), 
                                                    reverse=True)
                            patterns = dict(sorted_patterns[:max_patterns])
                            console.print(f"[dim]Analyzing top {max_patterns} patterns by failure count ({len(raw_patterns)} total detected) to control costs[/dim]")
                    else:
                        patterns = {}
                
                if patterns:
                    if len(patterns) < len(raw_patterns):
                        merged_count = len(raw_patterns) - len(patterns)
                        console.print(f"[green]✓[/green] Identified [bold]{len(raw_patterns)}[/bold] patterns, merged to [bold]{len(patterns)}[/bold] ([dim]{merged_count} merged[/dim])")
                    else:
                        console.print(f"[green]✓[/green] Identified [bold]{len(patterns)}[/bold] patterns")
                    # Show top patterns in a cleaner format
                    pattern_display = ", ".join([f"[dim]{name}[/dim] ({len(tests)})" for name, tests in list(patterns.items())[:5]])
                    if len(patterns) > 5:
                        pattern_display += f" [dim]... and {len(patterns) - 5} more[/dim]"
                    console.print(f"   {pattern_display}")
                    log_step_time("Pattern detection")
                else:
                    console.print("[yellow]⚠[/yellow] No failure patterns detected")
                    log_step_time("Pattern detection")
            else:
                print("\n🔎 Detecting failure patterns...")
                raw_patterns = self.pattern_detector.detect_patterns(parsed_tests)
                if raw_patterns:
                    patterns = self.pattern_merger.merge_similar_patterns(raw_patterns)
                    if len(patterns) < len(raw_patterns):
                        merged_count = len(raw_patterns) - len(patterns)
                        print(f"   Identified {len(raw_patterns)} patterns, merged to {len(patterns)} ({merged_count} similar patterns merged)")
                    else:
                        print(f"   Identified {len(patterns)} patterns:")
                    for pattern_name, tests in list(patterns.items())[:10]:
                        print(f"      - {pattern_name}: {len(tests)} failures")
                else:
                    patterns = {}
            
            if not patterns:
                if RICH_AVAILABLE:
                    console.print("\n[bold yellow]⚠️ No failure patterns detected[/bold yellow]")
                else:
                    print("   ⚠️  No failure patterns detected")
                report = self._create_empty_report(parsed_tests, scan_results)
                return report
            
            # Step 5-10: Main analysis with status updates
            
            # Initialize variables (needed for both Rich and non-Rich paths)
            all_recommendations = []
            pattern_summaries = {}
            passing_tests = [t for t in parsed_tests if t.status == "passed"]
            pattern_mappings = {}
            
            if RICH_AVAILABLE:
                console.print()  # Blank line for spacing
                console.print("[bold green]🔍 Starting Deep Analysis...[/bold green]")
                console.print()  # Blank line for spacing
                
                # Step 5: Discover system architecture
                with Status("[bold cyan]Analyzing system architecture...", console=console):
                    architecture = self.claude_agent.discover_system_architecture(scan_results)
                
                # Track tokens from architecture discovery
                if '_token_usage' in architecture:
                    self._total_input_tokens += architecture['_token_usage'].get('input_tokens', 0)
                    self._total_output_tokens += architecture['_token_usage'].get('output_tokens', 0)
                
                console.print(f"[green]✓[/green] System architecture analyzed")
                log_step_time("Architecture analysis")
                
                # Step 6: Analyze patterns (PARALLEL with OPTIMIZED prompts)
                console.print()  # Blank line for spacing
                all_recommendations = []
                pattern_summaries = {}
                passing_tests = [t for t in parsed_tests if t.status == "passed"]
                
                # Parallel execution with 60s timeout per pattern
                max_workers = min(self.config.max_workers, len(patterns), 15)
                
                with Progress(
                    SpinnerColumn(), 
                    TextColumn("[progress.description]{task.description}"), 
                    BarColumn(),
                    TextColumn("[bold cyan]{task.completed}/{task.total}"),
                    TextColumn("[dim]patterns[/dim]"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        "[bold cyan]Analyzing patterns...", 
                        total=len(patterns)
                    )
                    
                    # ThreadPoolExecutor INSIDE Progress context
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_pattern = {
                            executor.submit(
                                self._analyze_single_pattern,
                                pattern_name,
                                failing_tests,
                                passing_tests,
                                architecture
                            ): pattern_name
                            for pattern_name, failing_tests in patterns.items()
                        }
                        
                        for future in as_completed(future_to_pattern):
                            pattern_name = future_to_pattern[future]
                            try:
                                result = future.result(timeout=60)  # 60s timeout
                                if result:
                                    pattern, recommendations, summary = result
                                    all_recommendations.extend(recommendations)
                                    pattern_summaries[pattern_name] = summary
                                    progress.update(task, advance=1)
                                else:
                                    progress.update(task, advance=1)
                            except TimeoutError:
                                console.print(f"\n[yellow]⏱[/yellow] Timeout analyzing '[dim]{pattern_name}[/dim]'")
                                progress.update(task, advance=1)
                            except Exception as e:
                                console.print(f"\n[yellow]⚠[/yellow] Error analyzing '[dim]{pattern_name}[/dim]': {e}")
                                progress.update(task, advance=1)
                
                console.print(f"[green]✓[/green] Completed analysis of [bold]{len(pattern_summaries)}/{len(patterns)}[/bold] patterns")
                log_step_time("Pattern analysis")
                
                # Step 8: Early deduplication before prioritization
                console.print()  # Blank line for spacing
                with Status("[bold cyan]Deduplicating recommendations...", console=console):
                    all_recommendations = self._early_deduplicate_recommendations(all_recommendations)
                console.print(f"[green]✓[/green] Deduplicated to [bold]{len(all_recommendations)}[/bold] unique recommendations")
                log_step_time("Deduplication")
                
                # Step 9: Prioritize recommendations
                console.print()  # Blank line for spacing
                with Status("[bold cyan]Prioritizing recommendations...", console=console):
                    prioritized_recommendations = self.recommendation_prioritizer.prioritize_recommendations(
                        all_recommendations, pattern_summaries
                    )
                console.print(f"[green]✓[/green] Prioritized [bold]{len(prioritized_recommendations)}[/bold] recommendations")
                log_step_time("Prioritization")
                
                # Step 10: Generate report
                console.print()  # Blank line for spacing
                with Status("[bold cyan]Generating analysis report...", console=console):
                    report = self._generate_report(
                        parsed_tests,
                        pattern_summaries,
                        prioritized_recommendations,
                        architecture,
                        scan_results,
                        time.time() - start_time
                    )
                console.print(f"[green]✓[/green] Report generated")
                log_step_time("Report generation")
                
                # Store report for return
                final_report = report
                
                # Show total time
                total_time = time.time() - start_time
                console.print()  # Blank line
                console.print(f"[bold green]✅ Analysis complete![/bold green] Total time: [bold]{total_time:.1f}s[/bold]")
                
                # Show cost information
                total_tokens = self._total_input_tokens + self._total_output_tokens
                estimated_cost = self._calculate_estimated_cost()
                console.print()
                console.print(f"💰 [bold cyan]Cost Tracking:[/bold cyan]")
                console.print(f"   Input tokens: [bold]{self._total_input_tokens:,}[/bold]")
                console.print(f"   Output tokens: [bold]{self._total_output_tokens:,}[/bold]")
                console.print(f"   Total tokens: [bold]{total_tokens:,}[/bold]")
                console.print(f"   Estimated cost: [bold green]${estimated_cost:.4f}[/bold green]")
                
            # Non-Rich path (fallback)
            else:
                # Fallback to simple print statements
                print("\n🏗️  Analyzing system architecture...")
                architecture = self.claude_agent.discover_system_architecture(scan_results)
                
                # Track tokens from architecture discovery
                if '_token_usage' in architecture:
                    self._total_input_tokens += architecture['_token_usage'].get('input_tokens', 0)
                    self._total_output_tokens += architecture['_token_usage'].get('output_tokens', 0)
                
                print("   Architecture mapped successfully")
                
                print("\n🧠 Analyzing patterns (Claude Agent SDK finds code + analyzes)...")
                all_recommendations = []
                pattern_summaries = {}
                passing_tests = [t for t in parsed_tests if t.status == "passed"]
                
                # Analyze patterns in PARALLEL with optimized prompts (COST EFFICIENT + FAST)
                print(f"Analyzing {len(patterns)} patterns...")
                
                max_workers = min(self.config.max_workers, len(patterns), 15)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_pattern = {
                        executor.submit(
                            self._analyze_single_pattern,
                            pattern_name,
                            failing_tests,
                            passing_tests,
                            architecture
                        ): pattern_name
                        for pattern_name, failing_tests in patterns.items()
                    }
                    
                    for i, future in enumerate(as_completed(future_to_pattern), 1):
                        pattern_name = future_to_pattern[future]
                        try:
                            # Add 60s timeout per pattern to prevent hanging
                            result = future.result(timeout=60)
                            if result:
                                pattern, recommendations, summary = result
                                all_recommendations.extend(recommendations)
                                pattern_summaries[pattern_name] = summary
                                print(f"   ✅ Analyzed {i}/{len(patterns)}: {pattern_name}")
                            else:
                                print(f"   ⚠️  No results for {pattern_name}")
                        except TimeoutError:
                            print(f"   ⏱️  Timeout (60s) analyzing '{pattern_name}' - skipping")
                        except Exception as e:
                            print(f"   ⚠️  Error analyzing pattern '{pattern_name}': {e}")
                
                print(f"   ✅ Analyzed {len(pattern_summaries)}/{len(patterns)} patterns")
                
                # Early deduplication before prioritization
                print(f"\n🔄 Deduplicating {len(all_recommendations)} recommendations...")
                all_recommendations = self._early_deduplicate_recommendations(all_recommendations)
                print(f"   ✅ Removed duplicates: {len(all_recommendations)} unique recommendations")
                
                # Prioritize recommendations
                print("\n📊 Prioritizing recommendations...")
                prioritized_recommendations = self.recommendation_prioritizer.prioritize_recommendations(
                    all_recommendations, pattern_summaries
                )
                print(f"   Prioritized {len(prioritized_recommendations)} recommendations")
                
                # Generate report
                print("\n📊 Generating analysis report...")
                final_report = self._generate_report(
                    parsed_tests,
                    pattern_summaries,
                    prioritized_recommendations,
                    architecture,
                    scan_results,
                    time.time() - start_time
                )
                print(f"   Report generated")
            
            # Display results
            if RICH_AVAILABLE:
                self._display_rich_results(final_report, console)
            else:
                print("\n✅ Analysis complete!")
            
            return final_report
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"\n[red]❌ Analysis failed: {e}[/red]")
            else:
                print(f"\n❌ Analysis failed: {e}")
            raise

    def _parse_test_results(self, test_results: Union[List[TestResult], List[Dict], TestResultBatch]) -> List[TestResult]:
        """Parse and validate test results."""
        if isinstance(test_results, TestResultBatch):
            return test_results.results
        elif isinstance(test_results, list):
            if all(isinstance(t, TestResult) for t in test_results):
                return test_results
            elif all(isinstance(t, dict) for t in test_results):
                return [TestResult(**t) for t in test_results]
            else:
                raise ValueError("Mixed types in test results list")
        else:
            raise ValueError(f"Invalid test results type: {type(test_results)}")

    def _get_or_scan_project(self) -> Dict[str, Any]:
        """Get cached scan results or perform new scan."""
        if self._scan_results_cache is None:
            # Suppress verbose output - we'll show it with Rich formatting
            scan_results = self.project_scanner.scan(verbose=False)
            frameworks = self.framework_detector.detect(self.project_scanner.python_files, verbose=False)
            scan_results['frameworks'] = frameworks
            self._scan_results_cache = scan_results
        return self._scan_results_cache
    
    def _get_source_files_to_analyze(self) -> List[str]:
        """Get source files to analyze, excluding test files."""
        if not self._scan_results_cache:
            return []
        
        python_files = self._scan_results_cache.get('python_files', [])
        
        # Filter out test files
        source_files = []
        for file_path in python_files:
            file_name = Path(file_path).name.lower()
            # Skip test files
            if (file_name.startswith('test_') or 
                '/test/' in file_path or 
                '/tests/' in file_path or
                '/testing/' in file_path or
                file_name == 'conftest.py'):
                continue
            source_files.append(file_path)
        
        return source_files
    
    def _calculate_estimated_cost(self) -> float:
        """
        Calculate estimated cost based on Claude Sonnet 4 pricing.
        
        Pricing (as of Oct 2024):
        - Input: $3 per million tokens
        - Output: $15 per million tokens
        """
        input_cost = (self._total_input_tokens / 1_000_000) * 3.0
        output_cost = (self._total_output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost
    
    def _calculate_cost(self) -> float:
        """Calculate the total cost of all LLM calls."""
        input_cost = (self._total_input_tokens / 1_000_000) * 3.0
        output_cost = (self._total_output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost

    def _get_or_discover_architecture(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get cached architecture or discover it."""
        if self._architecture_cache is None:
            self._architecture_cache = self.claude_agent.discover_system_architecture(scan_results)
        return self._architecture_cache

    def _find_similar_passing_tests(self, 
                                   failing_tests: List[TestResult],
                                   passing_tests: List[TestResult]) -> List[TestResult]:
        """Find passing tests similar to failing ones for comparison."""
        if not passing_tests:
            return []
        
        # Use simple keyword matching for now
        failing_keywords = set()
        for test in failing_tests:
            input_text = self._extract_input_text(test)
            words = input_text.lower().split()
            failing_keywords.update(words)
        
        similar = []
        for test in passing_tests:
            input_text = self._extract_input_text(test)
            test_words = set(input_text.lower().split())
            overlap = len(failing_keywords & test_words)
            if overlap >= 2:  # At least 2 keywords in common
                similar.append(test)
        
        return similar[:5]  # Return up to 5 similar passing tests

    def _extract_input_text(self, test: TestResult) -> str:
        """Extract text from test input for keyword matching."""
        if isinstance(test.input, str):
            return test.input
        elif isinstance(test.input, dict):
            # Convert dict to string representation
            return json.dumps(test.input, sort_keys=True)
        else:
            return str(test.input)

    def _analyze_single_pattern(self, pattern_name: str, failing_tests: List[TestResult],
                                passing_tests: List[TestResult],
                                architecture: Dict[str, Any]) -> Optional[tuple]:
        """
        Analyze a single pattern (for parallel execution).
        
        Claude Agent SDK uses Grep/Glob to find relevant files, then analyzes them.
        
        Returns:
            Tuple of (Pattern, List[Recommendation], PatternSummary) or None
        """
        try:
            # Get similar passing tests for comparison
            similar_passing = self._find_similar_passing_tests(failing_tests, passing_tests)
            
            # Create pattern object
            pattern = Pattern(
                name=pattern_name,
                test_results=failing_tests,
                confidence=0.8,
                keywords=self._extract_pattern_keywords(failing_tests)
            )
            
            # Get source files to analyze (exclude test files)
            source_files = self._get_source_files_to_analyze()
            
            # For small codebases, read files upfront (saves Agent SDK overhead)
            code_content = None
            if len(source_files) < 3:
                try:
                    code_parts = []
                    for file_path in source_files:
                        full_path = self.project_path / file_path
                        if full_path.exists():
                            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                code_parts.append(f"=== {file_path} ===\n{content}\n")
                    code_content = "\n\n".join(code_parts)
                except Exception as e:
                    print(f"   ⚠️  Error reading files: {e}")
            
            # Analyze pattern
            analysis = self.claude_agent.analyze_pattern(
                pattern_name=pattern_name,
                failing_tests=failing_tests,
                passing_tests=similar_passing,
                source_files=source_files,
                code_content=code_content  # Pre-read code for small codebases
            )
            
            # Track costs
            if '_cost_info' in analysis:
                cost_info = analysis['_cost_info']
                self._total_input_tokens += cost_info.get('input_tokens', 0)
                self._total_output_tokens += cost_info.get('output_tokens', 0)
            
            # Use LLM's root cause (not hardcoded heuristics)
            # The LLM analysis already includes root_cause, pattern_characteristics, and code_issues
            llm_root_cause = analysis.get("root_cause", "Unable to determine root cause")
            
            # COST OPTIMIZATION: Use recommendations from analyze_pattern if available
            # The analyze_pattern prompt already includes recommendations, saving 1 LLM call per pattern!
            llm_recommendations = analysis.get("recommendations", [])
            
            if llm_recommendations:
                # Use recommendations from analyze_pattern (saves ~30-50% cost per pattern)
                from ..models.recommendation import Recommendation, PriorityLevel, RecommendationType
                import re
                
                # Extract location from root_cause if available (e.g., "file.py:123" or "file.py:123-456")
                location_from_root_cause = ""
                if llm_root_cause:
                    # Try to extract file:line pattern from root cause
                    match = re.search(r'([a-zA-Z0-9_/\-\.]+\.py:\d+(?:-\d+)?)', llm_root_cause)
                    if match:
                        location_from_root_cause = match.group(1)
                
                recommendations = []
                for rec_data in llm_recommendations[:3]:  # Limit to 3 per pattern
                    try:
                        # Helper parsing maps
                        priority_map = {"high": PriorityLevel.HIGH, "medium": PriorityLevel.MEDIUM, "low": PriorityLevel.LOW}
                        type_map = {"code": RecommendationType.CODE, "prompt": RecommendationType.PROMPT, 
                                   "architecture": RecommendationType.ARCHITECTURE, "configuration": RecommendationType.CONFIGURATION,
                                   "testing": RecommendationType.TESTING}
                        
                        # Use location from recommendation, fallback to root_cause location
                        rec_location = rec_data.get("location", "")
                        if not rec_location:
                            rec_location = location_from_root_cause
                        
                        recommendation = Recommendation(
                            priority=priority_map.get(rec_data.get("priority", "medium").lower(), PriorityLevel.MEDIUM),
                            type=type_map.get(rec_data.get("type", "code").lower(), RecommendationType.CODE),
                            title=rec_data.get("title", "Fix issue"),
                            description=rec_data.get("description", ""),
                            location=rec_location,
                            implementation=rec_data.get("implementation") or rec_data.get("code_diff") or None,
                            expected_impact=rec_data.get("expected_impact") or None,
                            effort_estimate=rec_data.get("effort_estimate", rec_data.get("effort", "Unknown")),
                            rationale=f"Based on code analysis of {pattern_name}",
                            code_diff=rec_data.get("code_diff"),
                            tags=[pattern_name]
                        )
                        recommendations.append(recommendation)
                    except Exception:
                        continue
            else:
                # Fallback: Generate recommendations separately (only if analyze_pattern didn't provide them)
                # Note: Pattern analysis should always provide recommendations now via Claude Agent SDK
                # This is just a safety fallback
                code_context = {}  # Claude Agent SDK finds and reads its own files
                recommendations = self.recommendation_generator.generate_recommendations(
                    pattern=pattern,
                    root_cause=llm_root_cause,
                    code_context=code_context,
                    architecture=architecture
                )
                # Limit to 2 recommendations per pattern for cost efficiency
                recommendations = recommendations[:2]
            
            # Create pattern summary
            # Use LLM's analysis results (not hardcoded root cause analyzer)
            summary = PatternSummary(
                name=pattern_name,
                failure_count=len(failing_tests),
                failure_rate=len(failing_tests) / (len(failing_tests) + len(passing_tests)) if passing_tests else 1.0,
                example_tests=failing_tests[:3],
                recommendations=recommendations,
                root_cause=llm_root_cause,  # From LLM code analysis, not hardcoded
                confidence=0.8,  # Default confidence (LLM analysis is reliable)
                keywords=pattern.keywords
            )
            
            return (pattern, recommendations, summary)
            
        except Exception as e:
            print(f"Error analyzing pattern '{pattern_name}': {e}")
            return None

    def _early_deduplicate_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """
        Early deduplication of recommendations based on title and description similarity.
        
        This happens before prioritization to reduce processing overhead.
        """
        if not recommendations:
            return []
        
        # Simple deduplication based on title similarity
        seen_titles = set()
        unique_recommendations = []
        
        for rec in recommendations:
            # Normalize title for comparison
            title_key = rec.title.lower().strip()
            
            # Check if we've seen a similar title
            is_duplicate = False
            for seen_title in seen_titles:
                # Simple similarity check (title contains or is contained in seen)
                if title_key in seen_title or seen_title in title_key:
                    # Check description similarity for confirmation
                    if rec.description:
                        desc_key = rec.description.lower()[:100]  # First 100 chars
                        if any(desc_key in seen_desc for seen_desc in seen_titles):
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                unique_recommendations.append(rec)
                seen_titles.add(title_key)
        
        return unique_recommendations

    def _extract_pattern_keywords(self, tests: List[TestResult]) -> List[str]:
        """Extract keywords from a pattern's tests."""
        keywords = set()
        for test in tests:
            if test.input:
                input_text = self._extract_input_text(test)
                words = input_text.lower().split()
                keywords.update(words)
            if test.failure_reason:
                words = test.failure_reason.lower().split()
                keywords.update(words)
        
        # Filter out common words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her',
                     'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its',
                     'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man',
                     'men', 'put', 'say', 'she', 'too', 'use', 'test', 'tests', 'testing', 'result',
                     'results', 'input', 'output', 'expected', 'actual', 'failure', 'failed',
                     'error', 'exception', 'message', 'string', 'value', 'values', 'data'}
        
        return [word for word in keywords if word not in stop_words][:10]

    def _get_code_context(self, file_paths: List[str]) -> Dict[str, str]:
        """Get code context for the given file paths."""
        context = {}
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Truncate very long files
                    if len(content) > 10000:
                        content = content[:10000] + "\n... (truncated)"
                    context[file_path] = content
            except Exception as e:
                print(f"    ⚠️  Error reading {file_path}: {e}")
                context[file_path] = f"Error reading file: {e}"
        return context

    def _generate_report(self, 
                        test_results: List[TestResult],
                        pattern_summaries: Dict[str, PatternSummary],
                        recommendations: List[Recommendation],
                        architecture: Dict[str, Any],
                        scan_results: Dict[str, Any],
                        duration: float) -> AnalysisReport:
        """Generate final analysis report."""
        
        # Calculate summary statistics
        total_tests = len(test_results)
        passed = sum(1 for t in test_results if t.status == "passed")
        failed = sum(1 for t in test_results if t.status == "failed")
        error = sum(1 for t in test_results if t.status == "error")
        skipped = sum(1 for t in test_results if t.status == "skipped")
        
        # Consolidate similar recommendations
        consolidated_recs, impact_map = self.recommendation_consolidator.consolidate_recommendations(
            recommendations,
            pattern_summaries
        )
        
        # Generate executive summary
        summary_stats = {
            'total_tests': total_tests,
            'passed': passed,
            'failed': failed,
            'error': error,
            'skipped': skipped,
            'analysis_duration_seconds': duration
        }
        executive_summary = self.executive_summary_generator.generate_summary(
            consolidated_recs,
            pattern_summaries,
            summary_stats
        )
        
        # Generate prioritized action plan from consolidated recommendations
        action_plan = self._generate_action_plan_from_consolidated(
            consolidated_recs, 
            executive_summary
        )
        
        # Create report summary
        summary = ReportSummary(
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            error=error,
            skipped=skipped,
            patterns_found=len(pattern_summaries),
            analysis_timestamp=datetime.now().isoformat(),
            analysis_duration_seconds=duration,
            project_path=str(self.project_path),
            frameworks_detected=list(scan_results.get('frameworks', {}).keys())
        )
        
        return AnalysisReport(
            summary=summary,
            patterns=pattern_summaries,
            action_plan=action_plan,
            architecture=architecture,
            recommendations=recommendations,
            executive_summary=executive_summary,
            consolidated_recommendations=consolidated_recs,
            metadata={
                "analyzer_version": "1.0.0",
                "config": self.config.model_dump(),
                "scan_results": scan_results,
                "cost_tracking": {
                    "input_tokens": self._total_input_tokens,
                    "output_tokens": self._total_output_tokens,
                    "estimated_cost": self._calculate_cost()
                }
            }
        )

    def _generate_action_plan_from_consolidated(
        self,
        consolidated_recs: List[Dict[str, Any]],
        executive_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate prioritized action plan from consolidated recommendations."""
        plan = []
        
        # Add implementation phases
        for phase in executive_summary.get('implementation_order', []):
            plan.append(f"Phase {phase['phase']}: {phase['name']}")
            plan.append(f"  {phase['description']}")
            plan.append(f"  Expected Impact: {phase['impact']}")
            plan.append("")
        
        # Add top fixes
        plan.append("Top Priority Fixes:")
        for i, rec in enumerate(executive_summary.get('top_fixes', [])[:5], 1):
            plan.append(
                f"{i}. [{rec['effort_estimate']}] {rec['title']} "
                f"→ fixes {rec['impact_score']} patterns"
            )
        
        return plan
    
    def _generate_action_plan(self, 
                             pattern_summaries: Dict[str, PatternSummary],
                             recommendations: List[Recommendation]) -> List[str]:
        """Generate prioritized action plan."""
        # Sort patterns by impact (failure count)
        sorted_patterns = sorted(
            pattern_summaries.items(),
            key=lambda x: x[1].failure_count,
            reverse=True
        )
        
        plan = []
        
        # Add pattern-based actions
        for i, (pattern_name, pattern) in enumerate(sorted_patterns, 1):
            if pattern.recommendations:
                top_rec = pattern.recommendations[0]  # Get top recommendation
                plan.append(
                    f"{i}. [{top_rec.effort_estimate}] {top_rec.title} → "
                    f"fixes {pattern.failure_count} tests in {pattern_name}"
                )
        
        # Add quick wins
        quick_wins = self.recommendation_prioritizer.get_quick_wins(recommendations)
        if quick_wins:
            plan.append(f"\nQuick Wins ({len(quick_wins)} recommendations):")
            for i, rec in enumerate(quick_wins[:3], 1):  # Top 3 quick wins
                plan.append(f"  - {rec.title} ({rec.effort_estimate})")
        
        return plan

    def _create_empty_report(self, 
                           test_results: List[TestResult],
                           scan_results: Dict[str, Any]) -> AnalysisReport:
        """Create an empty report when no patterns are detected."""
        total_tests = len(test_results)
        passed = sum(1 for t in test_results if t.status == "passed")
        failed = sum(1 for t in test_results if t.status == "failed")
        error = sum(1 for t in test_results if t.status == "error")
        skipped = sum(1 for t in test_results if t.status == "skipped")
        
        summary = ReportSummary(
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            error=error,
            skipped=skipped,
            patterns_found=0,
            analysis_timestamp=datetime.now().isoformat(),
            project_path=str(self.project_path),
            frameworks_detected=list(scan_results.get('frameworks', {}).keys())
        )
        
        return AnalysisReport(
            summary=summary,
            patterns={},
            action_plan=["No failure patterns detected - all tests are passing or have unique failures"],
            architecture={},
            recommendations=[],
            metadata={
                "analyzer_version": "1.0.0",
                "config": self.config.model_dump(),
                "scan_results": scan_results
            }
        )

    def save_report(self, report: AnalysisReport, output_path: str = "merit_analysis_output/merit_analysis_report.json"):
        """Save report to file."""
        # Create output directory if it doesn't exist
        from pathlib import Path
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)
        
        report.save(output_path, "json")
        
        # Also save markdown version
        md_path = output_path.replace('.json', '.md')
        report.save(md_path, "markdown")
        
        # Display highlighted report paths
        if RICH_AVAILABLE:
            from rich.console import Console  # type: ignore
            from rich.panel import Panel  # type: ignore
            console = Console()
            report_text = (
                f"[bold cyan]📄 Full recommendations available:[/bold cyan]\n\n"
                f"  • JSON: [bold green]{output_path}[/bold green]\n"
                f"  • Markdown: [bold green]{md_path}[/bold green]"
            )
            console.print("\n")
            console.print(Panel(report_text, border_style="green", title="📊 Reports Generated"))
        else:
            print(f"\n📄 Report saved to {output_path}")
            print(f"📄 Markdown report saved to {md_path}")

    def export_recommendations(self, 
                             report: AnalysisReport,
                             output_path: str,
                             format_type: str = "markdown"):
        """Export recommendations to a file."""
        self.recommendation_formatter.export_to_file(
            report.recommendations,
            output_path,
            format_type,
            report.patterns
        )

    def get_quick_wins(self, report: AnalysisReport) -> List[Recommendation]:
        """Get quick win recommendations from the report."""
        return self.recommendation_prioritizer.get_quick_wins(report.recommendations)

    def create_implementation_plan(self, report: AnalysisReport) -> List[Dict[str, Any]]:
        """Create an implementation plan from the report."""
        return self.recommendation_prioritizer.create_implementation_plan(report.recommendations)

    def clear_cache(self):
        """Clear all caches."""
        self._architecture_cache = None
        self._scan_results_cache = None
        self.claude_agent.clear_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        claude_stats = self.claude_agent.get_cache_stats()
        return {
            "claude_cache": claude_stats,
            "architecture_cached": self._architecture_cache is not None,
            "scan_results_cached": self._scan_results_cache is not None
        }

    def _display_rich_results(self, report: AnalysisReport, console: Console):
        """Display results using Rich formatting."""
        
        # Display executive summary if available
        if report.executive_summary:
            exec_summary_panels = self.executive_summary_generator.format_executive_summary_cli(
                report.executive_summary
            )
            console.print("\n")
            for panel in exec_summary_panels:
                console.print(panel)
                console.print("")

    def _generate_failure_contexts(self, patterns: Dict[str, List[TestResult]], discovered_schema: Dict[str, Any]):
        """Generate AI-powered failure contexts for patterns without failure reasons."""
        for pattern_name, tests in patterns.items():
            # Check if any tests in this pattern lack failure reasons
            tests_needing_context = [t for t in tests if not t.failure_reason]
            
            if tests_needing_context and self.claude_agent:
                try:
                    # Generate failure context for this pattern
                    context = self._generate_pattern_failure_context(pattern_name, tests, discovered_schema)
                    
                    # Apply context to tests that need it
                    for test in tests_needing_context:
                        test.failure_reason = context
                        
                except Exception as e:
                    print(f"Warning: Failed to generate failure context for pattern {pattern_name}: {e}")

    def _generate_pattern_failure_context(self, pattern_name: str, tests: List[TestResult], discovered_schema: Dict[str, Any]) -> str:
        """Generate failure context for a specific pattern using AI."""
        if not self.claude_agent:
            return "AI analysis unavailable"
        
        # Create pattern summary
        pattern_summary = self._create_pattern_summary_for_ai(tests)
        
        prompt = f"""
Analyze this failure pattern and generate a concise failure reason that explains why these tests are failing.

Pattern: {pattern_name}
System Type: {discovered_schema.get('system_type', 'Unknown')}
Number of failing tests: {len(tests)}

Pattern Summary:
{pattern_summary}

Generate a specific, actionable failure reason that explains the root cause of these failures.
Examples:
- "Missing required output fields: greeting, signoff"
- "Output format mismatch: expected object, got string"
- "Validation failure: missing data validation"
- "Incomplete response: response too short"

Provide only the failure reason, nothing else.
"""

        response = self.claude_agent._call_claude(prompt)
        return response.strip()

    def _create_pattern_summary_for_ai(self, tests: List[TestResult]) -> str:
        """Create a summary of the pattern for AI analysis."""
        summary_parts = []
        
        # Input analysis
        input_types = set()
        for test in tests:
            if isinstance(test.input, dict):
                input_types.update(test.input.keys())
            else:
                input_types.add(type(test.input).__name__)
        summary_parts.append(f"Input types: {', '.join(input_types)}")
        
        # Output analysis
        output_issues = []
        for test in tests:
            if test.expected_output and test.actual_output:
                if isinstance(test.expected_output, dict) and isinstance(test.actual_output, dict):
                    missing_keys = set(test.expected_output.keys()) - set(test.actual_output.keys())
                    if missing_keys:
                        output_issues.append(f"missing_keys: {list(missing_keys)[:3]}")
                elif isinstance(test.expected_output, str) and isinstance(test.actual_output, str):
                    if test.expected_output.lower() not in test.actual_output.lower():
                        output_issues.append("content_mismatch")
        
        if output_issues:
            summary_parts.append(f"Output issues: {'; '.join(set(output_issues))}")
        
        return "\n".join(summary_parts)

    def _analyze_pattern_universal(self, pattern_name: str, failing_tests: List[TestResult], 
                                 all_tests: List[TestResult], discovered_schema: Dict[str, Any], 
                                 architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single pattern using universal approach."""
        # Find similar passing tests
        passing_tests = self._find_similar_passing_tests(failing_tests, 
                                                        [t for t in all_tests if t.status == "passed"])
        
        # Create pattern object
        pattern = Pattern(
            name=pattern_name,
            test_results=failing_tests,
            confidence=0.8,
            keywords=self._extract_pattern_keywords(failing_tests)
        )
        
        # Map pattern to code locations
        code_locations = self.claude_agent.map_pattern_to_code(
            pattern_name, failing_tests, architecture
        )
        
        # Analyze pattern with LLM agent (reads code files, provides intelligent analysis)
        analysis = self.claude_agent.analyze_pattern(
            pattern_name=pattern_name,
            failing_tests=failing_tests,
            passing_tests=passing_tests,
            code_locations=code_locations  # Claude reads these files
        )
        
        # Use LLM's intelligent analysis (NO hardcoded root cause analyzer)
        llm_root_cause = analysis.get("root_cause", "Unable to determine root cause")
        llm_system_type = analysis.get("system_type", architecture.get("system_type", "unknown"))
        
        # Update architecture with LLM-discovered system type
        if llm_system_type != "unknown":
            architecture["system_type"] = llm_system_type
        
        # Get code context for recommendations
        code_context = self._get_code_context(code_locations)
        
        # Generate recommendations using LLM's root cause
        recommendations = self.recommendation_generator.generate_recommendations(
            pattern=pattern,
            root_cause=llm_root_cause,  # From LLM code analysis, not hardcoded
            code_context=code_context,
            architecture=architecture
        )
        
        # Create pattern summary using LLM analysis
        pattern_summary = PatternSummary(
            name=pattern_name,
            failure_count=len(failing_tests),
            failure_rate=len(failing_tests) / len(all_tests),
            example_tests=failing_tests[:3],
            recommendations=recommendations,
            root_cause=llm_root_cause,  # From LLM, not hardcoded
            confidence=0.8,  # LLM analysis is reliable
            keywords=pattern.keywords
        )
        
        return {
            'summary': pattern_summary,
            'recommendations': recommendations
        }

    def _generate_universal_recommendations(self, pattern: Pattern, root_cause_analysis: Dict[str, Any],
                                          code_context: Dict[str, str], discovered_schema: Dict[str, Any],
                                          architecture: Dict[str, Any]) -> List[Recommendation]:
        """Generate universal recommendations for any AI system type."""
        recommendations = []
        
        # Generate system-agnostic recommendations
        system_type = discovered_schema.get('system_type', 'Unknown')
        
        # Code-level fixes
        if root_cause_analysis.get("root_cause") == "code_issue":
            recommendations.append(Recommendation(
                title="Fix Code Logic",
                description=f"Address the code issue causing {pattern.name} failures",
                priority="HIGH",
                effort_estimate="30 minutes",
                impact="High - fixes multiple test failures",
                category="code",
                code_locations=code_context.keys()
            ))
        
        # Prompt-level fixes
        if "prompt" in root_cause_analysis.get("root_cause", "").lower():
            recommendations.append(Recommendation(
                title="Improve Prompt Design",
                description=f"Enhance prompts to prevent {pattern.name} failures",
                priority="HIGH",
                effort_estimate="20 minutes",
                impact="High - improves output quality",
                category="prompt",
                code_locations=[]
            ))
        
        # Validation fixes
        if "validation" in root_cause_analysis.get("root_cause", "").lower():
            recommendations.append(Recommendation(
                title="Add Input/Output Validation",
                description=f"Implement validation to catch {pattern.name} issues early",
                priority="MEDIUM",
                effort_estimate="45 minutes",
                impact="Medium - prevents future failures",
                category="validation",
                code_locations=[]
            ))
        
        # Architecture fixes
        if "architecture" in root_cause_analysis.get("root_cause", "").lower():
            recommendations.append(Recommendation(
                title="Improve System Architecture",
                description=f"Refactor architecture to handle {pattern.name} cases better",
                priority="LOW",
                effort_estimate="2 hours",
                impact="High - long-term improvement",
                category="architecture",
                code_locations=[]
            ))
        
        return recommendations

    def _generate_universal_report(self, test_results: List[TestResult], 
                                 pattern_summaries: Dict[str, PatternSummary],
                                 recommendations: List[Recommendation],
                                 discovered_schema: Dict[str, Any],
                                 architecture: Dict[str, Any],
                                 scan_results: Dict[str, Any]) -> AnalysisReport:
        """Generate universal analysis report."""
        # Calculate summary statistics
        total_tests = len(test_results)
        passed = sum(1 for t in test_results if t.status == "passed")
        failed = sum(1 for t in test_results if t.status == "failed")
        error = sum(1 for t in test_results if t.status == "error")
        skipped = sum(1 for t in test_results if t.status == "skipped")
        
        # Generate action plan
        action_plan = self._generate_action_plan(pattern_summaries, recommendations)
        
        # Create report summary
        summary = ReportSummary(
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            error=error,
            skipped=skipped,
            patterns_found=len(pattern_summaries),
            analysis_timestamp=datetime.now().isoformat(),
            analysis_duration_seconds=0.0,  # Will be set by caller
            project_path=str(self.project_path),
            frameworks_detected=list(scan_results.get('frameworks', {}).keys())
        )
        
        return AnalysisReport(
            summary=summary,
            patterns=pattern_summaries,
            action_plan=action_plan,
            architecture=architecture,
            recommendations=recommendations,
            metadata={
                "analyzer_version": "2.0.0-universal",
                "config": self.config.model_dump(),
                "scan_results": scan_results,
                "discovered_schema": discovered_schema,
                "analysis_mode": "universal"
            }
        )

    def _display_universal_results(self, report: AnalysisReport, console: Console):
        """Display universal analysis results."""
        # System type info
        system_type = report.metadata.get('discovered_schema', {}).get('system_type', 'Unknown')
        console.print(f"\n[bold cyan]🤖 System Type:[/bold cyan] [yellow]{system_type}[/yellow]")
        
        # Call the regular display method
        self._display_rich_results(report, console)

    def _display_universal_results_simple(self, report: AnalysisReport):
        """Display universal analysis results in simple format."""
        system_type = report.metadata.get('discovered_schema', {}).get('system_type', 'Unknown')
        print(f"\n🤖 System Type: {system_type}")
        
        # Call the regular simple display method
        self._display_rich_results_simple(report)
