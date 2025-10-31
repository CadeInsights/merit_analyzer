"""Configuration management for Merit Analyzer."""

from pydantic import BaseModel, Field  # type: ignore
from typing import Optional, Dict, Any, List
from pathlib import Path


class MeritConfig(BaseModel):
    """Configuration for Merit Analyzer."""

    # API Configuration
    api_key: str = Field(..., description="Anthropic API key (not needed for Bedrock)")
    provider: str = Field(default="anthropic", description="API provider: 'anthropic' (direct) or 'bedrock' (AWS)")
    model: str = Field(default="claude-sonnet-4-5-20250929", description="Claude model to use")
    aws_region: Optional[str] = Field(default=None, description="AWS region for Bedrock (defaults to us-east-1)")
    max_tokens: int = Field(default=4096, description="Maximum tokens per request")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Temperature for generation")

    # Analysis Configuration
    project_path: str = Field(..., description="Path to the AI system codebase")
    max_iterations: int = Field(default=20, description="Maximum AI analysis iterations")
    min_cluster_size: int = Field(default=2, description="Minimum cluster size for patterns")
    max_patterns: int = Field(default=10, description="Maximum number of patterns to analyze")
    similarity_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Similarity threshold for clustering"
    )

    # File Processing
    exclude_patterns: List[str] = Field(
        default_factory=lambda: [
            "**/__pycache__/**",
            "**/node_modules/**",
            "**/.git/**",
            "**/venv/**",
            "**/.venv/**",
            "**/env/**",
            "**/.env/**",
            "**/build/**",
            "**/dist/**",
            "**/*.pyc",
            "**/.pytest_cache/**",
            "**/coverage/**",
        ],
        description="File patterns to exclude from analysis"
    )
    include_extensions: List[str] = Field(
        default_factory=lambda: [".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".yaml", ".yml", ".json"],
        description="File extensions to include in analysis"
    )

    # Output Configuration
    output_format: str = Field(default="json", description="Default output format (json/markdown)")
    save_intermediate_results: bool = Field(
        default=False, description="Save intermediate analysis results"
    )
    verbose: bool = Field(default=True, description="Enable verbose logging")

    # AI Analysis Configuration
    ai_config: Dict[str, Any] = Field(
        default_factory=dict, description="Additional AI analysis configuration"
    )

    # Performance Configuration
    parallel_analysis: bool = Field(default=True, description="Enable parallel pattern analysis")
    max_workers: int = Field(default=10, description="Maximum worker threads for parallel processing")
    cache_architecture: bool = Field(default=True, description="Cache discovered architecture")

    class Config:
        """Pydantic configuration."""

        env_prefix = "MERIT_"
        case_sensitive = False

    @classmethod
    def from_file(cls, config_path: str) -> "MeritConfig":
        """Load configuration from file."""
        import json
        import yaml
        from pathlib import Path

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls(**data)

    def save(self, config_path: str) -> None:
        """Save configuration to file."""
        import json
        import yaml
        from pathlib import Path

        path = Path(config_path)
        data = self.model_dump()

        with open(path, "w", encoding="utf-8") as f:
            if path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2)

    def get_excluded_files(self) -> List[str]:
        """Get list of files to exclude based on patterns."""
        import glob
        from pathlib import Path

        project_path = Path(self.project_path)
        excluded = set()

        for pattern in self.exclude_patterns:
            # Convert pattern to absolute path
            if pattern.startswith("**/"):
                search_pattern = str(project_path / pattern)
            else:
                search_pattern = str(project_path / "**" / pattern)

            for file_path in glob.glob(search_pattern, recursive=True):
                excluded.add(str(Path(file_path).relative_to(project_path)))

        return list(excluded)

    def should_include_file(self, file_path: str) -> bool:
        """Check if file should be included in analysis."""
        from pathlib import Path

        path = Path(file_path)
        
        # Check extension
        if path.suffix.lower() not in self.include_extensions:
            return False

        # Check exclusion patterns
        excluded_files = self.get_excluded_files()
        relative_path = str(path.relative_to(Path(self.project_path)))
        
        return relative_path not in excluded_files
