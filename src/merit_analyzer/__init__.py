import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

from .processors.clustering import cluster_failures
from .interface.cli import CLIApplication

__all__ = ["cluster_failures", "CLIApplication"]

def main() -> None:
    load_dotenv(Path.cwd() / ".env")
    
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    warnings.filterwarnings("ignore", category=SyntaxWarning, module="hdbscan")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    
    CLIApplication().run()
