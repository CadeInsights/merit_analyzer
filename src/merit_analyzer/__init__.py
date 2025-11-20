import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning, message=".*invalid escape sequence.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")

from dotenv import load_dotenv

from .processors.clustering import cluster_failures
from .interface.cli import CLIApplication

__all__ = ["cluster_failures", "CLIApplication"]


def main() -> None:
    load_dotenv(Path.cwd() / ".env")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    CLIApplication().run()
