from .processors.clustering import cluster_failures
from .interface.cli import CLIApplication

__all__ = ["cluster_failures", "CLIApplication"]

def main() -> None:
    CLIApplication().run()
