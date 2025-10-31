"""Project structure scanning and analysis."""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
import re


class ProjectScanner:
    """Quick scan of project structure for Merit's analysis engine."""

    def __init__(self, project_path: str):
        """
        Initialize project scanner.

        Args:
            project_path: Path to the project to scan
        """
        self.project_path = Path(project_path)
        self.python_files: List[Path] = []
        self.prompt_files: List[Path] = []
        self.config_files: List[Path] = []
        self.entry_points: List[Path] = []
        self.imports: Dict[str, List[str]] = {}
        self.functions: Dict[str, List[str]] = {}
        self.classes: Dict[str, List[str]] = {}

    def scan(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive project scan.

        Args:
            verbose: If True, print verbose progress messages

        Returns:
            Dictionary with project structure information
        """
        if verbose:
            print("🔍 Scanning project structure...")
        
        self._find_files(verbose)
        self._identify_entry_points(verbose)
        self._analyze_imports(verbose)
        self._analyze_functions_and_classes(verbose)
        
        return {
            "project_path": str(self.project_path),
            "python_files": [str(f) for f in self.python_files],
            "prompt_files": [str(f) for f in self.prompt_files],
            "config_files": [str(f) for f in self.config_files],
            "entry_points": [str(f) for f in self.entry_points],
            "file_count": len(self.python_files),
            "estimated_loc": self._estimate_loc(),
            "imports": self.imports,
            "functions": self.functions,
            "classes": self.classes,
            "project_structure": self._get_project_structure(),
        }

    def _find_files(self, verbose: bool = False):
        """Find relevant files in the project."""
        if verbose:
            print("  📁 Finding files...")
        
        for root, dirs, files in os.walk(self.project_path):
            # Skip common ignore patterns
            dirs[:] = [d for d in dirs if d not in {
                '.git', '__pycache__', 'node_modules', '.venv', 'venv', 
                'env', '.env', 'build', 'dist', '.pytest_cache', 'coverage',
                '.mypy_cache', '.tox', 'htmlcov'
            }]
            
            for file in files:
                file_path = Path(root) / file
                
                if file.endswith('.py'):
                    self.python_files.append(file_path)
                elif self._is_prompt_file(file_path):
                    self.prompt_files.append(file_path)
                elif self._is_config_file(file_path):
                    self.config_files.append(file_path)

    def _is_prompt_file(self, file_path: Path) -> bool:
        """Check if file is likely a prompt file."""
        name = file_path.name.lower()
        return (
            name.endswith(('.txt', '.md', '.prompt', '.template')) or
            'prompt' in name or
            'template' in name or
            'system' in name or
            'instruction' in name
        )

    def _is_config_file(self, file_path: Path) -> bool:
        """Check if file is a configuration file."""
        name = file_path.name.lower()
        return name in {
            'config.json', 'config.yaml', 'config.yml', '.env', 'requirements.txt',
            'pyproject.toml', 'setup.py', 'package.json', 'dockerfile', 'docker-compose.yml',
            'docker-compose.yaml', 'environment.yml', 'conda.yml'
        }

    def _identify_entry_points(self, verbose: bool = False):
        """Identify likely entry points."""
        if verbose:
            print("  🚪 Identifying entry points...")
        
        candidates = []
        
        # Check for common entry point names
        common_names = {'main.py', 'app.py', 'run.py', 'agent.py', 'server.py', 'cli.py'}
        
        for py_file in self.python_files:
            if py_file.name in common_names:
                candidates.append(py_file)
                continue
            
            # Check if file has if __name__ == "__main__"
            if self._has_main_block(py_file):
                candidates.append(py_file)
        
        self.entry_points = candidates

    def _has_main_block(self, file_path: Path) -> bool:
        """Check if file has a main execution block."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return '__name__' in content and '__main__' in content
        except Exception:
            return False

    def _analyze_imports(self, verbose: bool = False):
        """Analyze imports across all Python files."""
        if verbose:
            print("  📦 Analyzing imports...")
        
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                imports = self._extract_imports(tree)
                self.imports[str(py_file)] = imports
            except Exception:
                self.imports[str(py_file)] = []

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        
        return imports

    def _analyze_functions_and_classes(self, verbose: bool = False):
        """Analyze functions and classes in Python files."""
        if verbose:
            print("  🔧 Analyzing functions and classes...")
        
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                functions, classes = self._extract_functions_and_classes(tree)
                self.functions[str(py_file)] = functions
                self.classes[str(py_file)] = classes
            except Exception:
                self.functions[str(py_file)] = []
                self.classes[str(py_file)] = []

    def _extract_functions_and_classes(self, tree: ast.AST) -> tuple[List[str], List[str]]:
        """Extract function and class names from AST."""
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return functions, classes

    def _estimate_loc(self) -> int:
        """Estimate lines of code."""
        total = 0
        for py_file in self.python_files[:100]:  # Sample first 100 files
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    total += len(f.readlines())
            except Exception:
                pass
        return total

    def _get_project_structure(self) -> Dict[str, Any]:
        """Get hierarchical project structure."""
        structure = {}
        
        for py_file in self.python_files:
            relative_path = py_file.relative_to(self.project_path)
            path_parts = relative_path.parts
            
            current = structure
            for part in path_parts[:-1]:  # Exclude filename
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Add file info
            filename = path_parts[-1]
            current[filename] = {
                "type": "python_file",
                "functions": self.functions.get(str(py_file), []),
                "classes": self.classes.get(str(py_file), []),
                "imports": self.imports.get(str(py_file), [])[:10],  # Limit imports
            }
        
        return structure

    def get_file_dependencies(self, file_path: str) -> List[str]:
        """Get files that this file depends on."""
        if file_path not in self.imports:
            return []
        
        dependencies = []
        file_imports = self.imports[file_path]
        
        for py_file in self.python_files:
            file_name = py_file.stem
            file_module = str(py_file.relative_to(self.project_path)).replace('/', '.').replace('\\', '.')[:-3]
            
            for imp in file_imports:
                if imp.startswith(file_module) or imp == file_name:
                    dependencies.append(str(py_file))
                    break
        
        return dependencies

    def find_files_by_content(self, pattern: str, case_sensitive: bool = False) -> List[str]:
        """Find files containing a specific pattern."""
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled_pattern = re.compile(pattern, flags)
        matching_files = []
        
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if compiled_pattern.search(content):
                        matching_files.append(str(py_file))
            except Exception:
                pass
        
        return matching_files

    def get_project_summary(self) -> Dict[str, Any]:
        """Get high-level project summary."""
        all_imports = []
        for imports in self.imports.values():
            all_imports.extend(imports)
        
        all_functions = []
        for functions in self.functions.values():
            all_functions.extend(functions)
        
        all_classes = []
        for classes in self.classes.values():
            all_classes.extend(classes)
        
        return {
            "total_files": len(self.python_files),
            "total_functions": len(all_functions),
            "total_classes": len(all_classes),
            "unique_imports": len(set(all_imports)),
            "entry_points": len(self.entry_points),
            "estimated_loc": self._estimate_loc(),
            "most_common_imports": self._get_most_common_imports(),
        }

    def _get_most_common_imports(self) -> List[tuple[str, int]]:
        """Get most common imports across the project."""
        from collections import Counter
        
        all_imports = []
        for imports in self.imports.values():
            all_imports.extend(imports)
        
        # Extract base module names
        base_modules = []
        for imp in all_imports:
            if '.' in imp:
                base_modules.append(imp.split('.')[0])
            else:
                base_modules.append(imp)
        
        return Counter(base_modules).most_common(10)
