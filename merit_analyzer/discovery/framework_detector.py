"""Framework detection for AI/ML libraries."""

import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from collections import Counter


class FrameworkDetector:
    """Detect which AI frameworks are in use."""

    # Framework signatures - patterns to look for in code
    FRAMEWORK_SIGNATURES = {
        "langchain": [
            "from langchain", "import langchain", "LangChain",
            "langchain.chat_models", "langchain.llms", "langchain.agents",
            "langchain.tools", "langchain.prompts", "langchain.memory",
            "langchain.chains", "langchain.vectorstores", "langchain.embeddings",
            "langchain.document_loaders", "langchain.text_splitter",
            "langchain.schema", "langchain.callbacks"
        ],
        "llamaindex": [
            "from llama_index", "import llama_index", "llama_index",
            "llama_index.llms", "llama_index.embeddings", "llama_index.vector_stores",
            "llama_index.indices", "llama_index.query_engine", "llama_index.retrievers",
            "llama_index.storage", "llama_index.service_context", "llama_index.llm_predictor"
        ],
        "anthropic": [
            "from anthropic", "import anthropic", "anthropic.Anthropic",
            "messages.create", "claude", "anthropic.completions",
            "anthropic.messages", "anthropic.types"
        ],
        "openai": [
            "from openai", "import openai", "openai.ChatCompletion",
            "openai.Completion", "openai.Embedding", "openai.Moderation",
            "openai.FineTuning", "openai.Images", "openai.Audio",
            "openai.ChatCompletion.create", "openai.Completion.create"
        ],
        "haystack": [
            "from haystack", "import haystack", "haystack.document_stores",
            "haystack.nodes", "haystack.pipelines", "haystack.retrievers",
            "haystack.readers", "haystack.generators", "haystack.preprocessors"
        ],
        "autogen": [
            "from autogen", "import autogen", "autogen.ConversableAgent",
            "autogen.GroupChat", "autogen.GroupChatManager", "autogen.AssistantAgent",
            "autogen.UserProxyAgent", "autogen.ChatCompletion"
        ],
        "crewai": [
            "from crewai", "import crewai", "crewai.Agent", "crewai.Task",
            "crewai.Crew", "crewai.Process", "crewai.Tools"
        ],
        "transformers": [
            "from transformers", "import transformers", "transformers.pipeline",
            "transformers.AutoTokenizer", "transformers.AutoModel",
            "transformers.Trainer", "transformers.TrainingArguments"
        ],
        "torch": [
            "import torch", "from torch", "torch.nn", "torch.optim",
            "torch.utils", "torch.cuda", "torch.device"
        ],
        "tensorflow": [
            "import tensorflow", "from tensorflow", "tensorflow.keras",
            "tensorflow.nn", "tensorflow.optimizers", "tensorflow.layers"
        ],
        "pytorch": [
            "import pytorch", "from pytorch", "pytorch_lightning"
        ],
        "fastapi": [
            "from fastapi", "import fastapi", "FastAPI", "fastapi.APIRouter",
            "fastapi.Depends", "fastapi.HTTPException"
        ],
        "flask": [
            "from flask", "import flask", "Flask", "flask.Blueprint",
            "flask.request", "flask.jsonify"
        ],
        "streamlit": [
            "import streamlit", "from streamlit", "streamlit.app",
            "streamlit.components", "streamlit.elements"
        ],
        "gradio": [
            "import gradio", "from gradio", "gradio.Interface",
            "gradio.Blocks", "gradio.components"
        ],
        "pinecone": [
            "from pinecone", "import pinecone", "pinecone.Index",
            "pinecone.Vector", "pinecone.Query"
        ],
        "weaviate": [
            "from weaviate", "import weaviate", "weaviate.Client",
            "weaviate.Object", "weaviate.Query"
        ],
        "chromadb": [
            "from chromadb", "import chromadb", "chromadb.Client",
            "chromadb.Collection", "chromadb.Query"
        ],
        "qdrant": [
            "from qdrant", "import qdrant", "qdrant.Client",
            "qdrant.collections", "qdrant.vectors"
        ],
        "redis": [
            "import redis", "from redis", "redis.Redis",
            "redis.ConnectionPool", "redis.StrictRedis"
        ],
        "celery": [
            "from celery", "import celery", "celery.Celery",
            "celery.task", "celery.worker"
        ],
        "django": [
            "from django", "import django", "django.db",
            "django.http", "django.views", "django.models"
        ],
        "pandas": [
            "import pandas", "from pandas", "pandas.DataFrame",
            "pandas.Series", "pandas.read_csv"
        ],
        "numpy": [
            "import numpy", "from numpy", "numpy.array",
            "numpy.ndarray", "numpy.random"
        ],
        "scikit-learn": [
            "from sklearn", "import sklearn", "sklearn.model_selection",
            "sklearn.ensemble", "sklearn.linear_model", "sklearn.metrics"
        ],
        "pytest": [
            "import pytest", "from pytest", "pytest.fixture",
            "pytest.mark", "pytest.parametrize"
        ],
        "unittest": [
            "import unittest", "from unittest", "unittest.TestCase",
            "unittest.mock", "unittest.main"
        ]
    }

    def __init__(self):
        """Initialize framework detector."""
        self.detected_frameworks: Dict[str, bool] = {}
        self.framework_usage: Dict[str, List[str]] = {}
        self.confidence_scores: Dict[str, float] = {}

    def detect(self, python_files: List[Path], verbose: bool = False) -> Dict[str, bool]:
        """
        Detect which frameworks are in use.

        Args:
            python_files: List of Python files to analyze
            verbose: If True, print verbose progress messages

        Returns:
            Dictionary mapping framework names to detection status
        """
        if verbose:
            print("🔍 Detecting AI frameworks...")
        
        self.detected_frameworks = {framework: False for framework in self.FRAMEWORK_SIGNATURES}
        self.framework_usage = {framework: [] for framework in self.FRAMEWORK_SIGNATURES}
        self.confidence_scores = {framework: 0.0 for framework in self.FRAMEWORK_SIGNATURES}
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self._analyze_file_content(py_file, content)
            except Exception as e:
                if verbose:
                    print(f"    ⚠️  Error reading {py_file}: {e}")
                continue
        
        # Calculate confidence scores
        self._calculate_confidence_scores()
        
        # Only return detected frameworks
        detected = {k: v for k, v in self.detected_frameworks.items() if v}
        
        if verbose:
            if detected:
                print(f"  ✅ Detected frameworks: {', '.join(detected.keys())}")
            else:
                print("  ℹ️  No AI frameworks detected")
        
        return detected

    def _analyze_file_content(self, file_path: Path, content: str):
        """Analyze file content for framework signatures."""
        for framework, signatures in self.FRAMEWORK_SIGNATURES.items():
            matches = []
            
            for signature in signatures:
                if signature in content:
                    matches.append(signature)
                    self.framework_usage[framework].append(str(file_path))
            
            if matches:
                self.detected_frameworks[framework] = True
                # Calculate confidence based on number of matches
                confidence = min(len(matches) / len(signatures), 1.0)
                self.confidence_scores[framework] = max(
                    self.confidence_scores[framework], 
                    confidence
                )

    def _calculate_confidence_scores(self):
        """Calculate confidence scores for detected frameworks."""
        for framework in self.FRAMEWORK_SIGNATURES:
            if self.detected_frameworks[framework]:
                # Base confidence from signature matches
                base_confidence = self.confidence_scores[framework]
                
                # Boost confidence based on usage frequency
                usage_count = len(set(self.framework_usage[framework]))
                usage_boost = min(usage_count * 0.1, 0.3)
                
                # Boost confidence based on import patterns
                import_boost = self._calculate_import_confidence(framework)
                
                self.confidence_scores[framework] = min(
                    base_confidence + usage_boost + import_boost, 
                    1.0
                )

    def _calculate_import_confidence(self, framework: str) -> float:
        """Calculate confidence boost from import patterns."""
        import_patterns = {
            "langchain": r"from\s+langchain\s+import",
            "llamaindex": r"from\s+llama_index\s+import",
            "anthropic": r"from\s+anthropic\s+import",
            "openai": r"from\s+openai\s+import",
            "transformers": r"from\s+transformers\s+import",
            "torch": r"import\s+torch",
            "tensorflow": r"import\s+tensorflow",
            "fastapi": r"from\s+fastapi\s+import",
            "flask": r"from\s+flask\s+import",
            "streamlit": r"import\s+streamlit",
            "gradio": r"import\s+gradio",
        }
        
        if framework not in import_patterns:
            return 0.0
        
        pattern = import_patterns[framework]
        import_count = 0
        
        for file_path in self.framework_usage[framework]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    import_count += len(re.findall(pattern, content, re.IGNORECASE))
            except Exception:
                continue
        
        return min(import_count * 0.05, 0.2)

    def get_framework_details(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about detected frameworks."""
        details = {}
        
        for framework, detected in self.detected_frameworks.items():
            if detected:
                details[framework] = {
                    "detected": True,
                    "confidence": self.confidence_scores[framework],
                    "files_using": list(set(self.framework_usage[framework])),
                    "usage_count": len(set(self.framework_usage[framework])),
                    "signatures_found": self._get_found_signatures(framework),
                }
        
        return details

    def _get_found_signatures(self, framework: str) -> List[str]:
        """Get signatures that were found for a framework."""
        found_signatures = []
        signatures = self.FRAMEWORK_SIGNATURES[framework]
        
        for file_path in self.framework_usage[framework]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for signature in signatures:
                    if signature in content and signature not in found_signatures:
                        found_signatures.append(signature)
            except Exception:
                continue
        
        return found_signatures

    def get_ai_frameworks(self) -> List[str]:
        """Get list of AI-specific frameworks (excluding general Python libraries)."""
        ai_frameworks = {
            "langchain", "llamaindex", "anthropic", "openai", "haystack",
            "autogen", "crewai", "transformers", "torch", "tensorflow",
            "pytorch", "pinecone", "weaviate", "chromadb", "qdrant"
        }
        
        return [
            framework for framework in ai_frameworks 
            if self.detected_frameworks.get(framework, False)
        ]

    def get_web_frameworks(self) -> List[str]:
        """Get list of web frameworks."""
        web_frameworks = {"fastapi", "flask", "streamlit", "gradio", "django"}
        
        return [
            framework for framework in web_frameworks 
            if self.detected_frameworks.get(framework, False)
        ]

    def get_ml_frameworks(self) -> List[str]:
        """Get list of ML frameworks."""
        ml_frameworks = {
            "transformers", "torch", "tensorflow", "pytorch", 
            "scikit-learn", "pandas", "numpy"
        }
        
        return [
            framework for framework in ml_frameworks 
            if self.detected_frameworks.get(framework, False)
        ]

    def get_vector_db_frameworks(self) -> List[str]:
        """Get list of vector database frameworks."""
        vector_db_frameworks = {"pinecone", "weaviate", "chromadb", "qdrant", "redis"}
        
        return [
            framework for framework in vector_db_frameworks 
            if self.detected_frameworks.get(framework, False)
        ]

    def get_framework_architecture_hints(self) -> Dict[str, List[str]]:
        """Get architecture hints based on detected frameworks."""
        hints = {
            "likely_rag": [],
            "likely_agent_system": [],
            "likely_web_app": [],
            "likely_ml_pipeline": [],
            "likely_vector_search": []
        }
        
        # RAG patterns
        if any(fw in self.detected_frameworks for fw in ["langchain", "llamaindex"]):
            hints["likely_rag"].extend(["langchain", "llamaindex"])
        
        # Agent systems
        if any(fw in self.detected_frameworks for fw in ["langchain", "autogen", "crewai"]):
            hints["likely_agent_system"].extend(["langchain", "autogen", "crewai"])
        
        # Web applications
        if any(fw in self.detected_frameworks for fw in ["fastapi", "flask", "streamlit", "gradio"]):
            hints["likely_web_app"].extend(["fastapi", "flask", "streamlit", "gradio"])
        
        # ML pipelines
        if any(fw in self.detected_frameworks for fw in ["transformers", "torch", "tensorflow", "scikit-learn"]):
            hints["likely_ml_pipeline"].extend(["transformers", "torch", "tensorflow", "scikit-learn"])
        
        # Vector search
        if any(fw in self.detected_frameworks for fw in ["pinecone", "weaviate", "chromadb", "qdrant"]):
            hints["likely_vector_search"].extend(["pinecone", "weaviate", "chromadb", "qdrant"])
        
        # Remove empty lists
        return {k: v for k, v in hints.items() if v}

    def suggest_analysis_approach(self) -> List[str]:
        """Suggest analysis approach based on detected frameworks."""
        suggestions = []
        
        ai_frameworks = self.get_ai_frameworks()
        web_frameworks = self.get_web_frameworks()
        ml_frameworks = self.get_ml_frameworks()
        
        if "langchain" in ai_frameworks:
            suggestions.append("Focus on LangChain chains, agents, and tools")
        
        if "llamaindex" in ai_frameworks:
            suggestions.append("Analyze LlamaIndex indices, query engines, and retrievers")
        
        if "autogen" in ai_frameworks or "crewai" in ai_frameworks:
            suggestions.append("Examine multi-agent coordination and communication")
        
        if web_frameworks:
            suggestions.append("Check API endpoints and request/response handling")
        
        if ml_frameworks:
            suggestions.append("Review model loading, inference, and data preprocessing")
        
        if not suggestions:
            suggestions.append("Perform general code analysis for AI system patterns")
        
        return suggestions
