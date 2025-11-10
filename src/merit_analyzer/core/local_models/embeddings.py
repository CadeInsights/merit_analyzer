from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer


MODEL_ID = "granite-embedding-small-english-r2"
MODEL_SNAPSHOT = (
    Path(__file__).resolve().parent
    / "weights"
    / "models--ibm-granite--granite-embedding-small-english-r2"
    / "snapshots"
    / "c949f235cb63fcbd58b1b9e139ff63c8be764eeb"
)


class LocalEmbeddingsEngine:
    """Load and run local embedding checkpoints via SentenceTransformers."""

    def __init__(self, model_paths: Dict[str, Path] | None = None):
        self.model_paths = model_paths or {MODEL_ID: MODEL_SNAPSHOT}
        self._models: Dict[str, SentenceTransformer] = {}

    async def generate_embeddings(
        self,
        input_values: List[str],
        model: str | None = None,
    ) -> List[List[float]]:
        model_name = model or MODEL_ID
        cached = self._models.get(model_name)
        if not cached:
            path = self.model_paths.get(model_name)
            if not path:
                raise ValueError(f"{model_name} is not available locally")
            cached = SentenceTransformer(str(path))
            self._models[model_name] = cached
        vectors = cached.encode(
            sentences=input_values,
            batch_size=min(len(input_values), 32) or 1,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()