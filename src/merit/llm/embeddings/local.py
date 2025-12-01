"""Local embedding models."""

from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_ID = "granite-embedding-small-english-r2"

# Path to pre-downloaded model weights in merit_analyzer
_WEIGHTS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "merit_analyzer" / "core" / "local_models" / "weights"
)
MODEL_SNAPSHOT = (
    _WEIGHTS_DIR
    / "models--ibm-granite--granite-embedding-small-english-r2"
    / "snapshots"
    / "c949f235cb63fcbd58b1b9e139ff63c8be764eeb"
)


class LocalEmbeddingsEngine:
    """Load and run local embedding checkpoints via SentenceTransformers."""

    def __init__(self, model_paths: dict[str, Path] | None = None):
        self.model_paths = model_paths or {MODEL_ID: MODEL_SNAPSHOT}
        self._models: dict[str, SentenceTransformer] = {}

    async def generate_embeddings(
        self,
        inputs: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for input texts.

        Args:
            inputs: List of texts to embed
            model: Model name (defaults to granite-embedding-small-english-r2)

        Returns:
            List of embedding vectors
        """
        model_name = model or MODEL_ID
        cached = self._models.get(model_name)

        if not cached:
            path = self.model_paths.get(model_name)
            if not path:
                raise ValueError(f"{model_name} is not available locally")
            cached = SentenceTransformer(str(path))
            self._models[model_name] = cached

        vectors = cached.encode(
            sentences=inputs,
            batch_size=min(len(inputs), 32) or 1,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


# Default instance
local_embeddings = LocalEmbeddingsEngine()
