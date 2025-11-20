from .embeddings import MODEL_ID, LocalEmbeddingsEngine


local_embeddings_engine = LocalEmbeddingsEngine()

__all__ = ["MODEL_ID", "LocalEmbeddingsEngine", "local_embeddings_engine"]
