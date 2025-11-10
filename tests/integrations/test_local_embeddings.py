import pytest

from merit_analyzer.core.local_models.embeddings import (
    MODEL_ID,
    LocalEmbeddingsEngine,
)


@pytest.mark.asyncio
async def test_local_engine_generates_embeddings_from_granite_weights() -> None:
    engine = LocalEmbeddingsEngine()
    inputs = ["alpha prompt", "beta prompt"]

    vectors = await engine.generate_embeddings(inputs, model=MODEL_ID)

    assert len(vectors) == len(inputs)
    assert len(vectors[0]) == 384
    assert vectors[0] != vectors[1]
