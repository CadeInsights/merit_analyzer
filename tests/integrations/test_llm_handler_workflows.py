import pytest
import pytest_asyncio
from pydantic import BaseModel

from merit_analyzer.core.llm_driver import get_llm_client
from merit_analyzer.core.llm_driver.abstract_provider_handler import LLMAbstractHandler
from merit_analyzer.core.local_models.embeddings import MODEL_ID as LOCAL_MODEL_ID


@pytest_asyncio.fixture(scope="session")
async def llm_client() -> LLMAbstractHandler:
    return await get_llm_client()


@pytest.mark.asyncio
async def test_generate_embeddings_with_production_llm(llm_client: LLMAbstractHandler) -> None:
    inputs = [
        "User service timed out during login attempts.",
        "Payment gateway rejected the transaction with code 502.",
        "Formatting mismatch for CSV export columns.",
    ]

    vectors = await llm_client.generate_embeddings(inputs)

    assert len(vectors) == len(inputs)
    expected_dim = 384 if llm_client.default_embedding_model == LOCAL_MODEL_ID else 1536
    assert len(vectors[0]) == expected_dim
    assert vectors[0] != vectors[1]


class SampleSchema(BaseModel):
    subject: str
    severity: int


@pytest.mark.asyncio
async def test_create_object_returns_expected_schema(llm_client: LLMAbstractHandler) -> None:
    prompt = """Return JSON saying that subject is "integration coverage" and it's severity is 2."""

    result = await llm_client.create_object(prompt=prompt, schema=SampleSchema)

    assert result.subject == "integration coverage"
    assert result.severity == 2
