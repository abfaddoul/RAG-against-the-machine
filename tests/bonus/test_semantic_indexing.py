import pytest
from langchain_core.embeddings import Embeddings

from src.bonus.embeddings import embed_chunks
from src.bonus.indexing import build_semantic_index
from src.chunking.models import Chunk


class FakeEmbeddingModel(Embeddings):
    """Provide deterministic embeddings for tests."""

    def __init__(self) -> None:
        self.received_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed test documents."""

        self.received_texts.extend(texts)

        return [([1.0, 0.0] if "LoRA" in text else [0.0, 1.0]) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a test query."""

        if "LoRA" in text:
            return [1.0, 0.0]

        return [0.0, 1.0]


def build_test_chunks() -> list[Chunk]:
    """Create chunks used by semantic indexing tests."""

    return [
        Chunk(
            file_path="lora.md",
            first_character_index=0,
            last_character_index=13,
            text="LoRA adapters",
        ),
        Chunk(
            file_path="cake.md",
            first_character_index=0,
            last_character_index=14,
            text="Chocolate cake",
        ),
    ]


def test_build_semantic_index_embeds_chunks() -> None:
    """Test embedding chunks and building HNSW."""

    chunks = build_test_chunks()
    embedding_model = FakeEmbeddingModel()

    index = build_semantic_index(
        chunks=chunks, embedding_model=embedding_model, batch_size=1
    )

    assert embedding_model.received_texts == ["LoRA adapters", "Chocolate cake"]
    assert index.chunks == chunks
    assert index.dimension == 2
    assert index.hnsw.get_current_count() == 2


def test_embed_chunks_rejects_invalid_batch_size() -> None:
    """Test rejecting a non-positive batch size."""

    with pytest.raises(ValueError, match="batch_size must be positive"):
        embed_chunks(
            chunks=build_test_chunks(),
            embedding_model=FakeEmbeddingModel(),
            batch_size=0,
        )
