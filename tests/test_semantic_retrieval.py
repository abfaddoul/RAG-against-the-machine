from langchain_core.embeddings import Embeddings

from src.chunking.models import Chunk
from src.indexing.hnsw import (
    HNSWIndex,
    build_hnsw_index,
)
from src.retrieval.semantic import (
    retrieve_semantic_chunks,
)


class FakeQueryEmbeddingModel(Embeddings):
    """Provide deterministic query embeddings."""

    def __init__(self) -> None:
        self.received_query: str | None = None

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed fake documents."""

        return [
            [1.0, 0.0]
            for _ in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Embed a fake query."""

        self.received_query = text

        if "LoRA" in text:
            return [1.0, 0.0]

        return [0.0, 1.0]


def build_semantic_retrieval_index() -> HNSWIndex:
    """Build an index for semantic retrieval tests."""

    chunks = [
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
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    return build_hnsw_index(
        chunks=chunks,
        vectors=vectors,
    )


def test_retrieve_semantic_chunks_embeds_query() -> None:
    """Test embedding a query before HNSW search."""

    index = build_semantic_retrieval_index()
    embedding_model = FakeQueryEmbeddingModel()

    retrieved = retrieve_semantic_chunks(
        index=index,
        query="How does LoRA work?",
        k=1,
        embedding_model=embedding_model,
    )

    assert (
        embedding_model.received_query
        == "How does LoRA work?"
    )
    assert retrieved == [
        index.chunks[0],
    ]


def test_retrieve_semantic_chunks_handles_empty_query() -> None:
    """Test returning no results for an empty query."""

    index = build_semantic_retrieval_index()
    embedding_model = FakeQueryEmbeddingModel()

    retrieved = retrieve_semantic_chunks(
        index=index,
        query="   ",
        k=2,
        embedding_model=embedding_model,
    )

    assert retrieved == []
    assert embedding_model.received_query is None
