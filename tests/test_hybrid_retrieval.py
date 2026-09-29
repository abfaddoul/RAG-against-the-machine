import pytest
from langchain_core.embeddings import Embeddings

from src.chunking.models import Chunk
from src.indexing.bm25 import build_bm25_index
from src.indexing.hnsw import build_hnsw_index
from src.retrieval.hybrid import (
    chunk_key,
    reciprocal_rank_fusion,
    retrieve_hybrid_chunks,
)


class FakeEmbeddingModel(Embeddings):
    """Return deterministic vectors for tests."""

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            self.embed_query(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return [1.0, 0.0]


def make_chunk(
    file_path: str,
    text: str,
) -> Chunk:
    """Create a test chunk."""

    return Chunk(
        file_path=file_path,
        first_character_index=0,
        last_character_index=len(text),
        text=text,
    )


def test_reciprocal_rank_fusion_prioritizes_shared_chunk(
) -> None:
    """Test prioritizing chunks found by multiple rankings."""

    chunk_a = make_chunk("a.md", "Chunk A")
    chunk_b = make_chunk("b.md", "Chunk B")
    chunk_c = make_chunk("c.md", "Chunk C")

    results = reciprocal_rank_fusion(
        rankings=[
            [chunk_a, chunk_b],
            [chunk_b, chunk_c],
        ],
        k=3,
    )

    assert results[0] == chunk_b
    assert len(results) == 3
    assert len({
        chunk_key(chunk)
        for chunk in results
    }) == 3


def test_retrieve_hybrid_chunks() -> None:
    """Test combining real BM25 and HNSW results."""

    shared_chunk = make_chunk(
        "shared.md",
        "Dynamic LoRA loading endpoint.",
    )
    lexical_chunk = make_chunk(
        "lexical.md",
        "Dynamic LoRA endpoint configuration.",
    )
    semantic_chunk = make_chunk(
        "semantic.md",
        "Adapter resolver plugin.",
    )

    chunks = [
        shared_chunk,
        lexical_chunk,
        semantic_chunk,
    ]

    bm25_index = build_bm25_index(chunks)

    semantic_index = build_hnsw_index(
        chunks=chunks,
        vectors=[
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        embedding_model_id="fake-model",
    )

    results = retrieve_hybrid_chunks(
        bm25_index=bm25_index,
        semantic_index=semantic_index,
        query="dynamic LoRA loading",
        k=3,
        embedding_model=FakeEmbeddingModel(),
    )

    assert results[0] == shared_chunk
    assert len(results) == 3
    assert len({
        chunk_key(chunk)
        for chunk in results
    }) == 3


def test_reciprocal_rank_fusion_rejects_negative_k(
) -> None:
    """Test rejecting a negative result count."""

    with pytest.raises(
        ValueError,
        match="k cannot be negative",
    ):
        reciprocal_rank_fusion(
            rankings=[],
            k=-1,
        )
