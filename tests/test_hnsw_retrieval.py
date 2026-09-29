import pytest

from src.chunking.models import Chunk
from src.indexing.hnsw import (
    HNSWIndex,
    build_hnsw_index,
)
from src.retrieval.hnsw import retrieve_hnsw_chunks


def build_test_hnsw_index() -> HNSWIndex:
    """Build a small HNSW index for retrieval tests."""

    chunks = [
        Chunk(
            file_path="first.txt",
            first_character_index=0,
            last_character_index=10,
            text="LoRA adapters",
        ),
        Chunk(
            file_path="second.txt",
            first_character_index=0,
            last_character_index=10,
            text="Chocolate cake",
        ),
        Chunk(
            file_path="third.txt",
            first_character_index=0,
            last_character_index=12,
            text="Loading LoRA",
        ),
    ]
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
    ]

    return build_hnsw_index(
        chunks=chunks,
        vectors=vectors,
    )


def test_retrieve_hnsw_chunks_returns_nearest() -> None:
    """Test returning chunks nearest to a query vector."""

    index = build_test_hnsw_index()

    retrieved = retrieve_hnsw_chunks(
        index=index,
        query_vector=[1.0, 0.0],
        k=2,
    )

    assert retrieved == [
        index.chunks[0],
        index.chunks[2],
    ]


def test_retrieve_hnsw_chunks_handles_k_boundaries() -> None:
    """Test zero, negative, and oversized limits."""

    index = build_test_hnsw_index()

    assert retrieve_hnsw_chunks(
        index=index,
        query_vector=[1.0, 0.0],
        k=0,
    ) == []

    assert len(
        retrieve_hnsw_chunks(
            index=index,
            query_vector=[1.0, 0.0],
            k=10,
        )
    ) == 3

    with pytest.raises(
        ValueError,
        match="k cannot be negative",
    ):
        retrieve_hnsw_chunks(
            index=index,
            query_vector=[1.0, 0.0],
            k=-1,
        )


def test_retrieve_hnsw_chunks_rejects_wrong_dimension() -> None:
    """Test rejecting a query with another dimension."""

    index = build_test_hnsw_index()

    with pytest.raises(
        ValueError,
        match="dimension does not match",
    ):
        retrieve_hnsw_chunks(
            index=index,
            query_vector=[1.0, 0.0, 0.0],
            k=1,
        )
