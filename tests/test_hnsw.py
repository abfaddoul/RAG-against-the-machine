import pytest

from src.chunking.models import Chunk
from src.embeddings.model import (
    DEFAULT_EMBEDDING_MODEL_ID,
)
from src.indexing.hnsw import build_hnsw_index


def test_build_hnsw_index() -> None:
    """Test building an HNSW index from vectors."""

    chunks = [
        Chunk(
            file_path="first.txt",
            first_character_index=0,
            last_character_index=10,
            text="LoRA adapter",
        ),
        Chunk(
            file_path="second.txt",
            first_character_index=0,
            last_character_index=14,
            text="Chocolate cake",
        ),
    ]
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    index = build_hnsw_index(
        chunks=chunks,
        vectors=vectors,
    )

    assert index.chunks == chunks
    assert index.dimension == 2
    assert (
        index.embedding_model_id
        == DEFAULT_EMBEDDING_MODEL_ID
    )
    assert index.hnsw.get_current_count() == 2


def test_build_hnsw_index_rejects_empty_chunks() -> None:
    """Test rejecting an empty chunk collection."""

    with pytest.raises(
        ValueError,
        match="without chunks",
    ):
        build_hnsw_index(
            chunks=[],
            vectors=[],
        )


def test_build_hnsw_index_rejects_misaligned_vectors() -> None:
    """Test rejecting misaligned chunks and vectors."""

    chunk = Chunk(
        file_path="example.txt",
        first_character_index=0,
        last_character_index=5,
        text="LoRA",
    )

    with pytest.raises(
        ValueError,
        match="equal lengths",
    ):
        build_hnsw_index(
            chunks=[chunk],
            vectors=[],
        )
