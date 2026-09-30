from pathlib import Path

import joblib
import pytest

from src.bonus.indexing import HNSWIndex, build_hnsw_index
from src.bonus.search import retrieve_hnsw_chunks
from src.bonus.storage import (
    HNSW_GRAPH_FILENAME,
    HNSW_METADATA_FILENAME,
    load_hnsw_index,
    save_hnsw_index,
)
from src.chunking.models import Chunk


def build_storage_test_index() -> HNSWIndex:
    """Build a small HNSW index for storage tests."""

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
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    return build_hnsw_index(chunks=chunks, vectors=vectors)


def test_save_and_load_hnsw_index(tmp_path: Path) -> None:
    """Test saving and restoring an HNSW index."""

    index = build_storage_test_index()
    index_directory = tmp_path / "semantic_index"

    save_hnsw_index(index=index, directory=index_directory)
    loaded_index = load_hnsw_index(index_directory)

    assert (index_directory / HNSW_GRAPH_FILENAME).exists()
    assert (index_directory / HNSW_METADATA_FILENAME).exists()

    assert loaded_index.chunks == index.chunks
    assert loaded_index.dimension == index.dimension
    assert loaded_index.embedding_model_id == index.embedding_model_id
    assert loaded_index.ef_search == index.ef_search
    assert loaded_index.active_count == 2
    assert loaded_index.hnsw.get_current_count() == 2

    retrieved = retrieve_hnsw_chunks(index=loaded_index, query_vector=[1.0, 0.0], k=1)

    assert retrieved == [loaded_index.chunks[0]]


def test_save_and_load_preserves_deleted_labels(tmp_path: Path) -> None:
    """Test preserving deletion state and active count."""

    index = build_storage_test_index()
    index.hnsw.mark_deleted(0)
    index.active_count = 1

    index_directory = tmp_path / "semantic_index"

    save_hnsw_index(index=index, directory=index_directory)
    loaded_index = load_hnsw_index(index_directory)

    retrieved = retrieve_hnsw_chunks(index=loaded_index, query_vector=[1.0, 0.0], k=10)

    assert loaded_index.active_count == 1
    assert retrieved == [loaded_index.chunks[1]]


def test_load_hnsw_index_rejects_wrong_metadata(tmp_path: Path) -> None:
    """Test rejecting invalid HNSW metadata."""

    index_directory = tmp_path / "semantic_index"
    index_directory.mkdir()

    metadata_path = index_directory / HNSW_METADATA_FILENAME
    joblib.dump(["not", "metadata"], metadata_path)

    with pytest.raises(TypeError, match="HNSWMetadata"):
        load_hnsw_index(index_directory)
