from pathlib import Path

import joblib
import pytest
from langchain_core.embeddings import Embeddings

from src.bonus.incremental import (
    IncrementalState,
    add_file_to_hnsw_index,
    build_incremental_state,
    detect_file_changes,
    hash_corpus_files,
    hash_file,
    load_incremental_state,
    remove_file_from_hnsw_index,
    save_incremental_state,
    update_semantic_corpus,
)
from src.bonus.indexing import build_hnsw_index, index_semantic_corpus
from src.chunking.models import Chunk


class DeterministicEmbeddings(Embeddings):
    """Create deterministic vectors for pipeline tests."""

    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents and record their text."""

        self.embedded_texts.extend(texts)

        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""

        return [float(len(text)), 1.0]


def test_hash_file_changes_with_content(tmp_path: Path) -> None:
    """Test detecting a file content change."""

    file_path = tmp_path / "example.md"
    file_path.write_text("Original content", encoding="utf-8")

    original_hash = hash_file(file_path)

    file_path.write_text("Modified content", encoding="utf-8")

    modified_hash = hash_file(file_path)

    assert original_hash != modified_hash


def test_hash_corpus_files(tmp_path: Path) -> None:
    """Test hashing supported corpus files."""

    nested_directory = tmp_path / "docs"
    nested_directory.mkdir()

    source_path = nested_directory / "example.md"
    source_path.write_text("Example documentation", encoding="utf-8")

    ignored_path = tmp_path / "image.png"
    ignored_path.write_bytes(b"not searchable")

    hashes = hash_corpus_files(tmp_path)

    assert list(hashes) == [str(source_path)]
    assert hashes[str(source_path)] == hash_file(source_path)


def test_detect_file_changes() -> None:
    """Test classifying corpus file changes."""

    previous_hashes = {
        "unchanged.md": "hash-a",
        "modified.py": "hash-b",
        "deleted.txt": "hash-c",
    }
    current_hashes = {
        "unchanged.md": "hash-a",
        "modified.py": "hash-d",
        "added.rst": "hash-e",
    }

    changes = detect_file_changes(
        previous_hashes=previous_hashes, current_hashes=current_hashes
    )

    assert changes.added == ["added.rst"]
    assert changes.modified == ["modified.py"]
    assert changes.deleted == ["deleted.txt"]
    assert changes.unchanged == ["unchanged.md"]
    assert changes.has_changes is True


def test_detect_file_changes_without_changes() -> None:
    """Test recognizing an unchanged corpus."""

    hashes = {"example.md": "same-hash"}

    changes = detect_file_changes(previous_hashes=hashes, current_hashes=hashes)

    assert changes.has_changes is False


def test_save_and_load_incremental_state(tmp_path: Path) -> None:
    """Test saving and loading incremental state."""

    state = IncrementalState(file_hashes={"data/raw/example.md": "example-hash"})

    state_path = tmp_path / "processed" / "incremental_state.joblib"

    save_incremental_state(state=state, path=state_path)

    loaded_state = load_incremental_state(state_path)

    assert state_path.exists()
    assert loaded_state == state


def test_load_incremental_state_rejects_wrong_object(tmp_path: Path) -> None:
    """Test rejecting invalid incremental state."""

    state_path = tmp_path / "wrong.joblib"

    joblib.dump({"not": "an incremental state"}, state_path)

    with pytest.raises(
        TypeError, match=("File does not contain " "an IncrementalState")
    ):
        load_incremental_state(state_path)


def test_remove_file_from_hnsw_index() -> None:
    """Test deleting only one file's active labels."""

    chunks = [
        Chunk(
            file_path="a.py",
            first_character_index=0,
            last_character_index=5,
            text="first",
        ),
        Chunk(
            file_path="a.py",
            first_character_index=5,
            last_character_index=11,
            text="second",
        ),
        Chunk(
            file_path="b.py",
            first_character_index=0,
            last_character_index=5,
            text="third",
        ),
    ]
    index = build_hnsw_index(
        chunks=chunks, vectors=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    )
    state = build_incremental_state(
        file_hashes={"a.py": "hash-a", "b.py": "hash-b"}, semantic_index=index
    )

    remove_file_from_hnsw_index(index=index, state=state, file_path="a.py")

    assert index.active_count == 1
    assert "a.py" not in state.file_hashes
    assert "a.py" not in state.labels_by_file
    assert state.labels_by_file["b.py"] == [2]
    assert state.next_label == 3


def test_add_file_to_hnsw_index() -> None:
    """Test appending one file with fresh labels."""

    original_chunk = Chunk(
        file_path="a.py", first_character_index=0, last_character_index=5, text="first"
    )
    index = build_hnsw_index(chunks=[original_chunk], vectors=[[1.0, 0.0]])
    state = build_incremental_state(
        file_hashes={"a.py": "hash-a"}, semantic_index=index
    )
    new_chunks = [
        Chunk(
            file_path="b.py",
            first_character_index=0,
            last_character_index=6,
            text="second",
        ),
        Chunk(
            file_path="b.py",
            first_character_index=6,
            last_character_index=11,
            text="third",
        ),
    ]

    add_file_to_hnsw_index(
        index=index,
        state=state,
        file_path="b.py",
        file_hash="hash-b",
        chunks=new_chunks,
        vectors=[[0.9, 0.1], [0.0, 1.0]],
    )

    assert index.chunks == [original_chunk, *new_chunks]
    assert index.active_count == 3
    assert index.hnsw.get_current_count() == 3
    assert state.file_hashes["b.py"] == "hash-b"
    assert state.labels_by_file["b.py"] == [1, 2]
    assert state.next_label == 3


def test_update_semantic_corpus_processes_changes(tmp_path: Path) -> None:
    """Test updating only added and modified files."""

    corpus_directory = tmp_path / "corpus"
    corpus_directory.mkdir()

    modified_path = corpus_directory / "a.py"
    unchanged_path = corpus_directory / "b.py"
    deleted_path = corpus_directory / "c.py"

    modified_path.write_text("original alpha", encoding="utf-8")
    unchanged_path.write_text("unchanged beta", encoding="utf-8")
    deleted_path.write_text("deleted gamma", encoding="utf-8")

    index_directory = tmp_path / "semantic_index"
    embeddings = DeterministicEmbeddings()

    index_semantic_corpus(
        corpus_directory=corpus_directory,
        index_directory=index_directory,
        embedding_model_id="test-embeddings",
        embedding_model=embeddings,
    )

    embeddings.embedded_texts.clear()

    modified_path.write_text("modified alpha", encoding="utf-8")
    deleted_path.unlink()

    added_path = corpus_directory / "d.py"
    added_path.write_text("added delta", encoding="utf-8")

    updated_index, changes = update_semantic_corpus(
        corpus_directory=corpus_directory,
        index_directory=index_directory,
        embedding_model=embeddings,
    )

    state = load_incremental_state(index_directory / "incremental_state.joblib")

    assert changes.added == [str(added_path)]
    assert changes.modified == [str(modified_path)]
    assert changes.deleted == [str(deleted_path)]
    assert changes.unchanged == [str(unchanged_path)]
    assert set(embeddings.embedded_texts) == {"modified alpha", "added delta"}
    assert updated_index.active_count == 3
    assert len(updated_index.chunks) == 5
    assert state.labels_by_file[str(modified_path)] == [3]
    assert state.labels_by_file[str(unchanged_path)] == [1]
    assert state.labels_by_file[str(added_path)] == [4]
    assert str(deleted_path) not in state.labels_by_file
    assert state.next_label == 5


def test_update_semantic_corpus_without_changes(tmp_path: Path) -> None:
    """Test skipping embeddings when nothing changed."""

    corpus_directory = tmp_path / "corpus"
    corpus_directory.mkdir()

    source_path = corpus_directory / "example.py"
    source_path.write_text("unchanged content", encoding="utf-8")

    index_directory = tmp_path / "semantic_index"
    embeddings = DeterministicEmbeddings()

    original_index = index_semantic_corpus(
        corpus_directory=corpus_directory,
        index_directory=index_directory,
        embedding_model_id="test-embeddings",
        embedding_model=embeddings,
    )

    embeddings.embedded_texts.clear()

    updated_index, changes = update_semantic_corpus(
        corpus_directory=corpus_directory,
        index_directory=index_directory,
        embedding_model=embeddings,
    )

    assert changes.has_changes is False
    assert changes.added == []
    assert changes.modified == []
    assert changes.deleted == []
    assert changes.unchanged == [str(source_path)]
    assert embeddings.embedded_texts == []
    assert updated_index.active_count == original_index.active_count
