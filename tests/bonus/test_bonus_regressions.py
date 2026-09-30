from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from src.bonus.incremental import (
    IncrementalState,
    build_incremental_state,
    hash_corpus_files,
    load_incremental_state,
    save_incremental_state,
    update_semantic_corpus,
)
from src.bonus.indexing import index_semantic_corpus
from src.bonus.search import retrieve_hnsw_chunks
from src.bonus.storage import HNSWMetadata, load_hnsw_index, save_hnsw_index


class RecordingEmbeddings(Embeddings):
    """Record which documents are embedded without downloading a model."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Record the documents and return deterministic vectors."""
        self.texts.extend(texts)
        return [[1.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a deterministic query vector."""
        return [1.0, float(len(text))]


def test_missing_state_does_not_accept_a_stale_index(tmp_path: Path) -> None:
    """Reject missing history instead of declaring changed files unchanged."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "a.py"
    source.write_text("original", encoding="utf-8")
    directory = tmp_path / "index"
    embeddings = RecordingEmbeddings()
    index_semantic_corpus(corpus, directory, embedding_model=embeddings)
    state_path = directory / "incremental_state.joblib"
    state_path.unlink()
    source.write_text("modified", encoding="utf-8")
    embeddings.texts.clear()

    with pytest.raises(FileNotFoundError, match="Run index_semantic"):
        update_semantic_corpus(corpus, directory, embedding_model=embeddings)

    assert not state_path.exists()
    assert embeddings.texts == []
    assert load_hnsw_index(directory).chunks[0].text == "original"


def test_repeated_updates_deletion_and_readdition(tmp_path: Path) -> None:
    """Keep active labels correct over several saved update cycles."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "a.py"
    source.write_text("original", encoding="utf-8")
    directory = tmp_path / "index"
    embeddings = RecordingEmbeddings()
    index_semantic_corpus(corpus, directory, embedding_model=embeddings)

    for text in ("first change", "second change"):
        source.write_text(text, encoding="utf-8")
        embeddings.texts.clear()
        index, _ = update_semantic_corpus(
            corpus, directory, embedding_model=embeddings
        )
        assert embeddings.texts == [text]
        assert index.active_count == 1
        assert retrieve_hnsw_chunks(index, [1.0, 1.0], 10)[0].text == text

    source.unlink()
    embeddings.texts.clear()
    index, _ = update_semantic_corpus(corpus, directory, embedding_model=embeddings)
    assert index.active_count == 0
    assert embeddings.texts == []
    assert retrieve_hnsw_chunks(load_hnsw_index(directory), [1.0, 1.0], 5) == []

    source.write_text("restored", encoding="utf-8")
    update_semantic_corpus(corpus, directory, embedding_model=embeddings)
    restored = load_hnsw_index(directory)
    assert restored.active_count == 1
    assert retrieve_hnsw_chunks(restored, [1.0, 1.0], 5)[0].text == "restored"


@pytest.mark.parametrize("content", [b" \n", b"\xff\xfe"])
def test_modified_file_that_becomes_unsearchable(
    tmp_path: Path, content: bytes
) -> None:
    """Remove old vectors when a file becomes empty or invalid UTF-8."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "a.py"
    source.write_text("original", encoding="utf-8")
    directory = tmp_path / "index"
    embeddings = RecordingEmbeddings()
    index_semantic_corpus(corpus, directory, embedding_model=embeddings)
    embeddings.texts.clear()

    source.write_bytes(content)
    index, changes = update_semantic_corpus(
        corpus, directory, embedding_model=embeddings
    )
    assert changes.modified == [str(source)]
    assert index.active_count == 0
    assert embeddings.texts == []
    _, repeated = update_semantic_corpus(
        corpus, directory, embedding_model=embeddings
    )
    assert not repeated.has_changes


def test_old_joblib_class_paths_remain_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load metadata and state serialized with the original module names."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.py").write_text("original", encoding="utf-8")
    directory = tmp_path / "index"
    index = index_semantic_corpus(
        corpus, directory, embedding_model=RecordingEmbeddings()
    )
    state = build_incremental_state(hash_corpus_files(corpus), index)
    state_path = directory / "incremental_state.joblib"
    with monkeypatch.context() as legacy:
        legacy.setattr(HNSWMetadata, "__module__", "src.indexing.hnsw_storage")
        legacy.setattr(IncrementalState, "__module__", "src.indexing.incremental")
        save_hnsw_index(index, directory)
        save_incremental_state(state, state_path)

    assert load_hnsw_index(directory).chunks == index.chunks
    assert load_incremental_state(state_path) == state
