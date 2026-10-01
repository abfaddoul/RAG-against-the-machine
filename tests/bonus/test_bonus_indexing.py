from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from src.bonus.indexing import (
    index_semantic_corpus,
    load_semantic_index,
)


class FakeEmbeddings(Embeddings):
    """Provide deterministic vectors without downloading a model."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [
            float(len(text) + 1),
            float(text.count("a") + 1),
            1.0,
        ]


def test_index_persists_documents(tmp_path: Path) -> None:
    """Stored chunks remain available when the collection is reopened."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.py").write_text("alpha", encoding="utf-8")
    (corpus / "b.py").write_text("beta", encoding="utf-8")
    directory = tmp_path / "index"
    model = FakeEmbeddings()

    count = index_semantic_corpus(
        corpus, directory, embedding_model=model,
    )
    store = load_semantic_index(directory, embedding_model=model)
    data = store.get()

    assert count == 2
    assert set(data["documents"] or []) == {"alpha", "beta"}
    assert (directory / "chroma.sqlite3").is_file()
    assert not (directory / "records.sqlite3").exists()


def test_rebuild_replaces_old_documents(tmp_path: Path) -> None:
    """A full rebuild removes deleted and outdated source content."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "a.py"
    deleted = corpus / "b.py"
    source.write_text("original", encoding="utf-8")
    deleted.write_text("removed", encoding="utf-8")
    directory = tmp_path / "index"
    model = FakeEmbeddings()

    index_semantic_corpus(corpus, directory, embedding_model=model)

    source.write_text("replacement", encoding="utf-8")
    deleted.unlink()
    count = index_semantic_corpus(
        corpus, directory, embedding_model=model,
    )
    store = load_semantic_index(directory, embedding_model=model)

    assert count == 1
    assert store.get()["documents"] == ["replacement"]


def test_index_uses_batches(tmp_path: Path) -> None:
    """Embedding calls respect the requested batch size."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for number in range(5):
        (corpus / f"{number}.py").write_text(
            f"value = {number}", encoding="utf-8",
        )
    model = FakeEmbeddings()

    count = index_semantic_corpus(
        corpus,
        tmp_path / "index",
        batch_size=2,
        embedding_model=model,
    )

    assert count == 5
    assert model.batch_sizes == [2, 2, 1]


def test_empty_corpus_clears_collection(tmp_path: Path) -> None:
    """Rebuilding an empty corpus leaves an empty collection."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "a.py"
    source.write_text("alpha", encoding="utf-8")
    directory = tmp_path / "index"
    model = FakeEmbeddings()

    index_semantic_corpus(corpus, directory, embedding_model=model)
    source.unlink()

    count = index_semantic_corpus(
        corpus, directory, embedding_model=model,
    )
    store = load_semantic_index(directory, embedding_model=model)

    assert count == 0
    assert store.get()["ids"] == []


def test_missing_index_raises(tmp_path: Path) -> None:
    """Searching requires a previously created index."""
    with pytest.raises(FileNotFoundError, match="Semantic index missing"):
        load_semantic_index(
            tmp_path / "missing",
            embedding_model=FakeEmbeddings(),
        )
