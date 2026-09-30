from pathlib import Path

from langchain_core.embeddings import Embeddings

from src.bonus.indexing import index_semantic_corpus
from src.bonus.storage import (
    HNSW_GRAPH_FILENAME,
    HNSW_METADATA_FILENAME,
    load_hnsw_index,
)


class FakeCorpusEmbeddingModel(Embeddings):
    """Provide embeddings for corpus pipeline tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed test documents."""

        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a test query."""

        return [float(len(text)), 1.0]


def test_index_semantic_corpus_builds_and_saves(tmp_path: Path) -> None:
    """Test the complete semantic indexing pipeline."""

    corpus_directory = tmp_path / "raw"
    corpus_directory.mkdir()

    source_path = corpus_directory / "example.py"
    source_path.write_text("def add(a, b):\n" "    return a + b\n", encoding="utf-8")

    index_directory = tmp_path / "semantic_index"

    index = index_semantic_corpus(
        corpus_directory=corpus_directory,
        index_directory=index_directory,
        max_chunk_size=30,
        batch_size=2,
        embedding_model=FakeCorpusEmbeddingModel(),
    )

    loaded_index = load_hnsw_index(index_directory)

    assert (index_directory / HNSW_GRAPH_FILENAME).exists()
    assert (index_directory / HNSW_METADATA_FILENAME).exists()
    assert len(index.chunks) > 0
    assert loaded_index.chunks == index.chunks
    assert loaded_index.dimension == index.dimension
    assert loaded_index.hnsw.get_current_count() == len(index.chunks)
