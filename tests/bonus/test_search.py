from pathlib import Path

import pytest
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma

from conftest import RecordingEmbeddings
from src.bonus.hybrid import retrieve_hybrid_chunks
from src.bonus.indexing import (
    chunk_to_document,
    index_semantic_corpus,
    load_semantic_index,
)
from src.bonus.search import retrieve_semantic_chunks
from src.chunking.models import Chunk
from src.indexing.bm25 import build_bm25_index


def test_semantic_search_ranks_by_cosine_similarity(
    corpus: Path, tmp_path: Path, embeddings: RecordingEmbeddings,
) -> None:
    (corpus / "a.txt").write_text("banana banana", encoding="utf-8")
    (corpus / "b.txt").write_text("apple apple", encoding="utf-8")
    directory = tmp_path / "index"
    index_semantic_corpus(corpus, directory, embedding_model=embeddings)
    store = load_semantic_index(directory, embeddings)
    chunks = retrieve_semantic_chunks(store, "apple apple", 100)
    assert [chunk.text for chunk in chunks] == ["apple apple", "banana banana"]
    assert embeddings.queries == ["apple apple"]


def test_hybrid_fuses_shared_sources_and_preserves_distinct_offsets(
    tmp_path: Path, embeddings: RecordingEmbeddings,
) -> None:
    # Three identical texts remain separate sources; overlap between rankings
    # is returned once. Document content alone is not a valid deduplication key.
    chunks = [
        Chunk(file_path="a.txt", first_character_index=0,
              last_character_index=5, text="apple"),
        Chunk(file_path="a.txt", first_character_index=10,
              last_character_index=15, text="apple"),
        Chunk(file_path="b.txt", first_character_index=0,
              last_character_index=5, text="apple"),
    ]
    lexical = build_bm25_index(chunks)
    semantic = load_semantic_index(tmp_path / "index", embeddings, create=True)
    semantic.add_documents([chunk_to_document(chunk) for chunk in chunks])
    result = retrieve_hybrid_chunks(lexical, semantic, "apple", 10)
    assert len(result) == 3
    assert {tuple(c.model_dump().values()) for c in result} == {
        tuple(c.model_dump().values()) for c in chunks
    }
    assert len(retrieve_hybrid_chunks(lexical, semantic, "apple", 1)) == 1


def test_hybrid_handles_no_lexical_matches(
    tmp_path: Path, embeddings: RecordingEmbeddings,
) -> None:
    chunks = [Chunk(file_path="a.txt", first_character_index=0,
                    last_character_index=5, text="apple")]
    semantic = load_semantic_index(tmp_path / "index", embeddings, create=True)
    semantic.add_documents([chunk_to_document(chunks[0])])
    result = retrieve_hybrid_chunks(build_bm25_index(chunks), semantic, "banana", 5)
    assert result == chunks


def test_shared_result_beats_results_in_only_one_ranking(
    tmp_path: Path, embeddings: RecordingEmbeddings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def chunk(path: str, text: str) -> Chunk:
        return Chunk(file_path=path, text=text, first_character_index=0,
                     last_character_index=len(text))

    shared = chunk("shared.txt", "apple")
    lexical_only = chunk("lexical.txt", "rare word")
    semantic_first = chunk("semantic.txt", "apple apple")
    semantic_last = chunk("last.txt", "banana banana")

    def lexical_search(index: object, query: str, k: int) -> list[Chunk]:
        return [shared, lexical_only]

    monkeypatch.setattr("src.bonus.hybrid.retrieve_chunks", lexical_search)
    semantic = load_semantic_index(tmp_path / "index", embeddings, create=True)
    semantic.add_documents([
        chunk_to_document(c) for c in (semantic_first, shared, semantic_last)
    ])
    lexical = build_bm25_index([shared, lexical_only])
    assert retrieve_semantic_chunks(semantic, "apple apple", 1) == [semantic_first]
    assert retrieve_hybrid_chunks(lexical, semantic, "apple apple", 1) == [shared]


@pytest.mark.parametrize("query,k", [("apple", 0), ("  ", 5)])
def test_no_op_searches_do_not_embed_queries(
    tmp_path: Path, embeddings: RecordingEmbeddings, query: str, k: int,
) -> None:
    semantic = load_semantic_index(tmp_path / "index", embeddings, create=True)
    chunk = Chunk(file_path="a.txt", first_character_index=0,
                  last_character_index=5, text="apple")
    lexical = build_bm25_index([chunk])
    assert retrieve_semantic_chunks(semantic, query, k) == []
    assert retrieve_hybrid_chunks(lexical, semantic, query, k) == []
    assert embeddings.queries == []


def test_negative_k_is_rejected(
    tmp_path: Path, embeddings: RecordingEmbeddings,
) -> None:
    semantic = load_semantic_index(tmp_path / "index", embeddings, create=True)
    chunk = Chunk(file_path="a.txt", first_character_index=0,
                  last_character_index=5, text="apple")
    with pytest.raises(ValueError, match="negative"):
        retrieve_semantic_chunks(semantic, "apple", -1)
    with pytest.raises(ValueError, match="negative"):
        retrieve_hybrid_chunks(build_bm25_index([chunk]), semantic, "apple", -1)


def test_missing_or_legacy_index_is_not_silently_created(
    tmp_path: Path, embeddings: RecordingEmbeddings,
) -> None:
    directory = tmp_path / "index"
    with pytest.raises(FileNotFoundError, match="Run index_semantic"):
        load_semantic_index(directory, embeddings)
    assert not directory.exists()
    directory.mkdir()
    (directory / "hnsw.bin").write_bytes(b"legacy graph")
    with pytest.raises(FileNotFoundError, match="Run index_semantic"):
        load_semantic_index(directory, embeddings)
    assert not (directory / "chroma.sqlite3").exists()


def test_search_does_not_create_a_missing_collection(
    tmp_path: Path, embeddings: RecordingEmbeddings,
) -> None:
    directory = tmp_path / "index"
    other = Chroma(
        collection_name="other_collection", persist_directory=str(directory),
        client_settings=Settings(anonymized_telemetry=False),
    )
    with pytest.raises(NotFoundError, match="rag_chunks"):
        load_semantic_index(directory, embeddings)
    assert other.get()["ids"] == []
