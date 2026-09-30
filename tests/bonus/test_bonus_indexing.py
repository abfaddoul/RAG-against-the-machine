import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from langchain_core.indexing import IndexingResult

from conftest import RecordingEmbeddings
from src.bonus.indexing import index_semantic_corpus, load_semantic_index
from src.bonus.search import retrieve_semantic_chunks


def test_persistent_search_preserves_exact_sources(
    corpus: Path, tmp_path: Path, embeddings: RecordingEmbeddings,
) -> None:
    text = "banana\n\n" + "apple tree " * 25
    source = corpus / "a.txt"
    source.write_text(text, encoding="utf-8")
    directory = tmp_path / "index"
    result = index_semantic_corpus(
        corpus, directory, max_chunk_size=80, embedding_model=embeddings,
    )
    store = load_semantic_index(directory, embeddings)
    chunks = retrieve_semantic_chunks(store, "apple", 100)
    assert len(chunks) == result['num_added']
    assert result['num_added'] > 1
    for chunk in chunks:
        assert chunk.file_path == str(source)
        original = text[chunk.first_character_index:chunk.last_character_index]
        assert original == chunk.text
        assert len(chunk.text) <= 80


def test_unchanged_update_never_reembeds(
    corpus: Path, tmp_path: Path, embeddings: RecordingEmbeddings,
) -> None:
    (corpus / "a.txt").write_text("apple", encoding="utf-8")
    directory = tmp_path / "index"
    index_semantic_corpus(corpus, directory, embedding_model=embeddings)
    embeddings.texts.clear()
    for _ in range(2):
        result = index_semantic_corpus(
            corpus, directory, embedding_model=embeddings, incremental=True,
        )
        assert result == IndexingResult(
            num_added=0, num_updated=0, num_deleted=0, num_skipped=1,
        )
    assert embeddings.texts == []
    assert len(load_semantic_index(directory, embeddings).get()["ids"]) == 1


def test_update_only_embeds_changed_and_added_chunks(
    corpus: Path, tmp_path: Path, embeddings: RecordingEmbeddings,
) -> None:
    (corpus / "a.txt").write_text("apple", encoding="utf-8")
    (corpus / "b.txt").write_text("banana", encoding="utf-8")
    directory = tmp_path / "index"
    index_semantic_corpus(corpus, directory, embedding_model=embeddings)
    old_ids = load_semantic_index(directory, embeddings).get(where={
        "file_path": str(corpus / "b.txt"),
    })["ids"]
    (corpus / "a.txt").write_text("apple changed", encoding="utf-8")
    (corpus / "c.txt").write_text("new apple", encoding="utf-8")
    embeddings.texts.clear()
    result = index_semantic_corpus(
        corpus, directory, embedding_model=embeddings, incremental=True,
    )
    assert sorted(embeddings.texts) == ["apple changed", "new apple"]
    assert result['num_added'] == 2
    assert result['num_deleted'] == 1
    assert result['num_skipped'] == 1
    store = load_semantic_index(directory, embeddings)
    assert store.get(where={"file_path": str(corpus / "b.txt")})["ids"] == old_ids
    assert sorted(store.get()["documents"]) == ["apple changed", "banana", "new apple"]


def test_deletion_and_readdition_with_an_empty_index(
    corpus: Path, tmp_path: Path, embeddings: RecordingEmbeddings,
) -> None:
    source = corpus / "a.txt"
    source.write_text("apple", encoding="utf-8")
    directory = tmp_path / "index"
    index_semantic_corpus(corpus, directory, embedding_model=embeddings)
    source.unlink()
    embeddings.texts.clear()
    result = index_semantic_corpus(
        corpus, directory, embedding_model=embeddings, incremental=True,
    )
    assert result['num_deleted'] == 1
    assert embeddings.texts == []
    store = load_semantic_index(directory, embeddings)
    assert retrieve_semantic_chunks(store, "apple") == []
    source.write_text("banana restored", encoding="utf-8")
    index_semantic_corpus(
        corpus, directory, embedding_model=embeddings, incremental=True,
    )
    assert embeddings.texts == ["banana restored"]
    assert load_semantic_index(directory, embeddings).get()["documents"] == [
        "banana restored",
    ]


@pytest.mark.parametrize("content", [b" \n", b"\xff\xfe"])
def test_unsearchable_file_loses_its_old_chunks(
    corpus: Path, tmp_path: Path, embeddings: RecordingEmbeddings, content: bytes,
) -> None:
    source = corpus / "a.txt"
    source.write_text("apple", encoding="utf-8")
    directory = tmp_path / "index"
    index_semantic_corpus(corpus, directory, embedding_model=embeddings)
    embeddings.texts.clear()
    source.write_bytes(content)
    result = index_semantic_corpus(
        corpus, directory, embedding_model=embeddings, incremental=True,
    )
    assert result['num_deleted'] == 1
    assert embeddings.texts == []
    assert load_semantic_index(directory, embeddings).get()["ids"] == []


def test_full_rebuild_can_recover_missing_records_without_stale_chunks(
    corpus: Path, tmp_path: Path, embeddings: RecordingEmbeddings,
) -> None:
    source = corpus / "a.txt"
    source.write_text("apple", encoding="utf-8")
    directory = tmp_path / "index"
    index_semantic_corpus(corpus, directory, embedding_model=embeddings)
    (directory / "records.sqlite3").unlink()
    source.write_text("banana", encoding="utf-8")
    embeddings.texts.clear()
    with pytest.raises(FileNotFoundError, match="Run index_semantic"):
        index_semantic_corpus(
            corpus, directory, embedding_model=embeddings, incremental=True,
        )
    assert embeddings.texts == []
    assert load_semantic_index(directory, embeddings).get()["documents"] == ["apple"]
    index_semantic_corpus(corpus, directory, embedding_model=embeddings)
    assert load_semantic_index(directory, embeddings).get()["documents"] == ["banana"]


def test_full_rebuild_reembeds_and_removes_deleted_files(
    corpus: Path, tmp_path: Path, embeddings: RecordingEmbeddings,
) -> None:
    (corpus / "a.txt").write_text("apple", encoding="utf-8")
    (corpus / "b.txt").write_text("banana", encoding="utf-8")
    directory = tmp_path / "index"
    index_semantic_corpus(corpus, directory, embedding_model=embeddings)
    (corpus / "b.txt").unlink()
    embeddings.texts.clear()
    index_semantic_corpus(corpus, directory, embedding_model=embeddings)
    assert embeddings.texts == ["apple"]
    assert load_semantic_index(directory, embeddings).get()["documents"] == ["apple"]


@pytest.mark.parametrize("option", ["batch_size", "max_chunk_size"])
def test_invalid_options_do_not_write_an_index(
    corpus: Path, tmp_path: Path, embeddings: RecordingEmbeddings, option: str,
) -> None:
    directory = tmp_path / "index"
    with pytest.raises(ValueError, match="must be positive"):
        index_semantic_corpus(
            corpus, directory, embedding_model=embeddings,
            batch_size=0 if option == "batch_size" else 32,
            max_chunk_size=0 if option == "max_chunk_size" else 2000,
        )
    assert not directory.exists()


def test_missing_corpus_does_not_clear_existing_index(
    corpus: Path, tmp_path: Path, embeddings: RecordingEmbeddings,
) -> None:
    (corpus / "a.txt").write_text("apple", encoding="utf-8")
    directory = tmp_path / "index"
    index_semantic_corpus(corpus, directory, embedding_model=embeddings)
    with pytest.raises(FileNotFoundError):
        index_semantic_corpus(corpus / "missing", directory, embedding_model=embeddings)
    assert load_semantic_index(directory, embeddings).get()["documents"] == ["apple"]


def test_persistence_in_another_process(
    corpus: Path, tmp_path: Path, embeddings: RecordingEmbeddings,
) -> None:
    (corpus / "a.txt").write_text("apple", encoding="utf-8")
    directory = tmp_path / "index"
    index_semantic_corpus(corpus, directory, embedding_model=embeddings)
    script = """
import json, sys
from conftest import RecordingEmbeddings
from src.bonus.indexing import load_semantic_index
from src.bonus.search import retrieve_semantic_chunks
store = load_semantic_index(sys.argv[1], RecordingEmbeddings())
print(json.dumps([c.model_dump() for c in retrieve_semantic_chunks(store, 'apple')]))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd()), str(Path(__file__).parent)])
    result = subprocess.run(
        [sys.executable, "-c", script, str(directory)],
        env=env, capture_output=True, text=True, check=True,
    )
    chunks = json.loads(result.stdout)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "apple"
    assert chunks[0]["file_path"] == str(corpus / "a.txt")
