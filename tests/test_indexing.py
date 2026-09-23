from src.indexing.tokenization import tokenize
from src.chunking.models import Chunk
from src.indexing.bm25 import build_bm25_index
import pytest
from pathlib import Path

import joblib

from src.indexing.storage import (
    load_bm25_index,
    save_bm25_index,
)


def test_tokenize_splits_code_identifiers() -> None:
    """Test splitting acronyms, snake case, and camel case."""

    text = (
        "HTTPServer uses max_chunk_size "
        "and PythonCodeTextSplitter."
    )

    assert tokenize(text) == [
        "http",
        "server",
        "uses",
        "max",
        "chunk",
        "size",
        "and",
        "python",
        "code",
        "text",
        "splitter",
    ]


def test_tokenize_normalizes_case_and_punctuation() -> None:
    """Test lowercase conversion and punctuation removal."""

    assert tokenize("RAG, Retrieval! Version2Parser.") == [
        "rag",
        "retrieval",
        "version2",
        "parser",
    ]


def test_tokenize_empty_text() -> None:
    """Test tokenizing an empty string."""

    assert tokenize("") == []


def test_build_bm25_index_preserves_chunks() -> None:
    """Test building a BM25 index with aligned chunks."""

    chunks = [
        Chunk(
            file_path="data/raw/lora.py",
            first_character_index=0,
            last_character_index=30,
            text="LoRA reduces GPU memory usage.",
        ),
        Chunk(
            file_path="data/raw/cuda.md",
            first_character_index=0,
            last_character_index=37,
            text="CUDA kernels improve inference speed.",
        ),
    ]

    index = build_bm25_index(chunks)

    assert index.chunks == chunks
    assert index.chunks is not chunks
    assert index.bm25.corpus_size == 2
    assert len(index.bm25.doc_len) == 2


def test_build_bm25_index_rejects_empty_chunks() -> None:
    """Test rejecting an empty chunk collection."""

    with pytest.raises(
        ValueError,
        match="Cannot build an index without chunks",
    ):
        build_bm25_index([])


def test_build_bm25_index_rejects_chunks_without_tokens() -> None:
    """Test rejecting chunks containing no searchable tokens."""

    chunks = [
        Chunk(
            file_path="---",
            first_character_index=0,
            last_character_index=3,
            text="!!!",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Cannot build an index without searchable tokens",
    ):
        build_bm25_index(chunks)


def test_save_and_load_bm25_index(
    tmp_path: Path,
) -> None:
    """Test saving and restoring a BM25 index."""

    chunks = [
        Chunk(
            file_path="data/raw/lora.py",
            first_character_index=0,
            last_character_index=30,
            text="LoRA reduces GPU memory usage.",
        ),
    ]

    index = build_bm25_index(chunks)
    index_path = tmp_path / "nested" / "bm25_index.joblib"

    save_bm25_index(index, index_path)
    loaded_index = load_bm25_index(index_path)

    assert index_path.exists()
    assert loaded_index.chunks == index.chunks
    assert loaded_index.bm25.corpus_size == 1
    assert loaded_index.bm25.doc_len == index.bm25.doc_len


def test_load_bm25_index_rejects_wrong_object(
    tmp_path: Path,
) -> None:
    """Test rejecting a file that is not a BM25 index."""

    index_path = tmp_path / "wrong_object.joblib"
    joblib.dump(["not", "an", "index"], index_path)

    with pytest.raises(
        TypeError,
        match="File does not contain a BM25Index",
    ):
        load_bm25_index(index_path)
