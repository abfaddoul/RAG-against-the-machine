import pytest

from src.chunking.models import Chunk
from src.indexing.bm25 import (
    BM25Index,
    build_bm25_index,
)
from src.retrieval.bm25 import retrieve_chunks
from src.models import (
    MinimalSource,
    RagDataset,
    UnansweredQuestion,
)
from src.retrieval.search import (
    chunk_to_source,
    retrieve_dataset,
    retrieve_question,
)


def build_retrieval_index() -> BM25Index:
    """Build a small BM25 index for retrieval tests."""

    chunks = [
        Chunk(
            file_path="data/raw/gpu.py",
            first_character_index=0,
            last_character_index=30,
            text="GPU memory usage optimization.",
        ),
        Chunk(
            file_path="data/raw/memory.md",
            first_character_index=0,
            last_character_index=22,
            text="GPU memory allocation.",
        ),
        Chunk(
            file_path="data/raw/cuda.py",
            first_character_index=0,
            last_character_index=27,
            text="CUDA kernels improve speed.",
        ),
        Chunk(
            file_path="data/raw/markdown.md",
            first_character_index=0,
            last_character_index=30,
            text="Markdown structures documents.",
        ),
        Chunk(
            file_path="data/raw/python.py",
            first_character_index=0,
            last_character_index=31,
            text="Python functions return values.",
        ),
    ]

    return build_bm25_index(chunks)


def test_retrieve_chunks_returns_best_match() -> None:
    """Test returning the highest-scoring BM25 chunk."""

    index = build_retrieval_index()

    retrieved_chunks = retrieve_chunks(
        index=index,
        query="GPU memory usage",
        k=1,
    )

    assert retrieved_chunks == [index.chunks[0]]


def test_retrieve_chunks_returns_empty_without_match() -> None:
    """Test returning no chunks for an unknown query."""

    index = build_retrieval_index()

    retrieved_chunks = retrieve_chunks(
        index=index,
        query="quantum banana",
        k=3,
    )

    assert retrieved_chunks == []


def test_retrieve_chunks_handles_k_boundaries() -> None:
    """Test zero and negative retrieval limits."""

    index = build_retrieval_index()

    assert retrieve_chunks(
        index=index,
        query="GPU memory",
        k=0,
    ) == []

    with pytest.raises(
        ValueError,
        match="k cannot be negative",
    ):
        retrieve_chunks(
            index=index,
            query="GPU memory",
            k=-1,
        )


def test_chunk_to_source_removes_chunk_text() -> None:
    """Test converting a chunk into source coordinates."""

    index = build_retrieval_index()
    chunk = index.chunks[0]

    source = chunk_to_source(chunk)

    assert source == MinimalSource(
        file_path="data/raw/gpu.py",
        first_character_index=0,
        last_character_index=30,
    )


def test_retrieve_question_returns_minimal_result() -> None:
    """Test retrieving sources for one question."""

    index = build_retrieval_index()
    question = UnansweredQuestion(
        question_id="q1",
        question="How is GPU memory usage optimized?",
    )

    result = retrieve_question(
        index=index,
        question=question,
        k=1,
    )

    assert result.question_id == "q1"
    assert result.question == question.question
    assert result.retrieved_sources == [
        MinimalSource(
            file_path="data/raw/gpu.py",
            first_character_index=0,
            last_character_index=30,
        ),
    ]


def test_retrieve_dataset_returns_batch_results() -> None:
    """Test retrieving sources for every dataset question."""

    index = build_retrieval_index()
    dataset = RagDataset(
        rag_questions=[
            UnansweredQuestion(
                question_id="q1",
                question="How is GPU memory usage optimized?",
            ),
            UnansweredQuestion(
                question_id="q2",
                question="What is quantum banana?",
            ),
        ]
    )

    result = retrieve_dataset(
        index=index,
        dataset=dataset,
        k=1,
    )

    assert result.k == 1
    assert len(result.search_results) == 2
    assert result.search_results[0].question_id == "q1"
    assert len(
        result.search_results[0].retrieved_sources
    ) == 1
    assert result.search_results[1].question_id == "q2"
    assert result.search_results[1].retrieved_sources == []


def test_retrieve_empty_dataset_rejects_negative_k() -> None:
    """Test validating k even when the dataset is empty."""

    index = build_retrieval_index()
    dataset = RagDataset(rag_questions=[])

    with pytest.raises(
        ValueError,
        match="k cannot be negative",
    ):
        retrieve_dataset(
            index=index,
            dataset=dataset,
            k=-1,
        )
