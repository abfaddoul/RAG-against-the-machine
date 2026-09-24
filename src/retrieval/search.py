from src.chunking.models import Chunk
from src.indexing.bm25 import BM25Index
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)
from src.retrieval.bm25 import retrieve_chunks
from tqdm import tqdm


def chunk_to_source(chunk: Chunk) -> MinimalSource:
    """Convert a retrieved chunk into a minimal source."""

    return MinimalSource(
        file_path=chunk.file_path,
        first_character_index=(
            chunk.first_character_index
        ),
        last_character_index=(
            chunk.last_character_index
        ),
    )


def retrieve_question(
    index: BM25Index,
    question: UnansweredQuestion,
    k: int,
) -> MinimalSearchResults:
    """Retrieve the top-k sources for one question."""

    retrieved_chunks = retrieve_chunks(
        index=index,
        query=question.question,
        k=k,
    )

    retrieved_sources: list[MinimalSource] = []

    for chunk in retrieved_chunks:
        source = chunk_to_source(chunk)
        retrieved_sources.append(source)

    return MinimalSearchResults(
        question_id=question.question_id,
        question=question.question,
        retrieved_sources=retrieved_sources,
    )


def retrieve_dataset(
    index: BM25Index,
    dataset: RagDataset,
    k: int,
) -> StudentSearchResults:
    """Retrieve top-k sources for every dataset question."""

    if k < 0:
        raise ValueError("k cannot be negative")

    search_results: list[MinimalSearchResults] = []

    for question in tqdm(
        dataset.rag_questions,
        desc="Searching questions",
        unit="question",
    ):
        result = retrieve_question(
            index=index,
            question=question,
            k=k,
        )
        search_results.append(result)

    return StudentSearchResults(
        search_results=search_results,
        k=k,
    )
