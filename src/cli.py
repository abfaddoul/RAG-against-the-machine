import sys
from pathlib import Path

import fire

from src.chunking.models import Chunk
from src.evaluation.recall import calculate_recall_at_k
from src.generation.answer import generate_answer, generate_answers
from src.generation.model import DEFAULT_MAX_NEW_TOKENS, load_chat_model
from src.query_cache import cached_search, clear_query_cache
from src.retrieval.bm25 import retrieve_chunks
from src.json_io import load_json_model, save_json_model
from src.indexing.storage import load_bm25_index
from src.indexing.pipeline import (
    DEFAULT_CORPUS_DIRECTORY,
    DEFAULT_INDEX_PATH,
    DEFAULT_MAX_CHUNK_SIZE,
    index_corpus,
)
from src.models import (
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from src.retrieval.search import (
    chunk_to_source,
    retrieve_dataset,
    retrieve_question,
)


def print_chunks(chunks: list[Chunk]) -> None:
    """Print retrieved chunks."""

    for rank, chunk in enumerate(chunks, start=1):
        print(f"\n[Result {rank}]")
        print(f"File: {chunk.file_path}")
        print(
            f"Characters: {chunk.first_character_index}-"
            f"{chunk.last_character_index}"
        )
        print(chunk.text)


def index(
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    corpus_directory: str = str(DEFAULT_CORPUS_DIRECTORY),
    index_path: str = str(DEFAULT_INDEX_PATH),
) -> None:
    """Build and save the BM25 index."""

    clear_query_cache()

    bm25_index = index_corpus(
        corpus_directory=corpus_directory,
        index_path=index_path,
        max_chunk_size=max_chunk_size,
    )

    print(f"Indexed {len(bm25_index.chunks)} chunks.")
    print(f"Saved index to: {index_path}")


def search(
    query: str,
    k: int = 5,
    index_path: str = str(DEFAULT_INDEX_PATH),
    output_path: str = "data/output/search_results/single_query.json",
) -> None:
    """Search one question and save its sources as JSON."""

    chunks = cached_search(
        mode="bm25",
        query=query,
        k=k,
        index_paths=(index_path,),
        search=lambda: retrieve_chunks(
            index=load_bm25_index(index_path),
            query=query,
            k=k,
        ),
    )

    question = UnansweredQuestion(question=query)

    search_result = MinimalSearchResults(
        question_id=question.question_id,
        question=question.question,
        retrieved_sources=[chunk_to_source(chunk) for chunk in chunks],
    )

    save_json_model(
        model=StudentSearchResults(
            search_results=[search_result],
            k=k,
        ),
        path=output_path,
    )

    print(f"Saved search results to {output_path}")

    if not chunks:
        print("No relevant sources found.")
        return


def search_dataset(
    dataset_path: str,
    k: int,
    save_directory: str,
    index_path: str = str(DEFAULT_INDEX_PATH),
) -> None:
    """Search every question contained in a dataset."""

    dataset = load_json_model(
        path=dataset_path,
        model_class=RagDataset,
    )

    bm25_index = load_bm25_index(index_path)

    student_search_results = retrieve_dataset(
        index=bm25_index,
        dataset=dataset,
        k=k,
    )

    output_path = (
        Path(save_directory)
        / Path(dataset_path).name
    )

    save_json_model(
        model=student_search_results,
        path=output_path,
    )

    print(
        "Saved student_search_results to "
        f"{output_path}"
    )


def evaluate(
    student_search_results_path: str,
    dataset_path: str,
) -> None:
    """Evaluate retrieval recall against an answered dataset."""

    student_results = load_json_model(
        path=student_search_results_path,
        model_class=StudentSearchResults,
    )
    dataset = load_json_model(
        path=dataset_path,
        model_class=RagDataset,
    )

    recall_ks = sorted({
        1,
        3,
        5,
        10,
        student_results.k,
    })

    for k in recall_ks:
        if k > student_results.k:
            continue

        recall = calculate_recall_at_k(
            student_results=student_results,
            dataset=dataset,
            k=k,
        )

        print(
            f"Recall@{k}: {recall:.3f} "
            f"({recall * 100:.1f}%)"
        )


def answer(
    query: str,
    k: int = 5,
    index_path: str = str(DEFAULT_INDEX_PATH),
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    output_path: str = "data/output/search_results_and_answer/single_query.json",
) -> None:
    """Answer one question and save its answer and sources as JSON."""

    bm25_index = load_bm25_index(index_path)

    question = UnansweredQuestion(
        question=query,
    )

    search_result = retrieve_question(
        index=bm25_index,
        question=question,
        k=k,
    )

    model = load_chat_model(
        max_new_tokens=max_new_tokens,
    )

    generated_answer = generate_answer(
        search_result=search_result,
        model=model,
    )

    save_json_model(
        model=StudentSearchResultsAndAnswer(
            search_results=[generated_answer],
            k=k,
        ),
        path=output_path,
    )

    print(f"Saved generated answer to {output_path}")

    print("\nAnswer:")
    print(generated_answer.answer)

    print("\nSources:")

    for source in generated_answer.retrieved_sources:
        print(
            f"- {source.file_path} "
            f"({source.first_character_index}-"
            f"{source.last_character_index})"
        )


def answer_dataset(
    student_search_results_path: str,
    save_directory: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> None:
    """Generate answers for retrieved dataset results."""

    student_results = load_json_model(
        path=student_search_results_path,
        model_class=StudentSearchResults,
    )

    model = load_chat_model(
        max_new_tokens=max_new_tokens,
    )

    results_with_answers = generate_answers(
        student_results=student_results,
        model=model,
    )

    output_path = (
        Path(save_directory)
        / Path(student_search_results_path).name
    )

    save_json_model(
        model=results_with_answers,
        path=output_path,
    )

    print(
        f"Saved generated answers to {output_path}"
    )


def run_cli() -> None:
    """Run the command-line interface."""
    if len(sys.argv) > 1 and sys.argv[1] in {
        "index_semantic", "search_semantic", "search_hybrid",
    }:
        from src.bonus.cli import run_cli as run_bonus_cli

        run_bonus_cli()
        return

    fire.Fire({
        "index": index,
        "search": search,
        "search_dataset": search_dataset,
        "answer": answer,
        "answer_dataset": answer_dataset,
        "evaluate": evaluate,
    })
