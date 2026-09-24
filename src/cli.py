import fire
from src.indexing.storage import load_bm25_index
from src.retrieval.bm25 import retrieve_chunks
from src.indexing.pipeline import (
    DEFAULT_CORPUS_DIRECTORY,
    DEFAULT_INDEX_PATH,
    DEFAULT_MAX_CHUNK_SIZE,
    index_corpus,
)
from pathlib import Path
from src.json_io import load_json_model, save_json_model
from src.models import (
    RagDataset,
    StudentSearchResults,
)
from src.retrieval.search import retrieve_dataset
from src.evaluation.recall import calculate_recall_at_k


def index(
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    corpus_directory: str = str(DEFAULT_CORPUS_DIRECTORY),
    index_path: str = str(DEFAULT_INDEX_PATH),
) -> None:
    """Build and save the BM25 index."""

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
) -> None:
    """Search the BM25 index for relevant chunks."""

    bm25_index = load_bm25_index(index_path)

    chunks = retrieve_chunks(
        index=bm25_index,
        query=query,
        k=k,
    )

    if not chunks:
        print("No relevant sources found.")
        return

    for rank, chunk in enumerate(chunks, start=1):
        print(f"\n[Result {rank}]")
        print(f"File: {chunk.file_path}")
        print(
            f"Characters: {chunk.first_character_index}-"
            f"{chunk.last_character_index}"
        )
        print(chunk.text)


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

    output_path = Path(save_directory) / Path(dataset_path).name

    save_json_model(
        model=student_search_results,
        path=output_path,
    )

    print(f"Saved student_search_results to {output_path}")


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


def run_cli() -> None:
    """Run the command-line interface."""

    fire.Fire({
        "index": index,
        "search": search,
        "search_dataset": search_dataset,
        "evaluate": evaluate,
    })
