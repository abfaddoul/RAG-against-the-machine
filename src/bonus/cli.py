import fire

from src.bonus.embeddings import DEFAULT_EMBEDDING_BATCH_SIZE
from src.bonus.hybrid import retrieve_hybrid_chunks
from src.bonus.indexing import (
    DEFAULT_SEMANTIC_INDEX_DIRECTORY,
    index_semantic_corpus,
    load_semantic_index,
)
from src.bonus.search import retrieve_semantic_chunks
from src.cli import print_chunks
from src.indexing.pipeline import (
    DEFAULT_CORPUS_DIRECTORY,
    DEFAULT_INDEX_PATH,
    DEFAULT_MAX_CHUNK_SIZE,
)
from src.indexing.storage import load_bm25_index


def index_semantic(
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    corpus_directory: str = str(DEFAULT_CORPUS_DIRECTORY),
    index_directory: str = str(DEFAULT_SEMANTIC_INDEX_DIRECTORY),
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> None:
    """Build and save the semantic HNSW index."""

    result = index_semantic_corpus(
        corpus_directory=corpus_directory,
        index_directory=index_directory,
        max_chunk_size=max_chunk_size,
        batch_size=batch_size,
    )

    print(f"Indexed {result['num_added'] + result['num_updated']} semantic chunks.")
    print(f"Saved semantic index to: {index_directory}")


def index_incremental(
    corpus_directory: str = str(DEFAULT_CORPUS_DIRECTORY),
    index_directory: str = str(DEFAULT_SEMANTIC_INDEX_DIRECTORY),
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> None:
    """Update changed files in the semantic index."""

    result = index_semantic_corpus(
        corpus_directory=corpus_directory,
        index_directory=index_directory,
        max_chunk_size=max_chunk_size,
        batch_size=batch_size,
        incremental=True,
    )

    if not (result['num_added'] or result['num_updated'] or result['num_deleted']):
        print("No corpus changes detected.")
    else:
        print(
            f"Chunks: {result['num_added']} added, "
            f"{result['num_updated']} updated, {result['num_deleted']} deleted, "
            f"{result['num_skipped']} unchanged."
        )
    print(f"Saved semantic index to: {index_directory}")


def search_semantic(
    query: str, k: int = 5, index_directory: str = str(DEFAULT_SEMANTIC_INDEX_DIRECTORY)
) -> None:
    """Search the semantic index for relevant chunks."""

    semantic_index = load_semantic_index(index_directory)

    chunks = retrieve_semantic_chunks(index=semantic_index, query=query, k=k)

    if not chunks:
        print("No relevant sources found.")
        return

    print_chunks(chunks)


def search_hybrid(
    query: str,
    k: int = 5,
    index_path: str = str(DEFAULT_INDEX_PATH),
    semantic_index_directory: str = str(DEFAULT_SEMANTIC_INDEX_DIRECTORY),
) -> None:
    """Search lexical and semantic indexes together."""

    bm25_index = load_bm25_index(index_path)
    semantic_index = load_semantic_index(semantic_index_directory)

    chunks = retrieve_hybrid_chunks(
        bm25_index=bm25_index, semantic_index=semantic_index, query=query, k=k
    )

    if not chunks:
        print("No relevant sources found.")
        return

    print_chunks(chunks)


def run_cli() -> None:
    """Run only the optional commands."""
    fire.Fire(
        {
            "index_semantic": index_semantic,
            "index_incremental": index_incremental,
            "search_semantic": search_semantic,
            "search_hybrid": search_hybrid,
        }
    )
