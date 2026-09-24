from pathlib import Path

from src.chunking.corpus import chunk_corpus
from src.indexing.bm25 import BM25Index, build_bm25_index
from src.indexing.storage import save_bm25_index
from src.ingestion.files import load_corpus


DEFAULT_CORPUS_DIRECTORY = Path("data/raw")
DEFAULT_INDEX_PATH = Path("data/processed/bm25_index.joblib")
DEFAULT_MAX_CHUNK_SIZE = 2000


def index_corpus(
    corpus_directory: str | Path = DEFAULT_CORPUS_DIRECTORY,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> BM25Index:
    """Build and save a BM25 index from a source corpus."""

    corpus = load_corpus(corpus_directory)

    chunks = chunk_corpus(
        corpus=corpus,
        max_chunk_size=max_chunk_size,
    )

    index = build_bm25_index(chunks)

    save_bm25_index(
        index=index,
        path=index_path,
    )

    return index
