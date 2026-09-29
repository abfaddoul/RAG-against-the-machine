from pathlib import Path

from langchain_core.embeddings import Embeddings

from src.chunking.corpus import chunk_corpus
from src.embeddings.model import (
    DEFAULT_EMBEDDING_MODEL_ID,
    load_embedding_model,
)
from src.indexing.bm25 import (
    BM25Index,
    build_bm25_index,
)
from src.indexing.hnsw import (
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_M,
    HNSWIndex,
)
from src.indexing.hnsw_storage import (
    save_hnsw_index,
)
from src.indexing.semantic import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    build_semantic_index,
)
from src.indexing.storage import save_bm25_index
from src.ingestion.files import load_corpus


DEFAULT_CORPUS_DIRECTORY = Path("data/raw")
DEFAULT_INDEX_PATH = Path(
    "data/processed/bm25_index.joblib"
)
DEFAULT_SEMANTIC_INDEX_DIRECTORY = Path(
    "data/processed/semantic_index"
)
DEFAULT_MAX_CHUNK_SIZE = 2000


def index_corpus(
    corpus_directory: str | Path = (
        DEFAULT_CORPUS_DIRECTORY
    ),
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


def index_semantic_corpus(
    corpus_directory: str | Path = (
        DEFAULT_CORPUS_DIRECTORY
    ),
    index_directory: str | Path = (
        DEFAULT_SEMANTIC_INDEX_DIRECTORY
    ),
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    embedding_model_id: str = (
        DEFAULT_EMBEDDING_MODEL_ID
    ),
    batch_size: int = (
        DEFAULT_EMBEDDING_BATCH_SIZE
    ),
    m: int = DEFAULT_HNSW_M,
    ef_construction: int = (
        DEFAULT_HNSW_EF_CONSTRUCTION
    ),
    ef_search: int = DEFAULT_HNSW_EF_SEARCH,
    embedding_model: Embeddings | None = None,
) -> HNSWIndex:
    """Build and save a semantic HNSW corpus index."""

    corpus = load_corpus(corpus_directory)

    chunks = chunk_corpus(
        corpus=corpus,
        max_chunk_size=max_chunk_size,
    )

    if embedding_model is None:
        embedding_model = load_embedding_model(
            embedding_model_id
        )

    index = build_semantic_index(
        chunks=chunks,
        embedding_model=embedding_model,
        embedding_model_id=embedding_model_id,
        batch_size=batch_size,
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )

    save_hnsw_index(
        index=index,
        directory=index_directory,
    )

    return index
