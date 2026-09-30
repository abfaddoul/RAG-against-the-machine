from dataclasses import dataclass
from pathlib import Path

import hnswlib
import numpy as np
from langchain_core.embeddings import Embeddings

from src.bonus.embeddings import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL_ID,
    embed_chunks,
    load_embedding_model,
)
from src.chunking.corpus import chunk_corpus
from src.chunking.models import Chunk
from src.indexing.pipeline import DEFAULT_CORPUS_DIRECTORY, DEFAULT_MAX_CHUNK_SIZE
from src.ingestion.files import load_corpus

DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_EF_SEARCH = 50
DEFAULT_SEMANTIC_INDEX_DIRECTORY = Path("data/processed/semantic_index")


@dataclass
class HNSWIndex:
    """Store an HNSW index with its aligned chunks."""

    chunks: list[Chunk]
    hnsw: hnswlib.Index
    dimension: int
    embedding_model_id: str
    ef_search: int
    active_count: int


def build_hnsw_index(
    chunks: list[Chunk],
    vectors: list[list[float]],
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
    m: int = DEFAULT_HNSW_M,
    ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
    ef_search: int = DEFAULT_HNSW_EF_SEARCH,
) -> HNSWIndex:
    """Build an HNSW index from chunks and vectors."""

    if not chunks:
        raise ValueError("Cannot build an index without chunks")

    if len(chunks) != len(vectors):
        raise ValueError("Chunks and vectors must have equal lengths")

    if not embedding_model_id.strip():
        raise ValueError("embedding_model_id cannot be empty")

    if m <= 0:
        raise ValueError("m must be positive")

    if ef_construction <= 0:
        raise ValueError("ef_construction must be positive")

    if ef_search <= 0:
        raise ValueError("ef_search must be positive")

    try:
        vector_array = np.asarray(vectors, dtype=np.float32)
    except ValueError as error:
        raise ValueError("Vectors must have equal dimensions") from error

    if vector_array.ndim != 2 or vector_array.shape[1] == 0:
        raise ValueError("Vectors must form a non-empty 2D matrix")

    if not np.isfinite(vector_array).all():
        raise ValueError("Vectors must contain finite values")

    dimension = int(vector_array.shape[1])

    hnsw_index = hnswlib.Index(space="cosine", dim=dimension)

    hnsw_index.init_index(
        max_elements=len(chunks), M=m, ef_construction=ef_construction
    )

    chunk_ids = np.arange(len(chunks))

    hnsw_index.add_items(vector_array, chunk_ids)
    hnsw_index.set_ef(ef_search)

    return HNSWIndex(
        chunks=list(chunks),
        hnsw=hnsw_index,
        dimension=dimension,
        embedding_model_id=embedding_model_id,
        ef_search=ef_search,
        active_count=len(chunks),
    )


def build_semantic_index(
    chunks: list[Chunk],
    embedding_model: Embeddings,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    m: int = DEFAULT_HNSW_M,
    ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
    ef_search: int = DEFAULT_HNSW_EF_SEARCH,
) -> HNSWIndex:
    """Embed chunks and build their HNSW index."""

    vectors = embed_chunks(
        chunks=chunks, embedding_model=embedding_model, batch_size=batch_size
    )

    return build_hnsw_index(
        chunks=chunks,
        vectors=vectors,
        embedding_model_id=embedding_model_id,
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )


def index_semantic_corpus(
    corpus_directory: str | Path = DEFAULT_CORPUS_DIRECTORY,
    index_directory: str | Path = DEFAULT_SEMANTIC_INDEX_DIRECTORY,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    m: int = DEFAULT_HNSW_M,
    ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
    ef_search: int = DEFAULT_HNSW_EF_SEARCH,
    embedding_model: Embeddings | None = None,
    state_path: str | Path | None = None,
) -> HNSWIndex:
    """Build and save a semantic HNSW corpus index."""

    from src.bonus.incremental import (
        build_incremental_state,
        hash_corpus_files,
        save_incremental_state,
    )
    from src.bonus.storage import save_hnsw_index

    corpus = load_corpus(corpus_directory)

    chunks = chunk_corpus(corpus=corpus, max_chunk_size=max_chunk_size)

    if embedding_model is None:
        embedding_model = load_embedding_model(embedding_model_id)

    index = build_semantic_index(
        chunks=chunks,
        embedding_model=embedding_model,
        embedding_model_id=embedding_model_id,
        batch_size=batch_size,
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )

    save_hnsw_index(index=index, directory=index_directory)

    file_hashes = hash_corpus_files(corpus_directory)
    state = build_incremental_state(file_hashes=file_hashes, semantic_index=index)
    resolved_state_path = (
        Path(state_path)
        if state_path is not None
        else Path(index_directory) / "incremental_state.joblib"
    )
    save_incremental_state(state=state, path=resolved_state_path)

    return index
