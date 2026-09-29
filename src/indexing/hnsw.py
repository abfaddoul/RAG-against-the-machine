from dataclasses import dataclass

import hnswlib
import numpy as np

from src.chunking.models import Chunk
from src.embeddings.model import (
    DEFAULT_EMBEDDING_MODEL_ID,
)


DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_EF_SEARCH = 50


@dataclass
class HNSWIndex:
    """Store an HNSW index with its aligned chunks."""

    chunks: list[Chunk]
    hnsw: hnswlib.Index
    dimension: int
    embedding_model_id: str
    ef_search: int


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
        raise ValueError(
            "Cannot build an index without chunks"
        )

    if len(chunks) != len(vectors):
        raise ValueError(
            "Chunks and vectors must have equal lengths"
        )

    if not embedding_model_id.strip():
        raise ValueError(
            "embedding_model_id cannot be empty"
        )

    if m <= 0:
        raise ValueError("m must be positive")

    if ef_construction <= 0:
        raise ValueError(
            "ef_construction must be positive"
        )

    if ef_search <= 0:
        raise ValueError(
            "ef_search must be positive"
        )

    try:
        vector_array = np.asarray(
            vectors,
            dtype=np.float32,
        )
    except ValueError as error:
        raise ValueError(
            "Vectors must have equal dimensions"
        ) from error

    if (
        vector_array.ndim != 2
        or vector_array.shape[1] == 0
    ):
        raise ValueError(
            "Vectors must form a non-empty 2D matrix"
        )

    if not np.isfinite(vector_array).all():
        raise ValueError(
            "Vectors must contain finite values"
        )

    dimension = int(vector_array.shape[1])

    hnsw_index = hnswlib.Index(
        space="cosine",
        dim=dimension,
    )

    hnsw_index.init_index(
        max_elements=len(chunks),
        M=m,
        ef_construction=ef_construction,
    )

    chunk_ids = np.arange(len(chunks))

    hnsw_index.add_items(
        vector_array,
        chunk_ids,
    )
    hnsw_index.set_ef(ef_search)

    return HNSWIndex(
        chunks=list(chunks),
        hnsw=hnsw_index,
        dimension=dimension,
        embedding_model_id=embedding_model_id,
        ef_search=ef_search,
    )
