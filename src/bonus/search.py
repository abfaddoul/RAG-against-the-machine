import numpy as np
from langchain_core.embeddings import Embeddings

from src.bonus.embeddings import load_embedding_model
from src.bonus.indexing import HNSWIndex
from src.chunking.models import Chunk


def retrieve_hnsw_chunks(
    index: HNSWIndex, query_vector: list[float], k: int
) -> list[Chunk]:
    """Return the top-k chunks nearest to a query vector."""

    if k < 0:
        raise ValueError("k cannot be negative")

    if k == 0:
        return []

    try:
        query_array = np.asarray(query_vector, dtype=np.float32)
    except ValueError as error:
        raise ValueError("Query vector must contain numbers") from error

    if query_array.ndim != 1:
        raise ValueError("Query vector must be one-dimensional")

    if len(query_array) != index.dimension:
        raise ValueError("Query vector dimension does not match index")

    if not np.isfinite(query_array).all():
        raise ValueError("Query vector must contain finite values")

    if index.active_count == 0:
        return []

    result_count = min(k, index.active_count)

    search_ef = max(index.ef_search, result_count + 1)
    index.hnsw.set_ef(search_ef)

    labels, _ = index.hnsw.knn_query(
        query_array.reshape(1, index.dimension), k=result_count
    )

    return [index.chunks[int(chunk_id)] for chunk_id in labels[0]]


def retrieve_semantic_chunks(
    index: HNSWIndex, query: str, k: int, embedding_model: Embeddings | None = None
) -> list[Chunk]:
    """Return chunks semantically related to a query."""

    if k < 0:
        raise ValueError("k cannot be negative")

    if k == 0:
        return []

    if not query.strip():
        return []

    if embedding_model is None:
        embedding_model = load_embedding_model(index.embedding_model_id)

    query_vector = embedding_model.embed_query(query)

    return retrieve_hnsw_chunks(index=index, query_vector=query_vector, k=k)
