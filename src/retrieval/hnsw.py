import numpy as np

from src.chunking.models import Chunk
from src.indexing.hnsw import HNSWIndex


def retrieve_hnsw_chunks(
    index: HNSWIndex,
    query_vector: list[float],
    k: int,
) -> list[Chunk]:
    """Return the top-k chunks nearest to a query vector."""

    if k < 0:
        raise ValueError("k cannot be negative")

    if k == 0:
        return []

    try:
        query_array = np.asarray(
            query_vector,
            dtype=np.float32,
        )
    except ValueError as error:
        raise ValueError(
            "Query vector must contain numbers"
        ) from error

    if query_array.ndim != 1:
        raise ValueError(
            "Query vector must be one-dimensional"
        )

    if len(query_array) != index.dimension:
        raise ValueError(
            "Query vector dimension does not match index"
        )

    if not np.isfinite(query_array).all():
        raise ValueError(
            "Query vector must contain finite values"
        )

    result_count = min(
        k,
        len(index.chunks),
    )

    search_ef = max(
        index.ef_search,
        result_count + 1,
    )
    index.hnsw.set_ef(search_ef)

    labels, _ = index.hnsw.knn_query(
        query_array.reshape(
            1,
            index.dimension,
        ),
        k=result_count,
    )

    return [
        index.chunks[int(chunk_id)]
        for chunk_id in labels[0]
    ]
