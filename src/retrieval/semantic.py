from langchain_core.embeddings import Embeddings

from src.chunking.models import Chunk
from src.embeddings.model import load_embedding_model
from src.indexing.hnsw import HNSWIndex
from src.retrieval.hnsw import retrieve_hnsw_chunks


def retrieve_semantic_chunks(
    index: HNSWIndex,
    query: str,
    k: int,
    embedding_model: Embeddings | None = None,
) -> list[Chunk]:
    """Return chunks semantically related to a query."""

    if k < 0:
        raise ValueError("k cannot be negative")

    if k == 0:
        return []

    if not query.strip():
        return []

    if embedding_model is None:
        embedding_model = load_embedding_model(
            index.embedding_model_id
        )

    query_vector = embedding_model.embed_query(
        query
    )

    return retrieve_hnsw_chunks(
        index=index,
        query_vector=query_vector,
        k=k,
    )
