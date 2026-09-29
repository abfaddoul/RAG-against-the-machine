from langchain_core.embeddings import Embeddings
from tqdm import tqdm

from src.chunking.models import Chunk
from src.embeddings.model import (
    DEFAULT_EMBEDDING_MODEL_ID,
)
from src.indexing.hnsw import (
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_M,
    HNSWIndex,
    build_hnsw_index,
)


DEFAULT_EMBEDDING_BATCH_SIZE = 32


def embed_chunks(
    chunks: list[Chunk],
    embedding_model: Embeddings,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """Create embedding vectors for source chunks."""

    if not chunks:
        raise ValueError(
            "Cannot embed an empty chunk collection"
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    vectors: list[list[float]] = []

    for start in tqdm(
        range(0, len(chunks), batch_size),
        desc="Embedding chunks",
        unit="batch",
    ):
        batch = chunks[start:start + batch_size]
        texts = [
            chunk.text
            for chunk in batch
        ]

        batch_vectors = (
            embedding_model.embed_documents(texts)
        )

        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                "Embedding model returned an invalid "
                "number of vectors"
            )

        vectors.extend(batch_vectors)

    return vectors


def build_semantic_index(
    chunks: list[Chunk],
    embedding_model: Embeddings,
    embedding_model_id: str = (
        DEFAULT_EMBEDDING_MODEL_ID
    ),
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    m: int = DEFAULT_HNSW_M,
    ef_construction: int = (
        DEFAULT_HNSW_EF_CONSTRUCTION
    ),
    ef_search: int = DEFAULT_HNSW_EF_SEARCH,
) -> HNSWIndex:
    """Embed chunks and build their HNSW index."""

    vectors = embed_chunks(
        chunks=chunks,
        embedding_model=embedding_model,
        batch_size=batch_size,
    )

    return build_hnsw_index(
        chunks=chunks,
        vectors=vectors,
        embedding_model_id=embedding_model_id,
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )
