from langchain_core.embeddings import Embeddings

from src.chunking.models import Chunk
from src.indexing.bm25 import BM25Index
from src.indexing.hnsw import HNSWIndex
from src.retrieval.bm25 import retrieve_chunks
from src.retrieval.semantic import (
    retrieve_semantic_chunks,
)


DEFAULT_RRF_CONSTANT = 60
DEFAULT_CANDIDATE_MULTIPLIER = 2
MINIMUM_CANDIDATE_COUNT = 10

ChunkKey = tuple[str, int, int]


def chunk_key(chunk: Chunk) -> ChunkKey:
    """Return the stable identity of a chunk."""

    return (
        chunk.file_path,
        chunk.first_character_index,
        chunk.last_character_index,
    )


def reciprocal_rank_fusion(
    rankings: list[list[Chunk]],
    k: int,
    rrf_constant: int = DEFAULT_RRF_CONSTANT,
) -> list[Chunk]:
    """Fuse multiple chunk rankings using RRF."""

    if k < 0:
        raise ValueError("k cannot be negative")

    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be positive")

    if k == 0:
        return []

    scores: dict[ChunkKey, float] = {}
    chunks: dict[ChunkKey, Chunk] = {}

    for ranking in rankings:
        seen: set[ChunkKey] = set()

        for rank, chunk in enumerate(ranking, start=1):
            identity = chunk_key(chunk)

            if identity in seen:
                continue

            seen.add(identity)
            chunks.setdefault(identity, chunk)

            scores[identity] = (
                scores.get(identity, 0.0)
                + 1.0 / (rrf_constant + rank)
            )

    ranked_identities = sorted(
        scores,
        key=scores.__getitem__,
        reverse=True,
    )

    return [
        chunks[identity]
        for identity in ranked_identities[:k]
    ]


def retrieve_hybrid_chunks(
    bm25_index: BM25Index,
    semantic_index: HNSWIndex,
    query: str,
    k: int,
    embedding_model: Embeddings | None = None,
    rrf_constant: int = DEFAULT_RRF_CONSTANT,
    candidate_multiplier: int = (
        DEFAULT_CANDIDATE_MULTIPLIER
    ),
) -> list[Chunk]:
    """Combine BM25 and semantic retrieval results."""

    if k < 0:
        raise ValueError("k cannot be negative")

    if candidate_multiplier <= 0:
        raise ValueError(
            "candidate_multiplier must be positive"
        )

    if k == 0:
        return []

    candidate_count = max(
        MINIMUM_CANDIDATE_COUNT,
        k * candidate_multiplier,
    )

    bm25_chunks = retrieve_chunks(
        index=bm25_index,
        query=query,
        k=candidate_count,
    )

    semantic_chunks = retrieve_semantic_chunks(
        index=semantic_index,
        query=query,
        k=candidate_count,
        embedding_model=embedding_model,
    )

    return reciprocal_rank_fusion(
        rankings=[
            bm25_chunks,
            semantic_chunks,
        ],
        k=k,
        rrf_constant=rrf_constant,
    )
