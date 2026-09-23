from src.chunking.models import Chunk
from src.indexing.bm25 import BM25Index
from src.indexing.tokenization import tokenize


def retrieve_chunks(
    index: BM25Index,
    query: str,
    k: int,
) -> list[Chunk]:
    """Return the top-k chunks matching a query."""

    if k < 0:
        raise ValueError("k cannot be negative")

    if k == 0:
        return []

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = index.bm25.get_scores(query_tokens)

    candidate_positions: list[int] = []

    for position, token_frequencies in enumerate(
        index.bm25.doc_freqs
    ):
        for token in query_tokens:
            if token in token_frequencies:
                candidate_positions.append(position)
                break

    def get_score(position: int) -> float:
        return float(scores[position])

    ranked_positions = sorted(
        candidate_positions,
        key=get_score,
        reverse=True,
    )

    return [
        index.chunks[position]
        for position in ranked_positions[:k]
    ]
