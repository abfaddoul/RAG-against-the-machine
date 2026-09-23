from dataclasses import dataclass
from rank_bm25 import BM25Okapi

from src.chunking.models import Chunk
from src.indexing.tokenization import tokenize


@dataclass
class BM25Index:
    """Store a BM25 model with its aligned chunks."""

    chunks: list[Chunk]
    bm25: BM25Okapi


def build_bm25_index(chunks: list[Chunk]) -> BM25Index:
    """Build a BM25 index from searchable chunks."""

    if not chunks:
        raise ValueError("Cannot build an index without chunks")

    tokenized_corpus: list[list[str]] = []

    for chunk in chunks:
        searchable_text = f"{chunk.file_path}\n{chunk.text}"
        chunk_tokens = tokenize(searchable_text)
        tokenized_corpus.append(chunk_tokens)

    if not any(tokenized_corpus):
        raise ValueError(
            "Cannot build an index without searchable tokens"
        )

    bm25 = BM25Okapi(tokenized_corpus)

    return BM25Index(
        chunks=list(chunks),
        bm25=bm25,
    )
