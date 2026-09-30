from langchain_chroma import Chroma

from src.bonus.indexing import document_to_chunk
from src.chunking.models import Chunk


def retrieve_semantic_chunks(index: Chroma, query: str, k: int = 5) -> list[Chunk]:
    """Retrieve nearest source chunks through Chroma's HNSW search."""
    if k < 0:
        raise ValueError("k cannot be negative")
    if k == 0 or not query.strip():
        return []
    return [document_to_chunk(doc) for doc in index.similarity_search(query, k=k)]
