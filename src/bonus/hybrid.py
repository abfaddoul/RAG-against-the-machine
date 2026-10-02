from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.bonus.indexing import chunk_to_document, document_to_chunk
from src.chunking.models import Chunk
from src.indexing.bm25 import BM25Index
from src.retrieval.bm25 import retrieve_chunks


def retrieve_hybrid_chunks(
    bm25_index: BM25Index, semantic_index: Chroma, query: str, k: int = 5,
) -> list[Chunk]:
    """Fuse the existing BM25 retriever and Chroma using LangChain RRF."""
    if k < 0:
        raise ValueError("k cannot be negative")
    if k == 0 or not query.strip():
        return []
    candidate_count = max(10, k * 2)

    def lexical_search(text: str) -> list[Document]:
        """Retrieve BM25 chunks and convert them to LangChain documents."""
        chunks = retrieve_chunks(bm25_index, text, candidate_count)
        return [chunk_to_document(chunk) for chunk in chunks]

    retriever = EnsembleRetriever(
        retrievers=[
            RunnableLambda(lexical_search),
            semantic_index.as_retriever(search_kwargs={"k": candidate_count}),
        ],
        weights=[0.5, 0.5],
        c=60,
        id_key="chunk_id",
    )
    return [document_to_chunk(doc) for doc in retriever.invoke(query)[:k]]
