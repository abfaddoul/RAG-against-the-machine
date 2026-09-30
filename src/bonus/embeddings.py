from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from src.chunking.models import Chunk

DEFAULT_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_BATCH_SIZE = 32


@lru_cache(maxsize=1)
def load_embedding_model(
    model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
) -> HuggingFaceEmbeddings:
    """Load and cache the local embedding model."""

    if not model_id.strip():
        raise ValueError("model_id cannot be empty")

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return embedding_model


def embed_chunks(
    chunks: list[Chunk],
    embedding_model: Embeddings,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """Create embedding vectors for source chunks."""

    if not chunks:
        raise ValueError("Cannot embed an empty chunk collection")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    vectors: list[list[float]] = []

    for start in tqdm(
        range(0, len(chunks), batch_size), desc="Embedding chunks", unit="batch"
    ):
        batch = chunks[start:start + batch_size]
        texts = [chunk.text for chunk in batch]

        batch_vectors = embedding_model.embed_documents(texts)

        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                "Embedding model returned an invalid " "number of vectors"
            )

        vectors.extend(batch_vectors)

    return vectors
