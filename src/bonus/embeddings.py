from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_BATCH_SIZE = 32


@lru_cache(maxsize=1)
def load_embedding_model(
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> HuggingFaceEmbeddings:
    """Use the embedding library's CPU batching and progress reporting."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDING_MODEL_ID,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
        show_progress=True,
    )
