from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_EMBEDDING_MODEL_ID = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


@lru_cache(maxsize=1)
def load_embedding_model(
    model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
) -> HuggingFaceEmbeddings:
    """Load and cache the local embedding model."""

    if not model_id.strip():
        raise ValueError("model_id cannot be empty")

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    return embedding_model
