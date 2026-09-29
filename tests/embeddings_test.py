import pytest

from src.embeddings.model import load_embedding_model


def test_load_embedding_model_rejects_empty_id() -> None:
    """Test rejecting an empty embedding model ID."""

    with pytest.raises(
        ValueError,
        match="model_id cannot be empty",
    ):
        load_embedding_model(" ")
