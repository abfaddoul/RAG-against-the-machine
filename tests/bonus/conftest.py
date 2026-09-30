from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings


class RecordingEmbeddings(Embeddings):
    """Deterministic embeddings to test real storage without model downloads."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [self.vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self.vector(text)

    @staticmethod
    def vector(text: str) -> list[float]:
        text = text.lower()
        return [1.0, float(text.count("apple")), float(text.count("banana"))]


@pytest.fixture
def embeddings() -> RecordingEmbeddings:
    return RecordingEmbeddings()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    directory = tmp_path / "corpus"
    directory.mkdir()
    return directory
