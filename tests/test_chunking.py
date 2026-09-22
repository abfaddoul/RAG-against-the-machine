import pytest
from src.chunking.langchain_splitters import (
    chunk_document,
    chunk_markdown,
    chunk_plain_text,
    chunk_python,
)
from pathlib import Path
from src.chunking.corpus import chunk_corpus


def test_chunk_python_preserves_source_positions() -> None:
    """Test Python chunking with exact source positions."""

    text = (
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "\n"
        "def subtract(a, b):\n"
        "    return a - b\n"
    )

    chunks = chunk_python(
        text=text,
        file_path="data/raw/example.py",
        max_chunk_size=40,
    )

    assert len(chunks) == 2
    assert "".join(chunk.text for chunk in chunks) == text

    for chunk in chunks:
        assert chunk.file_path == "data/raw/example.py"
        assert len(chunk.text) <= 40
        assert text[
            chunk.first_character_index:
            chunk.last_character_index
        ] == chunk.text


def test_chunk_markdown_preserves_source_positions() -> None:
    """Test Markdown chunking with exact source positions."""

    text = (
        "# Introduction\n"
        "\n"
        "RAG combines retrieval with generation.\n"
        "\n"
        "## Indexing\n"
        "\n"
        "Documents are split into chunks.\n"
    )

    chunks = chunk_markdown(
        text=text,
        file_path="data/raw/guide.md",
        max_chunk_size=55,
    )

    assert len(chunks) == 2
    assert "".join(chunk.text for chunk in chunks) == text

    for chunk in chunks:
        assert chunk.file_path == "data/raw/guide.md"
        assert len(chunk.text) <= 55
        assert text[
            chunk.first_character_index:
            chunk.last_character_index
        ] == chunk.text


def test_chunk_plain_text_preserves_source_positions() -> None:
    """Test plain-text chunking with exact source positions."""

    text = (
        "Retrieval finds relevant chunks.\n"
        "\n"
        "Generation uses those chunks as context.\n"
    )

    chunks = chunk_plain_text(
        text=text,
        file_path="data/raw/notes.txt",
        max_chunk_size=45,
    )

    assert len(chunks) == 2
    assert "".join(chunk.text for chunk in chunks) == text

    for chunk in chunks:
        assert chunk.file_path == "data/raw/notes.txt"
        assert len(chunk.text) <= 45
        assert text[
            chunk.first_character_index:
            chunk.last_character_index
        ] == chunk.text


@pytest.mark.parametrize(
    "file_path",
    [
        "data/raw/example.py",
        "data/raw/guide.md",
        "data/raw/notes.txt",
        "data/raw/reference.rst",
    ],
)
def test_chunk_document_supports_expected_extensions(
    file_path: str,
) -> None:
    """Test routing supported files to a chunking strategy."""

    text = "Short document content.\n"

    chunks = chunk_document(
        text=text,
        file_path=file_path,
        max_chunk_size=50,
    )

    assert len(chunks) == 1
    assert chunks[0].file_path == file_path
    assert chunks[0].text == text


def test_chunk_document_rejects_unsupported_extension() -> None:
    """Test rejecting a file without a chunking strategy."""

    with pytest.raises(
        ValueError,
        match="Unsupported file extension",
    ):
        chunk_document(
            text="image",
            file_path="data/raw/image.png",
            max_chunk_size=50,
        )


def test_chunk_corpus_combines_files_in_sorted_order() -> None:
    """Test chunking and combining all corpus files."""

    corpus = {
        Path("data/raw/z.md"): "# Last document\n",
        Path("data/raw/a.py"): "def first():\n    return 1\n",
    }

    chunks = chunk_corpus(
        corpus=corpus,
        max_chunk_size=100,
    )

    assert len(chunks) == 2
    assert [chunk.file_path for chunk in chunks] == [
        "data/raw/a.py",
        "data/raw/z.md",
    ]

    for chunk in chunks:
        original_text = corpus[Path(chunk.file_path)]

        assert original_text[
            chunk.first_character_index:
            chunk.last_character_index
        ] == chunk.text


@pytest.mark.parametrize(
    "max_chunk_size",
    [
        0,
        -1,
        2001,
    ],
)
def test_chunk_document_rejects_invalid_chunk_size(
    max_chunk_size: int,
) -> None:
    """Test rejecting invalid maximum chunk sizes."""

    with pytest.raises(ValueError):
        chunk_document(
            text="Example text",
            file_path="data/raw/example.txt",
            max_chunk_size=max_chunk_size,
        )
