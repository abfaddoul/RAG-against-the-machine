from pathlib import Path

import pytest

from src.ingestion.files import (
    discover_files,
    load_corpus,
    read_text_file,
)


def test_discover_files_returns_supported_files_recursively(
    tmp_path: Path,
) -> None:
    """Test recursive discovery and extension filtering."""

    docs_directory = tmp_path / "docs"
    source_directory = tmp_path / "src"
    ignored_directory = tmp_path / ".git"

    docs_directory.mkdir()
    source_directory.mkdir()
    ignored_directory.mkdir()

    markdown_file = docs_directory / "guide.md"
    python_file = source_directory / "main.PY"
    unsupported_file = tmp_path / "image.png"
    ignored_file = ignored_directory / "config.txt"

    markdown_file.write_text("# Guide", encoding="utf-8")
    python_file.write_text("print('hello')", encoding="utf-8")
    unsupported_file.write_bytes(b"PNG")
    ignored_file.write_text("ignored", encoding="utf-8")

    discovered_files = discover_files(tmp_path)

    assert discovered_files == sorted([
        markdown_file,
        python_file,
    ])


def test_discover_files_rejects_missing_path(
    tmp_path: Path,
) -> None:
    """Test rejecting a path that does not exist."""

    missing_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        discover_files(missing_path)


def test_discover_files_rejects_file_as_root(
    tmp_path: Path,
) -> None:
    """Test rejecting a file when a directory is expected."""

    file_path = tmp_path / "single.py"
    file_path.write_text("print('hello')", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        discover_files(file_path)


def test_read_text_file_returns_utf8_content(
    tmp_path: Path,
) -> None:
    """Test reading valid UTF-8 text."""

    file_path = tmp_path / "notes.txt"
    expected_content = "RAG uses retrieval and generation.\n"

    file_path.write_text(
        expected_content,
        encoding="utf-8",
    )

    assert read_text_file(file_path) == expected_content


def test_read_text_file_returns_none_for_invalid_utf8(
    tmp_path: Path,
) -> None:
    """Test ignoring a file that is not valid UTF-8 text."""

    file_path = tmp_path / "broken.txt"
    file_path.write_bytes(b"\xff\xfe\xfa")

    assert read_text_file(file_path) is None


def test_load_corpus_collects_valid_non_empty_text(
    tmp_path: Path,
) -> None:
    valid_file = tmp_path / "guide.md"
    empty_file = tmp_path / "empty.py"
    invalid_file = tmp_path / "broken.txt"
    unsupported_file = tmp_path / "image.png"

    valid_file.write_text("# RAG Guide\n", encoding="utf-8")
    empty_file.write_text(" \n", encoding="utf-8")
    invalid_file.write_bytes(b"\xff\xfe\xfa")
    unsupported_file.write_bytes(b"PNG")

    corpus = load_corpus(tmp_path)

    assert corpus == {
        valid_file: "# RAG Guide\n",
    }
