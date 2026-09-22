from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
)
from src.chunking.models import Chunk
from pathlib import Path


MAX_CHUNK_SIZE = 2000


def documents_to_chunks(
    documents: list[Document],
    original_text: str,
    file_path: str,
) -> list[Chunk]:
    """Convert LangChain documents into project chunks."""

    chunks: list[Chunk] = []

    for document in documents:
        start_index = document.metadata.get("start_index")

        if not isinstance(start_index, int):
            raise ValueError("LangChain document has no start index")

        end_index = start_index + len(document.page_content)

        if original_text[start_index:end_index] != document.page_content:
            raise ValueError("Chunk text does not match its source span")

        chunks.append(
            Chunk(
                file_path=file_path,
                first_character_index=start_index,
                last_character_index=end_index,
                text=document.page_content,
            )
        )

    return chunks


def chunk_python(
    text: str,
    file_path: str,
    max_chunk_size: int,
) -> list[Chunk]:
    """Split Python code using LangChain."""

    splitter = PythonCodeTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=0,
        length_function=len,
        add_start_index=True,
        strip_whitespace=False,
    )

    documents = splitter.create_documents([text])

    return documents_to_chunks(
        documents=documents,
        original_text=text,
        file_path=file_path,
    )


def chunk_markdown(
    text: str,
    file_path: str,
    max_chunk_size: int,
) -> list[Chunk]:
    """Split Markdown text using LangChain."""

    splitter = MarkdownTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=0,
        length_function=len,
        add_start_index=True,
        strip_whitespace=False,
    )

    documents = splitter.create_documents([text])

    return documents_to_chunks(
        documents=documents,
        original_text=text,
        file_path=file_path,
    )


def chunk_plain_text(
    text: str,
    file_path: str,
    max_chunk_size: int,
) -> list[Chunk]:
    """Split plain text using LangChain."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=0,
        length_function=len,
        add_start_index=True,
        strip_whitespace=False,
    )

    documents = splitter.create_documents([text])

    return documents_to_chunks(
        documents=documents,
        original_text=text,
        file_path=file_path,
    )


def chunk_document(
    text: str,
    file_path: str,
    max_chunk_size: int,
) -> list[Chunk]:
    """Chunk a document according to its file extension."""

    if max_chunk_size <= 0:
        raise ValueError(
            "max_chunk_size must be greater than zero"
        )

    if max_chunk_size > MAX_CHUNK_SIZE:
        raise ValueError(
            f"max_chunk_size cannot exceed {MAX_CHUNK_SIZE}"
        )

    suffix = Path(file_path).suffix.lower()

    if suffix == ".py":
        return chunk_python(text, file_path, max_chunk_size)

    if suffix == ".md":
        return chunk_markdown(text, file_path, max_chunk_size)

    if suffix in {".txt", ".rst"}:
        return chunk_plain_text(text, file_path, max_chunk_size)

    raise ValueError(f"Unsupported file extension: {suffix}")
