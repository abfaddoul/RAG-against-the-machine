import json
from pathlib import Path

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import IndexingResult, index
from sqlalchemy.engine import URL

from src.bonus.embeddings import DEFAULT_EMBEDDING_BATCH_SIZE, load_embedding_model
from src.chunking.corpus import chunk_corpus
from src.chunking.models import Chunk
from src.indexing.pipeline import DEFAULT_CORPUS_DIRECTORY, DEFAULT_MAX_CHUNK_SIZE
from src.ingestion.files import load_corpus

DEFAULT_SEMANTIC_INDEX_DIRECTORY = Path("data/processed/semantic_index")
COLLECTION_NAME = "rag_chunks"


def chunk_to_document(chunk: Chunk) -> Document:
    """Preserve source offsets and identity across both retrievers."""
    metadata = chunk.model_dump(exclude={"text"})
    metadata["chunk_id"] = json.dumps([
        chunk.file_path, chunk.first_character_index, chunk.last_character_index,
    ])
    return Document(page_content=chunk.text, metadata=metadata)


def document_to_chunk(document: Document) -> Chunk:
    """Restore the project's source model from a LangChain document."""
    return Chunk.model_validate({**document.metadata, "text": document.page_content})


def load_semantic_index(
    index_directory: str | Path = DEFAULT_SEMANTIC_INDEX_DIRECTORY,
    embedding_model: Embeddings | None = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    *,
    create: bool = False,
) -> Chroma:
    """Open a local persistent HNSW collection with CPU embeddings."""
    directory = Path(index_directory)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not create and not (directory / "chroma.sqlite3").is_file():
        raise FileNotFoundError("Semantic index missing. Run index_semantic first.")
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(directory.resolve()),
        embedding_function=(
            embedding_model if embedding_model is not None
            else load_embedding_model(batch_size=batch_size)
        ),
        collection_configuration={"hnsw": {"space": "cosine"}},
        client_settings=Settings(anonymized_telemetry=False),
        create_collection_if_not_exists=create,
    )


def index_semantic_corpus(
    corpus_directory: str | Path = DEFAULT_CORPUS_DIRECTORY,
    index_directory: str | Path = DEFAULT_SEMANTIC_INDEX_DIRECTORY,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    embedding_model: Embeddings | None = None,
    *,
    incremental: bool = False,
) -> IndexingResult:
    """Let LangChain hash, deduplicate, embed and clean up corpus chunks.

    Full cleanup sees the complete corpus, including deletions. Incremental
    runs skip existing chunk hashes, so only changed chunks are embedded.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    directory = Path(index_directory)
    records_path = directory / "records.sqlite3"
    if incremental and not records_path.is_file():
        raise FileNotFoundError("Incremental state missing. Run index_semantic first.")

    chunks = chunk_corpus(load_corpus(corpus_directory), max_chunk_size)
    store = load_semantic_index(
        directory, embedding_model, batch_size, create=not incremental,
    )
    records = SQLRecordManager(
        namespace=COLLECTION_NAME,
        db_url=URL.create("sqlite", database=str(records_path.resolve())),
    )
    try:
        records.create_schema()
        if not incremental:
            store.reset_collection()
        return index(
            [chunk_to_document(chunk) for chunk in chunks],
            records,
            store,
            cleanup="full",
            source_id_key="file_path",
            batch_size=batch_size,
            force_update=not incremental,
            key_encoder="sha256",
        )
    finally:
        records.engine.dispose()
