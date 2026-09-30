import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from langchain_core.embeddings import Embeddings

from src.bonus.embeddings import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    embed_chunks,
    load_embedding_model,
)
from src.bonus.indexing import DEFAULT_SEMANTIC_INDEX_DIRECTORY, HNSWIndex
from src.bonus.storage import load_hnsw_index, save_hnsw_index
from src.chunking.corpus import chunk_corpus
from src.chunking.models import Chunk
from src.indexing.pipeline import DEFAULT_CORPUS_DIRECTORY, DEFAULT_MAX_CHUNK_SIZE
from src.ingestion.files import discover_files, read_text_file

DEFAULT_INCREMENTAL_STATE_PATH = (
    DEFAULT_SEMANTIC_INDEX_DIRECTORY / "incremental_state.joblib"
)
HASH_BLOCK_SIZE = 1024 * 1024


@dataclass
class IncrementalState:
    """Store the previous corpus state."""

    file_hashes: dict[str, str]
    labels_by_file: dict[str, list[int]] = field(default_factory=dict)
    next_label: int = 0


@dataclass
class FileChanges:
    """Represent changes between two corpus states."""

    added: list[str]
    modified: list[str]
    deleted: list[str]
    unchanged: list[str]

    @property
    def has_changes(self) -> bool:
        """Return whether the corpus changed."""

        return bool(self.added or self.modified or self.deleted)


def hash_file(path: str | Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    digest = hashlib.sha256()
    file_path = Path(path)

    with file_path.open("rb") as file:
        block = file.read(HASH_BLOCK_SIZE)

        while block:
            digest.update(block)
            block = file.read(HASH_BLOCK_SIZE)

    return digest.hexdigest()


def hash_corpus_files(corpus_directory: str | Path) -> dict[str, str]:
    """Return corpus file paths and their hashes."""

    return {
        str(file_path): hash_file(file_path)
        for file_path in discover_files(corpus_directory)
    }


def detect_file_changes(
    previous_hashes: dict[str, str], current_hashes: dict[str, str]
) -> FileChanges:
    """Compare two corpus states."""

    previous_paths = set(previous_hashes)
    current_paths = set(current_hashes)

    added = sorted(current_paths - previous_paths)
    deleted = sorted(previous_paths - current_paths)

    shared_paths = previous_paths & current_paths

    modified = sorted(
        file_path
        for file_path in shared_paths
        if (previous_hashes[file_path] != current_hashes[file_path])
    )

    unchanged = sorted(
        file_path
        for file_path in shared_paths
        if (previous_hashes[file_path] == current_hashes[file_path])
    )

    return FileChanges(
        added=added, modified=modified, deleted=deleted, unchanged=unchanged
    )


def save_incremental_state(
    state: IncrementalState, path: str | Path = DEFAULT_INCREMENTAL_STATE_PATH
) -> None:
    """Save the incremental indexing state."""

    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(state, state_path)


def load_incremental_state(
    path: str | Path = DEFAULT_INCREMENTAL_STATE_PATH,
) -> IncrementalState:
    """Load the incremental indexing state."""

    state_path = Path(path)

    loaded_state: object = joblib.load(state_path)

    if not isinstance(loaded_state, IncrementalState):
        raise TypeError("File does not contain an IncrementalState")

    return loaded_state


def build_incremental_state(
    file_hashes: dict[str, str], semantic_index: HNSWIndex
) -> IncrementalState:
    """Build state from an existing semantic index."""

    labels_by_file: dict[str, list[int]] = {}

    for label, chunk in enumerate(semantic_index.chunks):
        labels_by_file.setdefault(chunk.file_path, []).append(label)

    return IncrementalState(
        file_hashes=dict(file_hashes),
        labels_by_file=labels_by_file,
        next_label=len(semantic_index.chunks),
    )


def remove_file_from_hnsw_index(
    index: HNSWIndex, state: IncrementalState, file_path: str
) -> None:
    """Mark one file's active HNSW labels as deleted."""

    labels = state.labels_by_file.get(file_path, [])

    if len(labels) > index.active_count:
        raise ValueError("File labels exceed the active HNSW count")

    for label in labels:
        index.hnsw.mark_deleted(label)

    index.active_count -= len(labels)
    state.labels_by_file.pop(file_path, None)
    state.file_hashes.pop(file_path, None)


def add_file_to_hnsw_index(
    index: HNSWIndex,
    state: IncrementalState,
    file_path: str,
    file_hash: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> None:
    """Add one file's chunks and vectors to an HNSW index."""

    if file_path in state.labels_by_file:
        raise ValueError("File already has active HNSW labels")

    if state.next_label != len(index.chunks):
        raise ValueError("Incremental state and chunks are misaligned")

    if len(chunks) != len(vectors):
        raise ValueError("Chunks and vectors must have equal lengths")

    if any(chunk.file_path != file_path for chunk in chunks):
        raise ValueError("All chunks must belong to the added file")

    if not chunks:
        state.file_hashes[file_path] = file_hash
        state.labels_by_file[file_path] = []
        return

    try:
        vector_array = np.asarray(vectors, dtype=np.float32)
    except ValueError as error:
        raise ValueError("Vectors must have equal dimensions") from error

    if vector_array.ndim != 2 or vector_array.shape[1] != index.dimension:
        raise ValueError("Vector dimension does not match index")

    if not np.isfinite(vector_array).all():
        raise ValueError("Vectors must contain finite values")

    first_label = state.next_label
    next_label = first_label + len(chunks)

    current_capacity = int(index.hnsw.get_max_elements())

    if next_label > current_capacity:
        index.hnsw.resize_index(next_label)

    labels = np.arange(first_label, next_label)

    index.hnsw.add_items(vector_array, labels)
    index.chunks.extend(chunks)
    index.active_count += len(chunks)

    state.file_hashes[file_path] = file_hash
    state.labels_by_file[file_path] = [int(label) for label in labels]
    state.next_label = next_label


def update_semantic_corpus(
    corpus_directory: str | Path = DEFAULT_CORPUS_DIRECTORY,
    index_directory: str | Path = DEFAULT_SEMANTIC_INDEX_DIRECTORY,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    embedding_model: Embeddings | None = None,
    state_path: str | Path | None = None,
) -> tuple[HNSWIndex, FileChanges]:
    """Update only changed files in a semantic index."""

    index = load_hnsw_index(index_directory)
    current_hashes = hash_corpus_files(corpus_directory)
    resolved_state_path = (
        Path(state_path)
        if state_path is not None
        else Path(index_directory) / "incremental_state.joblib"
    )
    if not resolved_state_path.exists():
        raise FileNotFoundError(
            "Incremental state is missing. Run index_semantic once to rebuild "
            "the semantic index and its file hashes."
        )
    state = load_incremental_state(resolved_state_path)
    changes = detect_file_changes(
        previous_hashes=state.file_hashes, current_hashes=current_hashes
    )

    if not changes.has_changes:
        return index, changes

    updates: dict[str, tuple[list[Chunk], list[list[float]]]] = {}
    for file_path in sorted(changes.added + changes.modified):
        content = read_text_file(file_path)
        corpus = {Path(file_path): content} if content and content.strip() else {}
        chunks = chunk_corpus(corpus=corpus, max_chunk_size=max_chunk_size)
        vectors: list[list[float]] = []
        if chunks:
            if embedding_model is None:
                embedding_model = load_embedding_model(index.embedding_model_id)
            vectors = embed_chunks(chunks, embedding_model, batch_size)
        updates[file_path] = chunks, vectors

    for file_path in changes.deleted + changes.modified:
        remove_file_from_hnsw_index(index=index, state=state, file_path=file_path)

    for file_path, (chunks, vectors) in updates.items():
        add_file_to_hnsw_index(
            index=index,
            state=state,
            file_path=file_path,
            file_hash=current_hashes[file_path],
            chunks=chunks,
            vectors=vectors,
        )

    save_hnsw_index(index=index, directory=index_directory)
    save_incremental_state(state=state, path=resolved_state_path)

    return index, changes
