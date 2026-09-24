from pathlib import Path
from tqdm import tqdm
from src.chunking.langchain_splitters import chunk_document
from src.chunking.models import Chunk
TARGET_CHUNK_SIZE = 1000


def chunk_corpus(
    corpus: dict[Path, str],
    max_chunk_size: int,
) -> list[Chunk]:
    """Chunk all documents contained in a corpus."""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")

    effective_chunk_size = min(
        max_chunk_size,
        TARGET_CHUNK_SIZE,
    )
    chunks: list[Chunk] = []

    for file_path in tqdm(
        sorted(corpus),
        desc="Chunking corpus",
        unit="file",
    ):
        file_chunks = chunk_document(
            text=corpus[file_path],
            file_path=str(file_path),
            max_chunk_size=effective_chunk_size,
        )

        chunks.extend(file_chunks)

    return chunks
