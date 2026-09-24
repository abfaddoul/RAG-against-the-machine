from pathlib import Path
from tqdm import tqdm
from src.chunking.langchain_splitters import chunk_document
from src.chunking.models import Chunk


def chunk_corpus(
    corpus: dict[Path, str],
    max_chunk_size: int,
) -> list[Chunk]:
    """Chunk all documents contained in a corpus."""

    chunks: list[Chunk] = []

    for file_path in tqdm(
        sorted(corpus),
        desc="Chunking corpus",
        unit="file",
    ):
        file_chunks = chunk_document(
            text=corpus[file_path],
            file_path=str(file_path),
            max_chunk_size=max_chunk_size,
        )

        chunks.extend(file_chunks)

    return chunks
