from dataclasses import dataclass
from pathlib import Path

import hnswlib
import joblib

from src.chunking.models import Chunk
from src.indexing.hnsw import HNSWIndex


HNSW_GRAPH_FILENAME = "hnsw.bin"
HNSW_METADATA_FILENAME = "metadata.joblib"


@dataclass
class HNSWMetadata:
    """Store metadata required to restore an HNSW index."""

    chunks: list[Chunk]
    dimension: int
    embedding_model_id: str
    ef_search: int


def save_hnsw_index(
    index: HNSWIndex,
    directory: str | Path,
) -> None:
    """Save an HNSW graph and its metadata."""

    directory_path = Path(directory)
    directory_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_count = int(
        index.hnsw.get_current_count()
    )

    if current_count != len(index.chunks):
        raise ValueError(
            "HNSW graph and chunks are misaligned"
        )

    graph_path = (
        directory_path / HNSW_GRAPH_FILENAME
    )
    metadata_path = (
        directory_path / HNSW_METADATA_FILENAME
    )

    index.hnsw.save_index(str(graph_path))

    metadata = HNSWMetadata(
        chunks=list(index.chunks),
        dimension=index.dimension,
        embedding_model_id=index.embedding_model_id,
        ef_search=index.ef_search,
    )

    joblib.dump(
        metadata,
        metadata_path,
    )


def load_hnsw_index(
    directory: str | Path,
) -> HNSWIndex:
    """Load an HNSW graph and its metadata."""

    directory_path = Path(directory)
    graph_path = (
        directory_path / HNSW_GRAPH_FILENAME
    )
    metadata_path = (
        directory_path / HNSW_METADATA_FILENAME
    )

    loaded_object: object = joblib.load(
        metadata_path
    )

    if not isinstance(
        loaded_object,
        HNSWMetadata,
    ):
        raise TypeError(
            "File does not contain HNSWMetadata"
        )

    metadata = loaded_object

    hnsw_index = hnswlib.Index(
        space="cosine",
        dim=metadata.dimension,
    )
    hnsw_index.load_index(
        str(graph_path)
    )
    hnsw_index.set_ef(
        metadata.ef_search
    )

    current_count = int(
        hnsw_index.get_current_count()
    )

    if current_count != len(metadata.chunks):
        raise ValueError(
            "HNSW graph and metadata are misaligned"
        )

    return HNSWIndex(
        chunks=list(metadata.chunks),
        hnsw=hnsw_index,
        dimension=metadata.dimension,
        embedding_model_id=(
            metadata.embedding_model_id
        ),
        ef_search=metadata.ef_search,
    )
