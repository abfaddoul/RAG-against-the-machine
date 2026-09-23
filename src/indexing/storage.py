from pathlib import Path

import joblib

from src.indexing.bm25 import BM25Index


def save_bm25_index(
    index: BM25Index,
    path: str | Path,
) -> None:
    """Save a BM25 index to disk."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, file_path)


def load_bm25_index(path: str | Path) -> BM25Index:
    """Load a BM25 index from disk."""

    file_path = Path(path)
    loaded_object: object = joblib.load(file_path)

    if not isinstance(loaded_object, BM25Index):
        raise TypeError("File does not contain a BM25Index")

    return loaded_object
