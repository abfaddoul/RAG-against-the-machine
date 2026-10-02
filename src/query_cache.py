from collections.abc import Callable
from pathlib import Path
from typing import cast

from diskcache import Cache

from src.chunking.models import Chunk


CACHE_DIRECTORY = Path("data/processed/query_cache")


def cached_search(
    mode: str,
    query: str,
    k: int,
    index_paths: tuple[str, ...],
    search: Callable[[], list[Chunk]],
) -> list[Chunk]:
    """Reuse saved retrieval results or run the search."""

    paths = tuple(str(Path(path).resolve()) for path in index_paths)
    key = (mode, paths, query, k)

    with Cache(str(CACHE_DIRECTORY)) as cache:
        saved = cache.get(key)

        if saved is not None:
            return cast(list[Chunk], saved)

        chunks = search()
        cache.set(key, chunks)
        return chunks


def clear_query_cache() -> None:
    """Discard cached retrieval results."""

    with Cache(str(CACHE_DIRECTORY)) as cache:
        cache.clear()
