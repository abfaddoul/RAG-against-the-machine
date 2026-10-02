from src.models import MinimalSource


class Chunk(MinimalSource):
    """Represent a source location together with its original text."""

    text: str
