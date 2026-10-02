from pathlib import Path

from src.models import MinimalSource


MAX_CONTEXT_CHARACTERS = 12_000


def read_source_text(source: MinimalSource) -> str:
    """Read the exact text covered by a retrieved source."""

    file_path = Path(source.file_path)

    try:
        file_text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""

    start = max(0, source.first_character_index)
    end = min(
        len(file_text),
        source.last_character_index,
    )

    if start >= end:
        return ""

    return file_text[start:end]


def build_context(
    sources: list[MinimalSource],
    max_context_characters: int = MAX_CONTEXT_CHARACTERS,
) -> str:
    """Build a size-limited LLM context."""

    if max_context_characters <= 0:
        raise ValueError(
            "max_context_characters must be positive"
        )

    context_blocks: list[str] = []
    context_length = 0

    for position, source in enumerate(sources, start=1):
        source_text = read_source_text(source)

        if not source_text:
            continue

        block = (
            f"[Source {position}]\n"
            f"File: {source.file_path}\n"
            f"Characters: "
            f"{source.first_character_index}-"
            f"{source.last_character_index}\n"
            f"{source_text}"
        )

        separator_length = 2 if context_blocks else 0
        remaining_length = (
            max_context_characters
            - context_length
            - separator_length
        )

        if remaining_length <= 0:
            break

        context_blocks.append(
            block[:remaining_length]
        )

        context_length += (
            separator_length
            + min(len(block), remaining_length)
        )

        if context_length >= max_context_characters:
            break

    return "\n\n".join(context_blocks)
