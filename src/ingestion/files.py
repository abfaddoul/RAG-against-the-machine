from pathlib import Path


SUPPORTED_EXTENSIONS = frozenset({
    ".md",
    ".py",
    ".rst",
    ".txt",
})

IGNORED_DIRECTORIES = frozenset({
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
})


def discover_files(root: str | Path) -> list[Path]:
    """Return supported text files found recursively under root."""

    root_path = Path(root)

    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_path}")

    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_path}")

    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(
            part in IGNORED_DIRECTORIES
            for part in path.parts
        )
    )


def read_text_file(path: str | Path) -> str | None:
    """Read a UTF-8 text file, or return None if it is not valid text."""

    file_path = Path(path)

    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def load_corpus(root: str | Path) -> dict[Path, str]:
    """Load supported text files from a directory."""

    documents: dict[Path, str] = {}

    for file_path in discover_files(root):
        content = read_text_file(file_path)

        if content is None or not content.strip():
            continue

        documents[file_path] = content

    return documents
