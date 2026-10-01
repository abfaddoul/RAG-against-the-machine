import pytest

from src.bonus.cli import index_semantic


def test_index_semantic_prints_count(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Forward CLI options and report the number of indexed chunks."""
    calls: dict[str, object] = {}

    def fake_index_semantic_corpus(
        corpus_directory: str,
        index_directory: str,
        max_chunk_size: int,
        batch_size: int,
    ) -> int:
        calls.update({
            "corpus_directory": corpus_directory,
            "index_directory": index_directory,
            "max_chunk_size": max_chunk_size,
            "batch_size": batch_size,
        })
        return 4

    monkeypatch.setattr(
        "src.bonus.cli.index_semantic_corpus",
        fake_index_semantic_corpus,
    )

    index_semantic(
        corpus_directory="test-corpus",
        index_directory="test-index",
        max_chunk_size=500,
        batch_size=8,
    )

    assert calls == {
        "corpus_directory": "test-corpus",
        "index_directory": "test-index",
        "max_chunk_size": 500,
        "batch_size": 8,
    }
    assert capsys.readouterr().out == (
        "Indexed 4 semantic chunks.\n"
        "Saved semantic index to: test-index\n"
    )
