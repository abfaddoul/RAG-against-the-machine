from types import SimpleNamespace

import pytest

from src.bonus.cli import index_incremental
from src.bonus.incremental import FileChanges


def test_index_incremental_prints_changes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test reporting incremental index changes."""

    calls: dict[str, object] = {}

    def fake_update_semantic_corpus(
        corpus_directory: str,
        index_directory: str,
        max_chunk_size: int,
        batch_size: int,
        state_path: str | None,
    ) -> tuple[SimpleNamespace, FileChanges]:
        calls["corpus_directory"] = corpus_directory
        calls["index_directory"] = index_directory
        calls["max_chunk_size"] = max_chunk_size
        calls["batch_size"] = batch_size
        calls["state_path"] = state_path

        return (
            SimpleNamespace(active_count=4),
            FileChanges(
                added=["added.py"],
                modified=["modified.py"],
                deleted=["deleted.py"],
                unchanged=["unchanged.py"],
            ),
        )

    monkeypatch.setattr(
        "src.bonus.cli.update_semantic_corpus", fake_update_semantic_corpus
    )

    index_incremental(
        corpus_directory="test-corpus",
        index_directory="test-index",
        max_chunk_size=500,
        batch_size=8,
        state_path="test-state.joblib",
    )

    output = capsys.readouterr().out

    assert calls == {
        "corpus_directory": "test-corpus",
        "index_directory": "test-index",
        "max_chunk_size": 500,
        "batch_size": 8,
        "state_path": "test-state.joblib",
    }

    assert "Added: added.py" in output
    assert "Modified: modified.py" in output
    assert "Deleted: deleted.py" in output
    assert "Active semantic chunks: 4" in output
    assert "Saved semantic index to: test-index" in output
