from pathlib import Path
import sys

import pytest

from conftest import RecordingEmbeddings
from src.bonus import cli
from src.bonus.indexing import load_semantic_index
from src.cli import run_cli
from src.indexing.pipeline import index_corpus


def test_bonus_commands_use_real_storage_and_incremental_updates(
    corpus: Path, tmp_path: Path, embeddings: RecordingEmbeddings,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_load_model(batch_size: int = 32) -> RecordingEmbeddings:
        return embeddings

    monkeypatch.setattr("src.bonus.indexing.load_embedding_model", fake_load_model)
    source = corpus / "a.txt"
    source.write_text("apple", encoding="utf-8")
    directory = tmp_path / "index"
    bm25_path = tmp_path / "bm25.joblib"
    index_corpus(corpus, bm25_path)

    cli.index_semantic(corpus_directory=str(corpus), index_directory=str(directory))
    assert "Indexed 1 semantic chunks" in capsys.readouterr().out
    cli.search_semantic("apple", index_directory=str(directory))
    assert str(source) in capsys.readouterr().out
    cli.search_hybrid("apple", index_path=str(bm25_path),
                      semantic_index_directory=str(directory))
    assert str(source) in capsys.readouterr().out

    embeddings.texts.clear()
    cli.index_incremental(corpus_directory=str(corpus), index_directory=str(directory))
    assert "No corpus changes detected" in capsys.readouterr().out
    assert embeddings.texts == []
    source.write_text("banana", encoding="utf-8")
    cli.index_incremental(corpus_directory=str(corpus), index_directory=str(directory))
    output = capsys.readouterr().out
    assert "Chunks: 1 added, 0 updated, 1 deleted, 0 unchanged" in output
    assert embeddings.texts == ["banana"]
    assert load_semantic_index(directory, embeddings).get()["documents"] == ["banana"]


def test_fire_routes_the_existing_bonus_command(
    corpus: Path, tmp_path: Path, embeddings: RecordingEmbeddings,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_load_model(batch_size: int = 32) -> RecordingEmbeddings:
        return embeddings

    monkeypatch.setattr("src.bonus.indexing.load_embedding_model", fake_load_model)
    (corpus / "a.txt").write_text("apple", encoding="utf-8")
    directory = tmp_path / "index"
    monkeypatch.setattr(sys, "argv", [
        "src", "index_semantic", f"--corpus_directory={corpus}",
        f"--index_directory={directory}", "--batch_size=2",
    ])
    run_cli()
    assert "Indexed 1 semantic chunks" in capsys.readouterr().out
    assert (directory / "records.sqlite3").is_file()
