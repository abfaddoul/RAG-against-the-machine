import uuid

import pytest
from pydantic import ValidationError
from src.json_io import load_json_model, save_json_model
from pathlib import Path

from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


def test_minimal_source_creation() -> None:
    """Test creating a valid minimal source."""

    source = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=0,
        last_character_index=32,
    )

    assert source.file_path == "data/raw/example.py"
    assert source.first_character_index == 0
    assert source.last_character_index == 32


def test_minimal_source_rejects_missing_fields() -> None:
    """Test rejecting source data with a missing field."""

    incomplete_data = {
        "file_path": "data/raw/example.py",
        "first_character_index": 0,
    }

    with pytest.raises(ValidationError):
        MinimalSource.model_validate(incomplete_data)


def test_unanswered_questions_get_unique_ids() -> None:
    """Test generating a different UUID for every question."""

    first_question = UnansweredQuestion(question="What is LoRA?")
    second_question = UnansweredQuestion(question="What is LoRA?")

    assert first_question.question_id != second_question.question_id
    assert str(uuid.UUID(first_question.question_id)) == (
        first_question.question_id
    )


def test_answered_question_contains_sources_and_answer() -> None:
    """Test creating an answered question with its source."""

    source = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=0,
        last_character_index=32,
    )

    question = AnsweredQuestion(
        question_id="q1",
        question="How does the add function work?",
        sources=[source],
        answer="The function returns the sum of two values.",
    )

    assert question.question_id == "q1"
    assert question.question == "How does the add function work?"
    assert question.sources == [source]
    assert question.answer == (
        "The function returns the sum of two values."
    )


def test_rag_dataset_parses_mixed_questions() -> None:
    """Test parsing answered and unanswered questions in one dataset."""

    dataset_data = {
        "rag_questions": [
            {
                "question_id": "q1",
                "question": "What is RAG?",
            },
            {
                "question_id": "q2",
                "question": "How does retrieval work?",
                "sources": [
                    {
                        "file_path": "data/raw/example.py",
                        "first_character_index": 0,
                        "last_character_index": 32,
                    }
                ],
                "answer": "Retrieval finds relevant source chunks.",
            },
        ]
    }

    dataset = RagDataset.model_validate(dataset_data)

    assert len(dataset.rag_questions) == 2
    assert type(dataset.rag_questions[0]) is UnansweredQuestion
    assert isinstance(dataset.rag_questions[1], AnsweredQuestion)


def test_minimal_search_results_contains_sources() -> None:
    """Test storing retrieved sources for one question."""

    source = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=0,
        last_character_index=32,
    )

    result = MinimalSearchResults(
        question_id="q1",
        question="How does retrieval work?",
        retrieved_sources=[source],
    )

    assert result.question_id == "q1"
    assert result.retrieved_sources == [source]


def test_minimal_answer_extends_search_results() -> None:
    """Test adding a generated answer to search results."""

    result = MinimalAnswer(
        question_id="q1",
        question="What is RAG?",
        retrieved_sources=[],
        answer="RAG combines retrieval with generation.",
    )

    assert result.retrieved_sources == []
    assert result.answer == "RAG combines retrieval with generation."


def test_student_search_results_json_round_trip() -> None:
    """Test serializing and restoring batch search results."""

    result = MinimalSearchResults(
        question_id="q1",
        question="What is RAG?",
        retrieved_sources=[],
    )

    batch = StudentSearchResults(
        search_results=[result],
        k=5,
    )

    json_text = batch.model_dump_json()
    restored_batch = StudentSearchResults.model_validate_json(json_text)

    assert restored_batch == batch
    assert restored_batch.k == 5


def test_student_answers_json_round_trip() -> None:
    """Test serializing and restoring batch answers."""

    result = MinimalAnswer(
        question_id="q1",
        question="What is RAG?",
        retrieved_sources=[],
        answer="RAG combines retrieval with generation.",
    )

    batch = StudentSearchResultsAndAnswer(
        search_results=[result],
        k=5,
    )

    json_text = batch.model_dump_json()
    restored_batch = StudentSearchResultsAndAnswer.model_validate_json(
        json_text
    )

    assert restored_batch == batch
    assert restored_batch.search_results[0].answer == result.answer


def test_save_json_model_writes_formatted_file(
    tmp_path: Path,
) -> None:
    """Test saving a model inside a JSON file."""

    source = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=10,
        last_character_index=50,
    )

    output_path = tmp_path / "nested" / "source.json"

    save_json_model(source, output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == (
        source.model_dump_json(indent=2) + "\n"
    )


def test_load_json_model_restores_model(
    tmp_path: Path,
) -> None:
    """Test loading a validated model from a JSON file."""

    source = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=10,
        last_character_index=50,
    )

    input_path = tmp_path / "source.json"
    input_path.write_text(
        source.model_dump_json(),
        encoding="utf-8",
    )

    loaded_source = load_json_model(input_path, MinimalSource)

    assert isinstance(loaded_source, MinimalSource)
    assert loaded_source == source


def test_load_json_model_rejects_malformed_json(
    tmp_path: Path,
) -> None:
    """Test rejecting a file containing invalid JSON."""

    input_path = tmp_path / "broken.json"
    input_path.write_text(
        '{"file_path": ',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_json_model(input_path, MinimalSource)
