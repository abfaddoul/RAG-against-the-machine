from pathlib import Path
from src.generation.prompt import build_messages
from src.generation.context import (
    build_context,
    read_source_text,
)
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)
from src.generation.answer import (
    clean_generated_answer,
    generate_answers,
)
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
)


def test_read_source_text_returns_exact_range(
    tmp_path: Path,
) -> None:
    """Test reading the exact character range of a source."""

    file_path = tmp_path / "example.txt"
    file_path.write_text(
        "0123456789",
        encoding="utf-8",
    )

    source = MinimalSource(
        file_path=str(file_path),
        first_character_index=2,
        last_character_index=6,
    )

    source_text = read_source_text(source)

    assert source_text == "2345"


def test_read_source_text_handles_missing_file(
    tmp_path: Path,
) -> None:
    """Test returning empty text for a missing source file."""

    source = MinimalSource(
        file_path=str(tmp_path / "missing.txt"),
        first_character_index=0,
        last_character_index=10,
    )

    source_text = read_source_text(source)

    assert source_text == ""


def test_read_source_text_handles_invalid_range(
    tmp_path: Path,
) -> None:
    """Test returning empty text for an invalid range."""

    file_path = tmp_path / "example.txt"
    file_path.write_text(
        "Short text",
        encoding="utf-8",
    )

    source = MinimalSource(
        file_path=str(file_path),
        first_character_index=20,
        last_character_index=30,
    )

    source_text = read_source_text(source)

    assert source_text == ""


def test_build_context_includes_source_metadata(
    tmp_path: Path,
) -> None:
    """Test building context with source text and metadata."""

    file_path = tmp_path / "example.py"
    file_path.write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )

    source = MinimalSource(
        file_path=str(file_path),
        first_character_index=0,
        last_character_index=31,
    )

    context = build_context([source])

    assert "[Source 1]" in context
    assert f"File: {file_path}" in context
    assert "Characters: 0-31" in context
    assert "def add(a, b):" in context


def test_build_context_respects_character_limit(
    tmp_path: Path,
) -> None:
    """Test limiting the generated context size."""

    source_path = tmp_path / "large.txt"
    source_text = "context " * 1000
    source_path.write_text(
        source_text,
        encoding="utf-8",
    )

    source = MinimalSource(
        file_path=str(source_path),
        first_character_index=0,
        last_character_index=len(source_text),
    )

    context = build_context(
        sources=[source] * 100,
        max_context_characters=500,
    )

    assert len(context) <= 500


def test_build_messages_contains_context_and_question() -> None:
    """Test building grounded system and user messages."""

    messages = build_messages(
        question="What is LoRA?",
        context="[Source 1]\nLoRA is an adapter technique.",
    )

    assert len(messages) == 2
    assert messages[0].type == "system"
    assert messages[1].type == "human"
    assert "Use only the retrieved context" in (
        messages[0].content
    )
    assert "What is LoRA?" in messages[1].content
    assert "LoRA is an adapter technique." in (
        messages[1].content
    )
    assert "/no_think" in messages[1].content


def test_generate_answers_returns_batch(
    tmp_path: Path,
) -> None:
    """Test generating answers for a result batch."""

    source_path = tmp_path / "source.txt"
    source_text = "Information about vLLM."
    source_path.write_text(
        source_text,
        encoding="utf-8",
    )

    source = MinimalSource(
        file_path=str(source_path),
        first_character_index=0,
        last_character_index=len(source_text),
    )

    student_results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="Question one?",
                retrieved_sources=[source],
            ),
            MinimalSearchResults(
                question_id="q2",
                question="Question two?",
                retrieved_sources=[source],
            ),
        ],
        k=1,
    )

    fake_model = FakeListChatModel(
        responses=[
            "Answer one.",
            "Answer two.",
        ]
    )

    result = generate_answers(
        student_results=student_results,
        model=fake_model,
    )

    assert result.k == 1
    assert len(result.search_results) == 2
    assert result.search_results[0].answer == (
        "Answer one."
    )
    assert result.search_results[1].answer == (
        "Answer two."
    )


def test_clean_generated_answer_removes_thinking() -> None:
    answer = (
        "<think>\nHidden reasoning\n</think>\n\n"
        "Final answer."
    )

    result = clean_generated_answer(answer)

    assert result == "Final answer."
