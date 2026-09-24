from src.evaluation.recall import (
    calculate_recall_at_k,
    sources_overlap,
)
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def test_sources_overlap_returns_true_for_shared_range() -> None:
    """Test sources that share characters inside the same file."""

    correct_source = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=100,
        last_character_index=200,
    )
    retrieved_source = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=150,
        last_character_index=250,
    )

    assert sources_overlap(
        correct_source,
        retrieved_source,
    ) is True


def test_sources_overlap_returns_false_for_separate_ranges() -> None:
    """Test sources that do not share any character."""

    correct_source = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=100,
        last_character_index=200,
    )
    retrieved_source = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=200,
        last_character_index=300,
    )

    assert sources_overlap(
        correct_source,
        retrieved_source,
    ) is False


def test_sources_overlap_returns_false_for_different_files() -> None:
    """Test overlapping positions belonging to different files."""

    correct_source = MinimalSource(
        file_path="data/raw/first.py",
        first_character_index=100,
        last_character_index=200,
    )
    retrieved_source = MinimalSource(
        file_path="data/raw/second.py",
        first_character_index=100,
        last_character_index=200,
    )

    assert sources_overlap(
        correct_source,
        retrieved_source,
    ) is False


def test_calculate_recall_at_k_uses_top_k_sources() -> None:
    """Test recall calculation using only the first k sources."""

    first_correct_source = MinimalSource(
        file_path="data/raw/first.py",
        first_character_index=10,
        last_character_index=20,
    )
    second_correct_source = MinimalSource(
        file_path="data/raw/second.py",
        first_character_index=30,
        last_character_index=40,
    )

    dataset = RagDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id="q1",
                question="First question",
                sources=[first_correct_source],
                answer="First answer",
            ),
            AnsweredQuestion(
                question_id="q2",
                question="Second question",
                sources=[second_correct_source],
                answer="Second answer",
            ),
        ]
    )

    student_results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="First question",
                retrieved_sources=[
                    MinimalSource(
                        file_path="data/raw/first.py",
                        first_character_index=15,
                        last_character_index=25,
                    ),
                ],
            ),
            MinimalSearchResults(
                question_id="q2",
                question="Second question",
                retrieved_sources=[
                    MinimalSource(
                        file_path="data/raw/wrong.py",
                        first_character_index=0,
                        last_character_index=10,
                    ),
                    MinimalSource(
                        file_path="data/raw/second.py",
                        first_character_index=35,
                        last_character_index=45,
                    ),
                ],
            ),
        ],
        k=2,
    )

    assert calculate_recall_at_k(
        student_results,
        dataset,
        k=1,
    ) == 0.5

    assert calculate_recall_at_k(
        student_results,
        dataset,
        k=2,
    ) == 1.0


def test_sources_overlap_rejects_tiny_intersection() -> None:
    """Test rejecting overlap below the minimum IoU."""

    correct_source = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=100,
        last_character_index=200,
    )
    retrieved_source = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=199,
        last_character_index=299,
    )

    assert sources_overlap(
        correct_source,
        retrieved_source,
    ) is False
