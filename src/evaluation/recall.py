from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


MIN_SOURCE_IOU = 0.05


def sources_overlap(
    correct: MinimalSource,
    retrieved: MinimalSource,
) -> bool:
    """Check whether two sources have enough overlap."""

    if correct.file_path != retrieved.file_path:
        return False

    intersection = max(
        0,
        min(
            correct.last_character_index,
            retrieved.last_character_index,
        )
        - max(
            correct.first_character_index,
            retrieved.first_character_index,
        ),
    )

    union = (
        max(
            correct.last_character_index,
            retrieved.last_character_index,
        )
        - min(
            correct.first_character_index,
            retrieved.first_character_index,
        )
    )

    return (
        union > 0
        and intersection / union >= MIN_SOURCE_IOU
    )


def calculate_recall_at_k(
    student_results: StudentSearchResults,
    dataset: RagDataset,
    k: int,
) -> float:
    """Calculate average recall at k."""

    if k < 0:
        raise ValueError("k cannot be negative")

    results_by_id = {
        result.question_id: result
        for result in student_results.search_results
    }

    recalls: list[float] = []

    for question in dataset.rag_questions:
        if (
            not isinstance(question, AnsweredQuestion)
            or not question.sources
        ):
            continue

        result = results_by_id.get(question.question_id)

        if result is None:
            retrieved_sources: list[MinimalSource] = []
        else:
            retrieved_sources = result.retrieved_sources[:k]

        found_sources = 0

        for correct_source in question.sources:
            if any(
                sources_overlap(correct_source, retrieved_source)
                for retrieved_source in retrieved_sources
            ):
                found_sources += 1

        recalls.append(
            found_sources / len(question.sources)
        )

    if not recalls:
        raise ValueError("Dataset contains no answered questions")

    return sum(recalls) / len(recalls)
