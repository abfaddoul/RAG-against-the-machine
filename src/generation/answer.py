import re

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.messages import BaseMessage

from src.generation.context import build_context
from src.generation.model import load_chat_model
from src.generation.prompt import build_messages
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from tqdm import tqdm


THINK_PATTERN = re.compile(
    r"<think>.*?</think>",
    flags=re.DOTALL,
)


def extract_message_text(message: BaseMessage) -> str:
    """Extract textual content from a model response."""

    if not isinstance(message.content, str):
        raise TypeError(
            "Chat model returned non-text content"
        )

    return message.content


def clean_generated_answer(answer: str) -> str:
    """Remove Qwen thinking tags from an answer."""

    cleaned_answer = THINK_PATTERN.sub("", answer)
    return cleaned_answer.strip()


def generate_answer(
    search_result: MinimalSearchResults,
    model: BaseChatModel | None = None,
) -> MinimalAnswer:
    """Generate an answer for one retrieved search result."""

    context = build_context(
        search_result.retrieved_sources
    )

    messages = build_messages(
        question=search_result.question,
        context=context,
    )

    chat_model = (
        model
        if model is not None
        else load_chat_model()
    )

    response = chat_model.invoke(messages)

    answer = clean_generated_answer(
        extract_message_text(response)
    )

    return MinimalAnswer(
        question_id=search_result.question_id,
        question=search_result.question,
        retrieved_sources=(
            search_result.retrieved_sources
        ),
        answer=answer,
    )


def generate_answers(
    student_results: StudentSearchResults,
    model: BaseChatModel | None = None,
) -> StudentSearchResultsAndAnswer:
    """Generate answers for all retrieved search results."""

    chat_model = (
        model
        if model is not None
        else load_chat_model()
    )

    answers: list[MinimalAnswer] = []

    for search_result in tqdm(
        student_results.search_results,
        desc="Generating answers",
        unit="question",
    ):
        answer = generate_answer(
            search_result=search_result,
            model=chat_model,
        )
        answers.append(answer)

    return StudentSearchResultsAndAnswer(
        search_results=answers,
        k=student_results.k,
    )
