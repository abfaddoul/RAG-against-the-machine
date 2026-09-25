from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import cast


SYSTEM_INSTRUCTION = (
    "You answer questions about the vLLM codebase. "
    "Use only the retrieved context provided by the user. "
    "Do not use outside knowledge or invent information. "
    "If the context does not contain enough information, "
    "say that the retrieved sources are insufficient. "
    "Keep the answer concise and grounded."
)


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        SYSTEM_INSTRUCTION,
    ),
    (
        "human",
        "Retrieved context:\n"
        "{context}\n\n"
        "Question:\n"
        "{question}\n\n"
        "Answer using only the retrieved context. /no_think",
    ),
])


def build_messages(
    question: str,
    context: str,
) -> list[BaseMessage]:
    """Build chat messages for grounded answer generation."""

    return cast(
        list[BaseMessage],
        RAG_PROMPT.format_messages(
            question=question,
            context=context,
        ),
    )
