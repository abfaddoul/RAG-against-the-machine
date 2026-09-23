import re


ACRONYM_BOUNDARY = re.compile(
    r"(?<=[A-Z])(?=[A-Z][a-z])"
)
CAMEL_CASE_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])"
)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Convert text and code identifiers into lowercase tokens."""

    normalized_text = ACRONYM_BOUNDARY.sub(" ", text)
    normalized_text = CAMEL_CASE_BOUNDARY.sub(
        " ",
        normalized_text,
    )

    return TOKEN_PATTERN.findall(normalized_text.lower())
