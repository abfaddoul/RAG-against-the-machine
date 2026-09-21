from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel


ModelType = TypeVar("ModelType", bound=BaseModel)


def save_json_model(model: BaseModel, path: str | Path) -> None:
    """Save a Pydantic model to a formatted JSON file."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        model.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def load_json_model(
    path: str | Path,
    model_class: type[ModelType],
) -> ModelType:
    """Load and validate a Pydantic model from a JSON file."""

    file_path = Path(path)
    json_text = file_path.read_text(encoding="utf-8")
    return model_class.model_validate_json(json_text)
