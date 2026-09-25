from functools import lru_cache

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
)
from transformers import GenerationConfig


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_NEW_TOKENS = 256


@lru_cache(maxsize=1)
def load_chat_model(
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> ChatHuggingFace:
    """Load and cache the local Qwen chat model."""

    if not model_id.strip():
        raise ValueError("model_id cannot be empty")

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be positive"
        )

    generation_config = GenerationConfig.from_pretrained(
        model_id
    )
    generation_config.max_length = None
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = True
    generation_config.temperature = 0.7
    generation_config.top_p = 0.8
    generation_config.top_k = 20

    llm: HuggingFacePipeline = (
        HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            device=-1,
            model_kwargs={
                "dtype": "auto",
            },
            pipeline_kwargs={
                "generation_config": generation_config,
                "return_full_text": False,
                "clean_up_tokenization_spaces": False,
            },
        )
    )

    chat_model: ChatHuggingFace = ChatHuggingFace(
        llm=llm,
    )

    return chat_model
