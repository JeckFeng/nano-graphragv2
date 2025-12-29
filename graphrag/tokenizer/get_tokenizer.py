# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Get Tokenizer."""

import logging

from graphrag.config.defaults import ENCODING_MODEL
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.tokenizer.litellm_tokenizer import LitellmTokenizer
from graphrag.tokenizer.tiktoken_tokenizer import TiktokenTokenizer
from graphrag.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def get_tokenizer(
    model_config: LanguageModelConfig | None = None,
    encoding_model: str = ENCODING_MODEL,
) -> Tokenizer:
    """
    Get the tokenizer for the given model configuration or fallback to a tiktoken based tokenizer.

    Args
    ----
        model_config: LanguageModelConfig, optional
            The model configuration. If not provided or model_config.encoding_model is manually set,
            use a tiktoken based tokenizer. Otherwise, use a LitellmTokenizer based on the model name.
            LiteLLM supports token encoding/decoding for the range of models it supports.
        encoding_model: str, optional
            A tiktoken encoding model to use if no model configuration is provided. Only used if a
            model configuration is not provided.

    Returns
    -------
        An instance of a Tokenizer.
    """
    if model_config is not None:
        if model_config.encoding_model.strip() != "":
            # User has manually specified a tiktoken encoding model to use for the provided model configuration.
            return TiktokenTokenizer(encoding_name=model_config.encoding_model)

        try:
            tokenizer = LitellmTokenizer(model_name=model_config.model)
            # Ensure the target model is supported by LiteLLM tokenization; fallback otherwise.
            tokenizer.encode("")
            return tokenizer
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Falling back to tiktoken tokenizer for model %s: %s",
                model_config.model,
                exc,
            )
            encoding_name = model_config.encoding_model or ENCODING_MODEL
            return TiktokenTokenizer(encoding_name=encoding_name)

    return TiktokenTokenizer(encoding_name=encoding_model)
