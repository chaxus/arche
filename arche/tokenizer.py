"""Thin wrapper around the official Llama 3.2 tokenizer."""

from __future__ import annotations

from typing import Any

DEFAULT_TOKENIZER_ID = "meta-llama/Llama-3.2-1B"


def load_tokenizer(model_id: str = DEFAULT_TOKENIZER_ID) -> Any:
    """Load the Hugging Face tokenizer for Llama 3.2.

    Requires accepting the model license and ``huggingface-cli login``.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)
