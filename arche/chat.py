"""Chat-template helpers on top of the official tokenizer."""

from __future__ import annotations

from typing import Any


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render a chat transcript with the tokenizer's template when available."""
    # Prefer HF templates only when ``chat_template`` is configured; otherwise
    # a stub ``apply_chat_template`` would recurse into this helper forever.
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    # Fallback for tokenizers without a chat template.
    parts: list[str] = []
    for msg in messages:
        parts.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
    if add_generation_prompt:
        parts.append("assistant:")
    return "\n".join(parts)


def encode_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> list[int]:
    text = apply_chat_template(
        tokenizer, messages, add_generation_prompt=add_generation_prompt
    )
    return list(tokenizer.encode(text))
