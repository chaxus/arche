"""Load Hugging Face Llama 3.2 weights into the hand-written ``Llama`` module."""

from __future__ import annotations

from typing import Any

import torch

from arche.config import Llama32Config, llama32_1b, llama32_3b
from arche.model import Llama

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"

_HF_PREFIX = "model."


def config_for_model_id(model_id: str) -> Llama32Config:
    lower = model_id.lower()
    if "3b" in lower:
        return llama32_3b()
    return llama32_1b()


def hf_to_arche_key(key: str) -> str | None:
    """Map a Hugging Face state-dict key onto our ``Llama`` module names.

    Returns ``None`` for keys we intentionally ignore (buffers, unused).
    """
    if key == "lm_head.weight":
        return "lm_head.weight"
    if not key.startswith(_HF_PREFIX):
        return None
    rest = key[len(_HF_PREFIX) :]
    if rest == "embed_tokens.weight":
        return "embed_tokens.weight"
    if rest == "norm.weight":
        return "norm.weight"
    if rest.startswith("layers."):
        # HF and arche already share the same nested names under layers.*
        return rest
    return None


def load_hf_state_dict_into_llama(
    model: Llama,
    hf_state: dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Copy HF tensors into ``model``. Returns (missing, unexpected) after load."""
    mapped: dict[str, torch.Tensor] = {}
    for src_key, tensor in hf_state.items():
        dst_key = hf_to_arche_key(src_key)
        if dst_key is None:
            continue
        # When embeddings are tied, prefer embed_tokens and skip a duplicate lm_head copy.
        if (
            dst_key == "lm_head.weight"
            and model.cfg.tie_word_embeddings
            and "embed_tokens.weight" in {hf_to_arche_key(k) for k in hf_state}
        ):
            continue
        mapped[dst_key] = tensor

    if model.cfg.tie_word_embeddings and "embed_tokens.weight" in mapped:
        # Ensure the shared parameter is filled; lm_head points at the same storage.
        pass

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    missing = [k for k in missing if not (model.cfg.tie_word_embeddings and k == "lm_head.weight")]
    if strict and (missing or unexpected):
        raise RuntimeError(f"weight load mismatch: missing={missing} unexpected={unexpected}")
    return list(missing), list(unexpected)


def from_pretrained(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = torch.float32,
) -> Llama:
    """Build our ``Llama`` and fill it from a Hugging Face checkpoint."""
    from transformers import AutoModelForCausalLM

    from arche.device import pick_device

    if device is None:
        target = pick_device()
    elif isinstance(device, str):
        target = pick_device(device)
    else:
        target = device

    cfg = config_for_model_id(model_id)
    model = Llama(cfg)
    hf = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    load_hf_state_dict_into_llama(model, hf.state_dict(), strict=True)
    del hf
    model.to(device=target)
    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()
    return model


def try_from_pretrained(model_id: str = DEFAULT_MODEL_ID, **kwargs: Any) -> Llama:
    """Same as ``from_pretrained``, but re-raises with a friendlier gated-model hint."""
    try:
        return from_pretrained(model_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 — teaching / CLI surface
        raise RuntimeError(
            "Could not load Hugging Face weights. Accept the model license, "
            "run `huggingface-cli login`, and optionally set "
            "HF_ENDPOINT=https://hf-mirror.com"
        ) from exc
