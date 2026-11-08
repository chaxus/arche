"""Helpers for numerical alignment against Hugging Face Llama 3.2."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from arche.load_weights import DEFAULT_MODEL_ID, from_pretrained


@dataclass
class AlignReport:
    model_id: str
    prompt: str
    max_abs: float
    argmax_match: bool
    shape: tuple[int, ...]


def hf_weights_reachable(model_id: str = DEFAULT_MODEL_ID) -> bool:
    """True when config (and ideally weights) can be resolved without forcing a full download failure."""
    if os.environ.get("ARCH_HF_ALIGN") == "0":
        return False
    try:
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig

        AutoConfig.from_pretrained(model_id)
        hf_hub_download(model_id, "config.json")
        return True
    except Exception:
        return False


def compare_logits_to_hf(
    prompt: str = "The cat sat",
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> AlignReport:
    """Load HF + arche, compare greedy logits on a short prompt."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    hf = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval()
    arche = from_pretrained(model_id, device=device, dtype=dtype)

    ids = torch.tensor([tok.encode(prompt)], device=device)
    with torch.no_grad():
        hf_logits = hf(ids.to("cpu" if next(hf.parameters()).device.type == "cpu" else device)).logits
        # Keep HF on its device; move to CPU for a stable compare.
        hf_logits = hf_logits.float().cpu()
        arche_logits, _ = arche(ids.to(device))
        arche_logits = arche_logits.float().cpu()

    max_abs = (arche_logits - hf_logits).abs().max().item()
    argmax_match = bool(torch.equal(arche_logits.argmax(dim=-1), hf_logits.argmax(dim=-1)))
    return AlignReport(
        model_id=model_id,
        prompt=prompt,
        max_abs=max_abs,
        argmax_match=argmax_match,
        shape=tuple(arche_logits.shape),
    )


def run_align_or_explain(
    prompt: str = "The cat sat",
    *,
    model_id: str = DEFAULT_MODEL_ID,
    max_abs_tol: float = 0.15,
) -> AlignReport | None:
    """Chapter/CLI entry: return a report, or ``None`` when weights are unavailable."""
    if not hf_weights_reachable(model_id) and os.environ.get("ARCH_HF_ALIGN") != "1":
        return None
    report = compare_logits_to_hf(prompt, model_id=model_id)
    if not report.argmax_match:
        raise AssertionError(f"argmax mismatch vs HF for prompt={prompt!r}")
    if report.max_abs >= max_abs_tol:
        raise AssertionError(f"logits drifted too far: max_abs={report.max_abs}")
    return report
