"""Optional numerical alignment against Hugging Face Llama 3.2.

Skipped unless weights are available locally / via login.
Force with: ARCH_HF_ALIGN=1
"""

from __future__ import annotations

import os

import pytest
import torch

from arche.load_weights import DEFAULT_MODEL_ID, from_pretrained, load_hf_state_dict_into_llama
from arche.model import Llama
from arche.config import Llama32Config


def _hf_available() -> bool:
    if os.environ.get("ARCH_HF_ALIGN") == "0":
        return False
    try:
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig

        AutoConfig.from_pretrained(DEFAULT_MODEL_ID)
        # Probe a tiny config file rather than the full weights when possible.
        hf_hub_download(DEFAULT_MODEL_ID, "config.json")
        return True
    except Exception:
        return os.environ.get("ARCH_HF_ALIGN") == "1"


@pytest.mark.integration
@pytest.mark.skipif(not _hf_available(), reason="Llama 3.2 weights / HF auth not available")
def test_arche_matches_hf_greedy_logits_on_short_prompt():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)
    hf = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    arche = from_pretrained(DEFAULT_MODEL_ID, device="cpu", dtype=torch.float32)

    ids = torch.tensor([tok.encode("The cat sat")])
    with torch.no_grad():
        hf_logits = hf(ids).logits
        arche_logits, _ = arche(ids)

    # Allow small drift from RoPE buffer caps / float paths; shapes and argmax must agree.
    assert arche_logits.shape == hf_logits.shape
    assert torch.equal(arche_logits.argmax(dim=-1), hf_logits.argmax(dim=-1))
    max_abs = (arche_logits.float() - hf_logits.float()).abs().max().item()
    assert max_abs < 0.15, f"logits drifted too far: max_abs={max_abs}"


def test_mapped_fake_hf_state_is_bit_exact():
    """Alignment plumbing works even without downloading Meta weights."""
    cfg = Llama32Config(
        name="tiny",
        vocab_size=32,
        num_layers=1,
        embed_dim=16,
        intermediate_dim=32,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        rope_theta=10_000.0,
        rope_scale_factor=1.0,
        norm_eps=1e-5,
        tie_word_embeddings=True,
    )
    src = Llama(cfg)
    dst = Llama(cfg)
    with torch.no_grad():
        for p in dst.parameters():
            p.zero_()
    hf: dict[str, torch.Tensor] = {}
    for key, value in src.state_dict().items():
        if key == "lm_head.weight":
            continue
        hf[f"model.{key}"] = value.clone()
    load_hf_state_dict_into_llama(dst, hf, strict=True)
    ids = torch.randint(0, 32, (1, 4))
    with torch.no_grad():
        a, _ = src(ids)
        b, _ = dst(ids)
    assert torch.equal(a, b)
