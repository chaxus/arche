"""Optional numerical alignment against Hugging Face Llama 3.2.

Skipped unless weights are available locally / via login.
Force with: ARCH_HF_ALIGN=1
"""

from __future__ import annotations

import os

import pytest
import torch

from arche.align_hf import compare_logits_to_hf, hf_weights_reachable
from arche.config import Llama32Config
from arche.load_weights import DEFAULT_MODEL_ID, load_hf_state_dict_into_llama
from arche.model import Llama


def _should_run_integration() -> bool:
    if os.environ.get("ARCH_HF_ALIGN") == "0":
        return False
    if hf_weights_reachable():
        return True
    return os.environ.get("ARCH_HF_ALIGN") == "1"


@pytest.mark.integration
@pytest.mark.skipif(not _should_run_integration(), reason="Llama 3.2 weights / HF auth not available")
def test_arche_matches_hf_greedy_logits_on_short_prompt():
    report = compare_logits_to_hf("The cat sat", model_id=DEFAULT_MODEL_ID, device="cpu")
    assert report.argmax_match
    assert report.max_abs < 0.15, f"logits drifted too far: max_abs={report.max_abs}"


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


def test_hf_probe_helper_is_boolean():
    assert isinstance(hf_weights_reachable(DEFAULT_MODEL_ID), bool)
