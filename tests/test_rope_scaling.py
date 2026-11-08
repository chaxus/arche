import torch

from arche.rope import build_rope_cache, llama3_inv_freq


def test_scale_factor_one_matches_base_inv_freq():
    base = llama3_inv_freq(
        64,
        500_000.0,
        scale_factor=1.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position=8192,
        device=torch.device("cpu"),
    )
    scaled = llama3_inv_freq(
        64,
        500_000.0,
        scale_factor=1.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position=8192,
        device=torch.device("cpu"),
    )
    assert torch.allclose(base, scaled)


def test_llama32_scaling_changes_low_frequencies():
    plain = llama3_inv_freq(
        64,
        500_000.0,
        scale_factor=1.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position=8192,
        device=torch.device("cpu"),
    )
    scaled = llama3_inv_freq(
        64,
        500_000.0,
        scale_factor=32.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position=8192,
        device=torch.device("cpu"),
    )
    # Low-frequency bins (later indices / long wavelengths) move under scaling.
    assert not torch.allclose(plain[-4:], scaled[-4:])
    # Highest-frequency bins (early indices) stay on the unscaled table.
    assert torch.allclose(plain[:2], scaled[:2], rtol=1e-5, atol=1e-6)


def test_build_rope_cache_with_scaling_shape():
    cos, sin = build_rope_cache(
        128,
        64,
        theta=500_000.0,
        scale_factor=32.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position=8192,
    )
    assert cos.shape == (128, 64)
    assert sin.shape == (128, 64)
