import torch

from arche.norms import RMSNorm


def test_rmsnorm_unit_rms_with_unit_weight():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 64)
    y = RMSNorm(64, eps=1e-5)(x)
    rms = y.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rmsnorm_preserves_shape_and_dtype():
    x = torch.randn(1, 3, 32, dtype=torch.float32)
    y = RMSNorm(32)(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
