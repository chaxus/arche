import torch

from arche.mlp import SwiGLU


def test_swiglu_preserves_shape():
    mlp = SwiGLU(64, 128)
    x = torch.randn(2, 5, 64)
    assert mlp(x).shape == x.shape
