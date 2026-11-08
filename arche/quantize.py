"""Simple weight-only INT8 quantization for Linear layers (educational)."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from arche.model import Llama


class Int8Linear(nn.Module):
    """Store weights as int8 + scale; dequantize on the fly in float compute."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        # Per-output-channel absmax scaling.
        w = weight.detach().float()
        scale = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        q = torch.round(w / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        self.register_buffer("weight_int8", q)
        self.register_buffer("scale", scale)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_int8.float() * self.scale.unsqueeze(1)
        return torch.nn.functional.linear(x, w.to(dtype=x.dtype), self.bias)


def quantize_linear_modules(module: nn.Module) -> int:
    """Replace ``nn.Linear`` children with ``Int8Linear`` in-place. Returns count."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.bias is None:
            setattr(module, name, Int8Linear(child.weight, child.bias))
            replaced += 1
        else:
            replaced += quantize_linear_modules(child)
    return replaced


def quantize_llama_int8(model: Llama) -> Llama:
    """Return a copy of ``model`` with Linear layers weight-only INT8 quantized."""
    qmodel = deepcopy(model)
    quantize_linear_modules(qmodel)
    qmodel.eval()
    return qmodel
