"""CUDA Graph helpers for fixed-shape compute (teaching).

Full Llama decode graphs need static KV arenas (in-place page writes). This
module captures a fixed ``nn.Module`` step — the same API production engines
use once KV writes no longer allocate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from arche.cache import KVCache
from arche.model import Llama


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


@dataclass
class CapturedStep:
    """Replayable CUDA graph for ``module(static_in) -> static_out``."""

    graph: "torch.cuda.CUDAGraph"
    static_in: torch.Tensor
    static_out: torch.Tensor

    def replay(self, x: torch.Tensor) -> torch.Tensor:
        self.static_in.copy_(x)
        self.graph.replay()
        return self.static_out


@torch.no_grad()
def capture_module_step(module: nn.Module, example_input: torch.Tensor) -> CapturedStep:
    """Capture ``module(example_input)`` on CUDA. Input/output shapes stay fixed."""
    if not cuda_available():
        raise RuntimeError("CUDA graphs require a CUDA device")
    device = torch.device("cuda")
    module = module.to(device).eval()
    static_in = example_input.to(device).clone()
    # Warmup
    _ = module(static_in)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    static_out: torch.Tensor
    with torch.cuda.graph(graph):
        static_out = module(static_in)
    return CapturedStep(graph=graph, static_in=static_in, static_out=static_out)


@torch.no_grad()
def generate_with_optional_graph(
    model: Llama,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 8,
    use_cuda_graph: bool = True,
) -> torch.Tensor:
    """Prefill + decode. On CUDA, also warms a tiny captured Linear (API demo).

    Token generation itself stays eager so KV cache ``torch.cat`` stays correct.
    Pair with ``paged_kv`` when you want in-place page writes that *can* be graphed.
    """
    from arche.generate import _sample_next

    model.eval()
    device = next(model.parameters()).device
    out = input_ids.to(device)

    if use_cuda_graph and device.type == "cuda" and cuda_available():
        # Side-channel demo: capture lm_head-shaped matmul API without touching KV.
        probe = nn.Linear(model.cfg.embed_dim, model.cfg.embed_dim, bias=False).to(device)
        probe.weight.data.copy_(torch.eye(model.cfg.embed_dim, device=device))
        captured = capture_module_step(probe, torch.randn(1, model.cfg.embed_dim, device=device))
        _ = captured.replay(torch.randn(1, model.cfg.embed_dim, device=device))

    caches: list[KVCache | None] = [None] * len(model.layers)
    logits, caches = model(out, caches=caches, start_pos=0)
    next_id = int(_sample_next(logits[:, -1, :], temperature=0.0, top_k=None).item())
    out = torch.cat([out, torch.tensor([[next_id]], device=device)], dim=1)
    start = out.shape[1] - 1
    for _ in range(max_new_tokens - 1):
        step = out[:, -1:]
        logits, caches = model(step, caches=caches, start_pos=start)
        start += 1
        next_id = int(_sample_next(logits[:, -1, :], temperature=0.0, top_k=None).item())
        out = torch.cat([out, torch.tensor([[next_id]], device=device)], dim=1)
    return out
