"""ep03 — Token embedding: table shape, lookup, and one toy train step.

Run:
    python chapters/01-inference/ep03_embedding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from arche import TokenEmbedding, llama32_1b


def main() -> None:
    cfg = llama32_1b()
    demo_vocab = 32
    dim = 8  # tiny for the train demo; real Llama uses cfg.embed_dim
    emb = TokenEmbedding(demo_vocab, dim)
    ids = torch.tensor([[3, 7, 11]])
    x = emb(ids)

    print("1) What the table is")
    print(f"  real Llama vocab_size = {cfg.vocab_size}")
    print(f"  real Llama embed_dim  = {cfg.embed_dim}")
    print(f"  demo table shape      = {tuple(emb.weight.weight.shape)}  # (vocab, dim)")
    print(f"  ids                   = {ids.tolist()}")
    print(f"  gathered shape        = {tuple(x.shape)}  # (batch, seq, dim)")
    print(f"  row 3 (first 4 dims)  = {emb.weight.weight[3, :4].detach().tolist()}")
    print()

    # One SGD step: pretend next-token head is a linear on the last position.
    print("2) How the table is *generated* (toy): random init → train updates rows")
    torch.manual_seed(0)
    head = torch.nn.Linear(dim, demo_vocab, bias=False)
    opt = torch.optim.SGD(list(emb.parameters()) + list(head.parameters()), lr=0.5)
    # context ids → predict next id 5
    ctx = torch.tensor([[3, 7, 11]])
    target = torch.tensor([5])
    before = emb.weight.weight[11].detach().clone()
    opt.zero_grad()
    h = emb(ctx)[:, -1, :]
    loss = F.cross_entropy(head(h), target)
    loss.backward()
    opt.step()
    after = emb.weight.weight[11].detach()
    delta = (after - before).abs().mean().item()
    print(f"  loss                = {loss.item():.4f}")
    print(f"  mean |Δ| on row 11  = {delta:.6f}  # last context id got a gradient")
    print("  (rows never seen in the batch barely move — rare tokens need data)")
    print()

    print("3) What “good” means (checks you can run)")
    print(f"  shape OK            = {tuple(x.shape) == (1, 3, dim)}")
    print(f"  finite              = {torch.isfinite(x).all().item()}")
    print(f"  row i ↔ tokenizer i = hard rule (same vocab); not shown in this toy")
    print()
    print("Modern trunk note: position is RoPE on Q/K later, not a PE table here.")


if __name__ == "__main__":
    main()
