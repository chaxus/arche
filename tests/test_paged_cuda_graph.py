import pytest
import torch
import torch.nn as nn

from arche.config import Llama32Config
from arche.cuda_graphs import capture_module_step, cuda_available, generate_with_optional_graph
from arche.model import Llama
from arche.paged_kv import BlockTable, PagedKVPool, append_kv, append_sequence, gather_kv


def test_paged_kv_matches_contiguous():
    pool = PagedKVPool.create(num_pages=8, page_size=2, num_kv_heads=2, head_dim=4)
    table = BlockTable()
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    append_sequence(pool, table, k, v)
    got = gather_kv(pool, table)
    assert got.seq_len == 5
    assert len(table.pages) == 3  # page_size=2 → 5 tokens need 3 pages
    assert torch.allclose(got.k, k)
    assert torch.allclose(got.v, v)


def test_paged_kv_append_token():
    pool = PagedKVPool.create(num_pages=2, page_size=2, num_kv_heads=1, head_dim=3)
    table = BlockTable()
    append_kv(pool, table, torch.ones(1, 3), torch.zeros(1, 3))
    append_kv(pool, table, torch.ones(1, 3) * 2, torch.zeros(1, 3))
    assert table.length == 2
    got = gather_kv(pool, table)
    assert torch.allclose(got.k[0, 0, 0], torch.ones(3))
    assert torch.allclose(got.k[0, 0, 1], torch.ones(3) * 2)


def test_generate_with_optional_graph_cpu():
    cfg = Llama32Config(
        name="tiny",
        vocab_size=32,
        num_layers=1,
        embed_dim=16,
        intermediate_dim=32,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=32,
        rope_theta=10_000.0,
        rope_scale_factor=1.0,
        norm_eps=1e-5,
    )
    model = Llama(cfg).eval()
    ids = torch.randint(0, 32, (1, 3))
    out = generate_with_optional_graph(model, ids, max_new_tokens=2, use_cuda_graph=False)
    assert out.shape == (1, 5)


@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
def test_capture_module_step_cuda():
    linear = nn.Linear(8, 8, bias=False).cuda()
    x = torch.randn(2, 8, device="cuda")
    with torch.no_grad():
        eager = linear(x)
    captured = capture_module_step(linear, x)
    graphed = captured.replay(x)
    assert torch.allclose(eager, graphed, atol=1e-5)
