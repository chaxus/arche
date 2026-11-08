"""Paged KV cache — teaching stand-in for vLLM-style block tables."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from arche.cache import KVCache


@dataclass
class PagedKVPool:
    """Shared physical pages for many sequences.

    Layout per page: ``(num_kv_heads, page_size, head_dim)``.
    """

    k_pages: torch.Tensor
    v_pages: torch.Tensor
    page_size: int
    free: list[int] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> PagedKVPool:
        device = device or torch.device("cpu")
        k = torch.zeros(num_pages, num_kv_heads, page_size, head_dim, device=device, dtype=dtype)
        v = torch.zeros_like(k)
        return cls(k_pages=k, v_pages=v, page_size=page_size, free=list(range(num_pages)))

    def alloc(self) -> int:
        if not self.free:
            raise RuntimeError("paged KV pool exhausted")
        return self.free.pop()

    def free_page(self, page: int) -> None:
        self.free.append(page)


@dataclass
class BlockTable:
    """Logical → physical page map for one sequence."""

    pages: list[int] = field(default_factory=list)
    length: int = 0


def append_kv(
    pool: PagedKVPool,
    table: BlockTable,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Append one step of K/V.

    ``k`` / ``v`` shape: ``(num_kv_heads, head_dim)`` or ``(1, num_kv_heads, 1, head_dim)``.
    """
    if k.ndim == 4:
        k = k[0, :, 0, :]
        v = v[0, :, 0, :]
    assert k.ndim == 2 and v.ndim == 2
    slot = table.length % pool.page_size
    if slot == 0:
        table.pages.append(pool.alloc())
    page = table.pages[-1]
    pool.k_pages[page, :, slot, :] = k
    pool.v_pages[page, :, slot, :] = v
    table.length += 1


def gather_kv(pool: PagedKVPool, table: BlockTable) -> KVCache:
    """Materialize a contiguous ``KVCache`` for attention (gather from pages)."""
    if table.length == 0:
        raise ValueError("empty block table")
    heads = pool.k_pages.shape[1]
    dim = pool.k_pages.shape[-1]
    k = torch.empty(1, heads, table.length, dim, device=pool.k_pages.device, dtype=pool.k_pages.dtype)
    v = torch.empty_like(k)
    for i in range(table.length):
        page = table.pages[i // pool.page_size]
        slot = i % pool.page_size
        k[0, :, i, :] = pool.k_pages[page, :, slot, :]
        v[0, :, i, :] = pool.v_pages[page, :, slot, :]
    return KVCache(k=k, v=v)


def append_sequence(
    pool: PagedKVPool,
    table: BlockTable,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
) -> None:
    """Append a full prefill: ``k_seq`` / ``v_seq`` are ``(1, kv_heads, seq, head_dim)``."""
    seq = k_seq.shape[-2]
    for t in range(seq):
        append_kv(pool, table, k_seq[:, :, t : t + 1, :], v_seq[:, :, t : t + 1, :])
