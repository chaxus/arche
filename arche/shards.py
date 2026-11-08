"""Synthetic sharded checkpoint helpers (index.json + shard files)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


def write_sharded_state(
    directory: str | Path,
    state: dict[str, torch.Tensor],
    *,
    shard_size: int = 2,
    prefix: str = "model",
) -> Path:
    """Split ``state`` into ``prefix-00001-of-N.pt`` shards + ``index.json``.

    Teaching stand-in for Hugging Face ``model.safetensors.index.json`` layouts.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    keys = list(state.keys())
    if not keys:
        raise ValueError("empty state")
    n_shards = max(1, (len(keys) + shard_size - 1) // shard_size)
    weight_map: dict[str, str] = {}
    for shard_i in range(n_shards):
        chunk = keys[shard_i * shard_size : (shard_i + 1) * shard_size]
        name = f"{prefix}-{shard_i + 1:05d}-of-{n_shards:05d}.pt"
        payload = {k: state[k].detach().cpu() for k in chunk}
        torch.save(payload, directory / name)
        for k in chunk:
            weight_map[k] = name
    index = {"metadata": {"total_shards": n_shards}, "weight_map": weight_map}
    index_path = directory / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index_path


def load_sharded_state(directory: str | Path) -> dict[str, torch.Tensor]:
    """Load every tensor referenced by ``index.json`` in ``directory``."""
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    by_file: dict[str, list[str]] = {}
    for key, fname in weight_map.items():
        by_file.setdefault(fname, []).append(key)
    out: dict[str, torch.Tensor] = {}
    for fname, keys in by_file.items():
        shard = torch.load(directory / fname, map_location="cpu", weights_only=False)
        for key in keys:
            out[key] = shard[key]
    return out


def partition_keys(keys: Iterable[str], shard_size: int) -> list[list[str]]:
    items = list(keys)
    return [items[i : i + shard_size] for i in range(0, len(items), shard_size)] or [[]]
