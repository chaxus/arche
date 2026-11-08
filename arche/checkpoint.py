"""Save / load arche modules (including MoE and MLA primers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(path: str | Path, model: nn.Module, meta: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: nn.Module, *, strict: bool = True) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=strict)
    return dict(payload.get("meta") or {})
