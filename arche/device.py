"""Device helpers for CPU / CUDA / MPS."""

from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Choose a torch device.

    ``prefer`` may be ``cpu``, ``cuda``, ``mps``, or ``None`` (auto).
    """
    if prefer:
        device = torch.device(prefer)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device | None = None) -> str:
    device = device or pick_device()
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        return f"cuda:{idx} ({name})"
    return str(device)
