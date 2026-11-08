"""ep21 — Align arche logits against Hugging Face Llama 3.2.

Run:
    python chapters/01-inference/ep21_hf_align.py

Needs gated weights (``huggingface-cli login``). Optional:
    export HF_ENDPOINT=https://hf-mirror.com
    ARCH_HF_ALIGN=1 python chapters/01-inference/ep21_hf_align.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arche.align_hf import hf_weights_reachable, run_align_or_explain
from arche.load_weights import DEFAULT_MODEL_ID


def main() -> None:
    print(f"model: {DEFAULT_MODEL_ID}")
    if not hf_weights_reachable():
        print("HF weights not reachable (offline / gated / no login).")
        print("Accept the license, run `huggingface-cli login`, then retry.")
        print("Synthetic mapping still covered by tests/test_align_hf.py::test_mapped_fake_hf_state_is_bit_exact")
        return
    report = run_align_or_explain()
    assert report is not None
    print("alignment ok")
    print(f"  prompt       = {report.prompt!r}")
    print(f"  shape        = {report.shape}")
    print(f"  argmax_match = {report.argmax_match}")
    print(f"  max_abs      = {report.max_abs:.6f}")


if __name__ == "__main__":
    main()
