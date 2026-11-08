"""ep00 — Check that your computer can run arche chapters.

If you are new: you do not need to understand this file yet.
Just run it and look for the OK line at the end.

Run:
    python chapters/00-start/ep00_check_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    print("arche setup check")
    print(f"  repo root     = {ROOT}")
    print(f"  Python        = {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 10):
        print("FAIL: need Python 3.10 or newer")
        raise SystemExit(1)

    try:
        import torch  # noqa: F401

        print(f"  torch import  = ok ({torch.__version__})")
    except Exception as exc:  # noqa: BLE001 — beginner-facing
        print("FAIL: could not import torch")
        print(f"  {type(exc).__name__}: {exc}")
        print("  Fix: activate your venv, then: pip install -r requirements.txt")
        raise SystemExit(1) from exc

    print()
    print("OK — continue to Part 0 on the site, then ep01_overview.py")


if __name__ == "__main__":
    main()
