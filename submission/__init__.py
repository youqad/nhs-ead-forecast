"""Package init.

Runs before any submodule import, so this is the right place to apply
process-wide environment fixes that must be set before torch / sklearn /
lightgbm load their respective OpenMP runtimes on macOS. Multiple libomp
copies coexisting in one process cause silent deadlocks during torch
forward passes (see chronos.py).
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402, F401  - load torch's libomp before sklearn's
