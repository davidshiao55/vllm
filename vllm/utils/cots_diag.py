# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""§1c.24/§1c.25: env-gated diagnostic helpers for COTS attribution.

Split diagnostic gates:

* `VLLM_COTS_NVTX=1` enables NVTX `range_push`/`range_pop` pairs.
* `VLLM_COTS_COUNTERS=1` enables cheap C++/Python counters.
* `VLLM_COTS_WAIT_KERNEL_DIAG=1` enables the diagnostic wait kernel.
* `VLLM_COTS_DIAG=1` remains a backward-compatible alias for all three.

Centralized so each call site reads the env once at module import,
not on every entry, and so the gate-check pattern is consistent
across `cots.py`, `cots_ops.py`, `gpu_model_runner.py`,
`cudagraph_utils.py`, and `latency.py`.
"""

from __future__ import annotations

import contextlib
import os

import torch


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


# Single env reads at first import. `ENABLED` is kept as the legacy
# "some COTS diagnostic is active" umbrella for older tests/imports.
LEGACY_ENABLED = _flag("VLLM_COTS_DIAG")
NVTX_ENABLED = LEGACY_ENABLED or _flag("VLLM_COTS_NVTX")
COUNTERS_ENABLED = LEGACY_ENABLED or _flag("VLLM_COTS_COUNTERS")
WAIT_KERNEL_DIAG_ENABLED = LEGACY_ENABLED or _flag("VLLM_COTS_WAIT_KERNEL_DIAG")
ENABLED = NVTX_ENABLED or COUNTERS_ENABLED or WAIT_KERNEL_DIAG_ENABLED


@contextlib.contextmanager
def nvtx_range(name: str):
    """Env-gated NVTX range. Production-default: no NVTX calls fire,
    just a `yield`. Use at boundaries the COTS diagnostic timeline
    cares about (model-forward, replay, attention, sampling, etc.)."""
    if not NVTX_ENABLED:
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def push(name: str) -> None:
    """Manual push for sites that can't easily use a contextmanager
    (e.g., crossing a captured-graph boundary). Pair with `pop()`."""
    if NVTX_ENABLED:
        torch.cuda.nvtx.range_push(name)


def pop() -> None:
    if NVTX_ENABLED:
        torch.cuda.nvtx.range_pop()
