# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated diagnostic helpers for COTS attribution.

Split diagnostic gates:

* `VLLM_COTS_NVTX=1` enables NVTX `range_push`/`range_pop` pairs.
* `VLLM_COTS_COUNTERS=1` enables cheap C++/Python counters.

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


# Single env reads at first import. `ENABLED` is the "some COTS diagnostic is
# active" umbrella for callers that only need a coarse gate.
NVTX_ENABLED = _flag("VLLM_COTS_NVTX")
COUNTERS_ENABLED = _flag("VLLM_COTS_COUNTERS")
ENABLED = NVTX_ENABLED or COUNTERS_ENABLED


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
