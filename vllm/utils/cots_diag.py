# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""§1c.24/§1c.25: env-gated NVTX helper for COTS attribution.

`VLLM_COTS_DIAG=1` enables NVTX `range_push`/`range_pop` pairs at the
boundaries the §1c.24 / §1c.25 diagnostics use. When the flag is unset
(production default), `nvtx_range` is a near-no-op contextmanager —
no NVTX calls, no exception-safe wrapper cost beyond the `yield`.

Centralized so each call site reads the env once at module import,
not on every entry, and so the gate-check pattern is consistent
across `cots.py`, `cots_ops.py`, `gpu_model_runner.py`,
`cudagraph_utils.py`, and `latency.py`.
"""

from __future__ import annotations

import contextlib
import os

import torch

# Single env read at first import.
ENABLED = os.environ.get("VLLM_COTS_DIAG", "0") == "1"


@contextlib.contextmanager
def nvtx_range(name: str):
    """Env-gated NVTX range. Production-default: no NVTX calls fire,
    just a `yield`. Use at boundaries the COTS diagnostic timeline
    cares about (model-forward, replay, attention, sampling, etc.)."""
    if not ENABLED:
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
    if ENABLED:
        torch.cuda.nvtx.range_push(name)


def pop() -> None:
    if ENABLED:
        torch.cuda.nvtx.range_pop()
