# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom ops for the COTS native CPU runner — Phase 1c.

Mirrors `prefetch_ops.py`'s pattern: registers two ops via
`direct_register_custom_op` (which lands them under `torch.ops.vllm.*`),
each with a `mutates_args` list that declares barrier-installing
dependencies so torch.compile / CUDA graph capture preserve the overlap
ordering between submit, GPU GEMMs, sync, and UVA copy.

Schemas (see /root/.claude/plans/pleaes-implement-phase1c-in-quizzical-mist.md
"Op schemas with CUDA dispatch anchors AND barrier-installing mutates_args"):

  * vllm.cots_submit_gemm(x_gpu, x_pinned, y_pinned, runner_id, task_id, num_tokens)
      mutates_args=["x_gpu", "y_pinned"]
      x_gpu pins submit BEFORE every GPU GEMM that reads x_gpu.

  * vllm.cots_sync_then_uva(y_pinned, y_gpu, gpu_anchor_a, gpu_anchor_b, runner_id)
      mutates_args=["y_gpu", "gpu_anchor_a", "gpu_anchor_b"]
      Two distinct anchors pin sync AFTER each independent GPU GEMM.

Multi-engine safety: each NativeCotsRunner registers itself in
`_COTS_RUNNERS` (a module-private WeakValueDictionary) under a unique
runner_id; the op impls look up the right runner by id, so two
offloaders (e.g., FastTTS gen + ver) coexist with independent slab
pools and no C++ singleton.
"""

from __future__ import annotations

import itertools
import weakref

import torch

from vllm.utils.torch_utils import direct_register_custom_op

# Module-private runner registry. Weak refs so a runner that's been
# torn down (e.g., engine shutdown) is auto-cleared without leaking.
_COTS_RUNNERS: weakref.WeakValueDictionary[int, object] = weakref.WeakValueDictionary()
_NEXT_RUNNER_ID = itertools.count(1)


def _register_runner(runner: object) -> int:
    """Register a NativeCotsRunner instance and return its runner_id.
    Caller must hold a strong reference to `runner` for the lifetime of
    any in-flight op call (the registry is weak)."""
    rid = next(_NEXT_RUNNER_ID)
    _COTS_RUNNERS[rid] = runner
    return rid


def _unregister_runner(runner_id: int) -> None:
    """Drop the registry entry for a runner. Idempotent."""
    _COTS_RUNNERS.pop(runner_id, None)


def _lookup_runner(runner_id: int, op_name: str) -> object:
    runner = _COTS_RUNNERS.get(runner_id)
    if runner is None:
        raise RuntimeError(
            f"{op_name}: runner_id={runner_id} not in registry "
            f"(known ids: {list(_COTS_RUNNERS.keys())}). The owning "
            f"NativeCotsRunner was likely torn down before its "
            f"in-flight ops drained."
        )
    return runner


# --- vllm.cots_submit_gemm -------------------------------------------------


def _cots_submit_gemm_impl(
    x_gpu: torch.Tensor,
    x_pinned: torch.Tensor,
    y_pinned: torch.Tensor,
    runner_id: int,
    task_id: int,
    num_tokens: int,
) -> None:
    """Real impl: dispatched to the per-runner pybind handle.

    The host-side D2H `x_pinned.copy_(x_gpu, non_blocking=True)` happens
    BEFORE this op is invoked (in NativeCotsRunner.submit_with_d2h);
    torch.compile sees x_pinned as use-after-mutate and keeps that
    ordering. We just enqueue the GEMM task on the current CUDA stream.
    """
    runner = _lookup_runner(runner_id, "cots_submit_gemm")
    stream = torch.cuda.current_stream().cuda_stream
    runner._infer.submit_on_stream(task_id, num_tokens, stream)  # type: ignore[attr-defined]


def _cots_submit_gemm_fake(
    x_gpu: torch.Tensor,
    x_pinned: torch.Tensor,
    y_pinned: torch.Tensor,
    runner_id: int,
    task_id: int,
    num_tokens: int,
) -> None:
    """torch.compile tracing: no side effects, just a barrier."""
    return


# --- vllm.cots_sync_then_uva -----------------------------------------------


def _cots_sync_then_uva_impl(
    y_pinned: torch.Tensor,
    y_gpu: torch.Tensor,
    gpu_anchor_a: torch.Tensor,
    gpu_anchor_b: torch.Tensor,
    runner_id: int,
) -> None:
    """Real impl: schedule the sync host callback then run the UVA copy.

    Bundling sync + uva_copy into one op (instead of two ops registered
    separately) gives torch.compile a single dependency-bearing entry —
    the alternative would let it reorder the Triton UVA copy across
    sync. `gpu_anchor_a` / `gpu_anchor_b` are CUDA tensors that the
    GPU GEMMs produced; mutating them pins this op AFTER both
    independent GEMMs (out_perm, out_pref).
    """
    runner = _lookup_runner(runner_id, "cots_sync_then_uva")
    stream = torch.cuda.current_stream().cuda_stream
    runner._infer.sync_on_stream(stream)  # type: ignore[attr-defined]
    # Lazy import to avoid a top-level circular import (cots.py imports
    # this module via cots_ops and we'd loop on `from .cots import ...`).
    from vllm.model_executor.offloader.cots import uva_copy_into_gpu

    uva_copy_into_gpu(y_pinned, y_gpu)


def _cots_sync_then_uva_fake(
    y_pinned: torch.Tensor,
    y_gpu: torch.Tensor,
    gpu_anchor_a: torch.Tensor,
    gpu_anchor_b: torch.Tensor,
    runner_id: int,
) -> None:
    return


# --- registration ----------------------------------------------------------


def register_cots_offloader_ops() -> None:
    """Register the two custom ops. Idempotent at import time."""
    direct_register_custom_op(
        op_name="cots_submit_gemm",
        op_func=_cots_submit_gemm_impl,
        # x_gpu is the CUDA dispatch anchor AND the ordering pin (see
        # plan §design-decision 6). Mutating it forces every subsequent
        # GPU GEMM that reads x_gpu (F.linear permanent / prefetched) to
        # be ordered after submit. y_pinned is the buffer the worker
        # fills.
        mutates_args=["x_gpu", "y_pinned"],
        fake_impl=_cots_submit_gemm_fake,
    )
    direct_register_custom_op(
        op_name="cots_sync_then_uva",
        op_func=_cots_sync_then_uva_impl,
        # y_gpu is the CUDA dispatch anchor + downstream-scatter dep.
        # gpu_anchor_a / gpu_anchor_b pin sync AFTER each independent
        # GPU GEMM (in QKV, out_perm and out_pref are independent
        # F.linears; one anchor only pins one of them). The runner
        # passes two DISTINCT dummy CUDA tensors when an anchor is
        # absent — never alias.
        mutates_args=["y_gpu", "gpu_anchor_a", "gpu_anchor_b"],
        fake_impl=_cots_sync_then_uva_fake,
    )


# Register at module import time so `torch.ops.vllm.cots_*` exist as
# soon as cots.py imports this module.
register_cots_offloader_ops()
