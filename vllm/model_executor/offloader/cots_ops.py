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

§1c.19 split (see `phase1c_findings.md §1c.19`): the registry stores
the `CotsCpuInfer` pybind handle DIRECTLY by runner_id, NOT a
`NativeCotsRunner` instance. The compile-visible runner is a thin
facade with only pickleable state (runner_id, task_id map, flags);
the unpicklable C++ handle lives here. Custom op impls and the
offloader's install/teardown helpers all dereference the registry.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import torch

from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    # Type-only import; avoids forcing _cots_C at module load on
    # non-CUDA builds.
    from vllm import _cots_C  # noqa: F401

# Module-private infer registry. Strong refs (NOT weak) — the registry
# IS the storage for the `CotsCpuInfer` instance. NativeCotsRunner's
# `__del__` / `close()` is the only thing that removes entries; if a
# runner is GC'd without close() the __del__ unregisters there.
_COTS_INFER: dict[int, Any] = {}
_NEXT_RUNNER_ID = itertools.count(1)


def _register_infer(infer: Any) -> int:
    """Register a `CotsCpuInfer` instance and return a fresh runner_id.
    The registry takes ownership of the strong reference; the caller
    should retain only the runner_id. See §1c.19 for the rationale."""
    rid = next(_NEXT_RUNNER_ID)
    _COTS_INFER[rid] = infer
    return rid


def _unregister_infer(runner_id: int) -> None:
    """Drop the registry entry for a runner. Idempotent."""
    _COTS_INFER.pop(runner_id, None)


def _lookup_infer(runner_id: int, op_name: str) -> Any:
    """Resolve runner_id → `CotsCpuInfer` instance. Raises a clear
    error if the runner was already torn down."""
    infer = _COTS_INFER.get(runner_id)
    if infer is None:
        raise RuntimeError(
            f"{op_name}: runner_id={runner_id} not in registry "
            f"(known ids: {list(_COTS_INFER.keys())}). The owning "
            f"NativeCotsRunner was likely torn down before its "
            f"in-flight ops drained."
        )
    return infer


# Offloader-side install/teardown helpers. These all run OUTSIDE the
# compiled forward path, so they can dereference the pybind handle
# freely. They exist so the runner facade itself never has to hold
# the handle on its `__dict__`.


def install_infer(
    runner_id: int,
    n_slabs: int,
    scratch_max_tokens: int,
    scratch_max_intermediate_per_half: int,
) -> None:
    """Allocate the C++ slab pool. Called once at offloader post_init."""
    infer = _lookup_infer(runner_id, "install_infer")
    infer.install(
        n_slabs=int(n_slabs),
        scratch_max_tokens=int(scratch_max_tokens),
        scratch_max_intermediate_per_half=int(scratch_max_intermediate_per_half),
    )


def populate_slab_via_spec(
    runner_id: int,
    spec: Any,
    task_id: int,
    *,
    dry_run: bool,
) -> None:
    """Populate slot `task_id` via the spec's `populate(infer, ...)`
    method. The spec carries the per-op pointer + stride layout (QKV
    vs MLP vs dryrun); the helper just hands it the resolved infer."""
    infer = _lookup_infer(runner_id, "populate_slab_via_spec")
    spec.populate(infer, task_id, dry_run=dry_run)


def set_worker_affinity(runner_id: int, mask: int) -> None:
    """Pin the worker thread to a CPU set (uint64 bitmask). One-shot
    call from `CotsOffloader.post_init` after install."""
    infer = _lookup_infer(runner_id, "set_worker_affinity")
    infer.set_worker_affinity(int(mask))


def sync_blocking(runner_id: int) -> None:
    """Drain any in-flight worker task synchronously. Called from
    `NativeCotsRunner.close()`."""
    infer = _COTS_INFER.get(runner_id)
    if infer is None:
        # Already torn down — nothing to drain.
        return
    infer.sync_blocking()


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
    infer = _lookup_infer(runner_id, "cots_submit_gemm")
    stream = torch.cuda.current_stream().cuda_stream
    infer.submit_on_stream(task_id, num_tokens, stream)


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
    infer = _lookup_infer(runner_id, "cots_sync_then_uva")
    stream = torch.cuda.current_stream().cuda_stream
    infer.sync_on_stream(stream)
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
