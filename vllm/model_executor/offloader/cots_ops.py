# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom ops for the COTS native CPU runner — Phase 1c.

Mirrors `prefetch_ops.py`'s pattern: registers two ops via
`direct_register_custom_op` (which lands them under `torch.ops.vllm.*`),
each with a `mutates_args` list that declares barrier-installing
dependencies so torch.compile / CUDA graph capture preserve the overlap
ordering between submit, GPU GEMMs, sync, and UVA copy.

Final schemas after the §1c.20 simplification (see
`phase1c_findings.md §1c.20`):

  * vllm.cots_submit_gemm(x_gpu, runner_id, task_id, num_tokens)
      mutates_args=["x_gpu"]
      x_gpu mutated → pins submit BEFORE every GPU GEMM that reads
      x_gpu, AND provides the (submit → sync) edge consumed by
      cots_sync_then_uva's `submit_anchor` read. The C++ impl
      bundles `cudaMemcpyAsync(D2H)` from x_gpu to slab.x_pinned_ptr
      with the host-callback enqueue.

  * vllm.cots_sync_then_uva(y_gpu, gpu_anchor_a, gpu_anchor_b,
                            submit_anchor, runner_id, task_id,
                            num_tokens)
      mutates_args=["y_gpu", "gpu_anchor_a", "gpu_anchor_b"]
      `gpu_anchor_a/_b` pin sync AFTER each independent GPU GEMM.
      `submit_anchor` (read-only, == x_gpu) pins sync AFTER submit.
      The impl reaches the slab's pinned output via
      `CotsCpuInfer.y_pinned_view(task_id, num_tokens)`.

Both ops accept ONLY CUDA tensors and scalar ids — no CPU tensor
arguments. Inductor's functionalization on captured graphs
materializes any CPU view it sees (in the worst case via a GPU
intermediate + blocking GPU→CPU copy that CUDA Graph capture
rejects with cudaErrorStreamCaptureUnsupported), so the design
keeps pinned-buffer addresses entirely on the C++ side.

§1c.19 split (see `phase1c_findings.md §1c.19`): the registry stores
the `CotsCpuInfer` pybind handle DIRECTLY by runner_id, NOT a
`NativeCotsRunner` instance. The compile-visible runner is a thin
facade with only pickleable state (runner_id, task_id map, flags);
the unpicklable C++ handle lives here. Custom op impls and the
offloader's install/teardown helpers all dereference the registry.
"""

from __future__ import annotations

import atexit
import itertools
import os
import sys
from typing import TYPE_CHECKING, Any

import torch

from vllm.utils.cots_diag import ENABLED as _COTS_DIAG_ENABLED
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


def reset_all_counters() -> None:
    """§1c.22: zero every CotsCpuInfer counter currently in the
    registry. Used as the env-gated post-cudagraph-capture hook
    (`VLLM_COTS_RESET_COUNTERS_AFTER_CUDAGRAPH_CAPTURE=1`) so the
    byte-accounting bench artifact reflects ONLY the measured
    replay, not capture-time activity."""
    import contextlib

    for infer in _COTS_INFER.values():
        # Best-effort — a stale infer shouldn't break the reset
        # for the rest.
        with contextlib.suppress(Exception):
            infer.reset_counters()


def set_runtime_num_tokens(runner_id: int, n: int) -> None:
    """§1c.21 live-token plumb-through. Called by
    `CotsOffloader.set_runtime_num_tokens` (a thin override on the
    `BaseOffloader` lifecycle hook) which is invoked by
    `gpu_model_runner.execute_model` BEFORE the
    FULL/PIECEWISE/eager dispatch with the live unpadded token
    count from `scheduler_output.total_num_scheduled_tokens`. The
    worker reads the value on the next host-callback fire and uses
    it for row-count arithmetic instead of the captured bucket
    size."""
    infer = _COTS_INFER.get(runner_id)
    if infer is None:
        # Best-effort: a stale runner_id call here shouldn't crash —
        # just skip. The next custom-op call will surface the missing
        # registry entry with a clearer error.
        return
    infer.set_runtime_num_tokens(int(n))


def sync_blocking(runner_id: int) -> None:
    """Drain any in-flight worker task synchronously. Called from
    `NativeCotsRunner.close()`."""
    infer = _COTS_INFER.get(runner_id)
    if infer is None:
        # Already torn down — nothing to drain.
        return
    infer.sync_blocking()


# --- vllm.cots_submit_gemm -------------------------------------------------
#
# §1c.20 schema: BOTH y_pinned AND x_pinned are intentionally
# excluded from this op's argument list. The earlier intermediate
# schema (which kept x_pinned but dropped y_pinned) still hit
# Inductor's CPU-tensor functionalization on the SUBMIT side: the
# operator's `x_pinned.copy_(x_gpu, non_blocking=True)` was
# expanded into (GPU intermediate → fresh pageable CPU →
# `cpp_fused_copy_slice_view` into the pinned slice), where the
# blocking GPU→CPU step is rejected under CUDA Graph capture. The
# fix bundles the D2H into C++: `submit_on_stream(task_id,
# num_tokens, x_gpu_ptr, stream)` issues `cudaMemcpyAsync(D2H)`
# from x_gpu's data pointer to slab.x_pinned_ptr, then enqueues
# the host callback. The custom op's only Python-visible tensor
# argument is x_gpu (a CUDA tensor — Inductor handles GPU tensors
# natively without CPU/GPU shuffles).
# See `phase1c_findings.md §1c.20`.


def _cots_submit_gemm_impl(
    x_gpu: torch.Tensor,
    runner_id: int,
    task_id: int,
    num_tokens: int,
) -> None:
    """Real impl: dispatched to the per-runner pybind handle.

    §1c.20: bundles the x_gpu → slab.x_pinned_ptr D2H copy WITH the
    host-callback enqueue, all on the current CUDA stream. x_pinned
    is intentionally NOT a Python-side argument — Inductor's
    functionalization materializes any CPU tensor visible in the
    captured graph (in the worst case via a GPU intermediate +
    blocking GPU→CPU copy that CUDA Graph capture rejects). Both
    custom ops are now CUDA-tensors-and-scalars only; the worker
    reaches the pinned input via the slab pointer populated at
    install time.

    Moving the D2H from Python `copy_(non_blocking=True)` to a raw
    `cudaMemcpyAsync` loses Python's automatic shape/stride/dtype
    handling, so we validate `x_gpu` here BEFORE handing the raw
    pointer to C++. The slab's `in_dim` and `num_tokens` together
    determine the byte count; mismatched shapes would silently copy
    the wrong amount of data.
    """
    assert x_gpu.is_cuda, f"cots_submit_gemm: x_gpu must be on CUDA, got {x_gpu.device}"
    assert x_gpu.dtype == torch.bfloat16, (
        f"cots_submit_gemm: x_gpu must be bfloat16 (matches slab dtype), "
        f"got {x_gpu.dtype}"
    )
    assert x_gpu.dim() == 2, (
        f"cots_submit_gemm: x_gpu must be 2D (num_tokens, in_dim); "
        f"got shape {tuple(x_gpu.shape)}"
    )
    assert x_gpu.shape[0] == num_tokens, (
        f"cots_submit_gemm: num_tokens mismatch — x_gpu.shape[0]="
        f"{x_gpu.shape[0]}, num_tokens arg={num_tokens}"
    )
    # `stride(1) == 1` is the production contract — feature-dim
    # contiguous, possibly row-strided. The C++ side handles the
    # row-strided case via cudaMemcpy2DAsync; transposed/exotic
    # layouts (stride(1) != 1) would need a separate copy plan and
    # are rejected. Real Qwen2-style hidden_states tensors satisfy
    # this even when they come from padded / sliced bases.
    assert x_gpu.stride(1) == 1, (
        f"cots_submit_gemm: x_gpu.stride(1)={x_gpu.stride(1)} (must be 1; "
        f"no transposed-stride layouts in production decode). For "
        f"row-strided inputs (stride(0) > shape[1]) the C++ D2H uses "
        f"cudaMemcpy2DAsync."
    )
    infer = _lookup_infer(runner_id, "cots_submit_gemm")
    stream = torch.cuda.current_stream().cuda_stream
    # §1c.24: NVTX scope so the nsys timeline can attribute the
    # Python-side dispatch boundary separately from the C++ submit
    # body's d2h_record / launch_dispatch_cb sub-ranges. Env-gated
    # (VLLM_COTS_DIAG=1) — diagnostic only, off by default.
    if _COTS_DIAG_ENABLED:
        torch.cuda.nvtx.range_push("cots:py_submit_gemm")
    try:
        # Pass shape/stride so the C++ D2H can dispatch the right
        # cudaMemcpy* variant — see CotsCpuInfer::submit_on_stream for
        # the 1D-vs-2D branch.
        infer.submit_on_stream(
            task_id,
            num_tokens,
            x_gpu.data_ptr(),
            x_gpu.shape[1],
            x_gpu.stride(0),
            x_gpu.stride(1),
            stream,
        )
    finally:
        if _COTS_DIAG_ENABLED:
            torch.cuda.nvtx.range_pop()


def _cots_submit_gemm_fake(
    x_gpu: torch.Tensor,
    runner_id: int,
    task_id: int,
    num_tokens: int,
) -> None:
    """torch.compile tracing: no side effects, just a barrier."""
    return


# --- vllm.cots_sync_then_uva -----------------------------------------------


def _cots_sync_then_uva_impl(
    y_gpu: torch.Tensor,
    gpu_anchor_a: torch.Tensor,
    gpu_anchor_b: torch.Tensor,
    submit_anchor: torch.Tensor,
    runner_id: int,
    task_id: int,
    num_tokens: int,
) -> None:
    """Real impl: schedule the sync host callback then run the UVA copy.

    `gpu_anchor_a` / `gpu_anchor_b` are CUDA tensors that the GPU
    GEMMs produced; mutating them pins this op AFTER both independent
    GEMMs (out_perm, out_pref). `submit_anchor` is `x_gpu` from the
    matching `cots_submit_gemm`; reading it pins this op AFTER submit.

    §1c.20: `y_pinned` is intentionally NOT a parameter. Inductor's
    functionalization materializes any CPU tensor visible in the
    captured graph by allocating a fresh pageable CPU buffer and
    cloning into it (in the worst case via a GPU intermediate +
    blocking GPU→CPU copy that CUDA Graph capture rejects). The slab
    pointer the worker wrote to IS the source of truth; we reach it
    directly via the C++-side `y_pinned_view(task_id, num_tokens)`
    helper. The Python-visible custom-op signature contains only CUDA
    tensors and scalar ids, so Inductor has nothing CPU-side to
    materialize. The trust boundary is install-time: the slab pointer
    came from `_y_pinned`, allocated `pin_memory=True` and validated
    there.
    """
    infer = _lookup_infer(runner_id, "cots_sync_then_uva")
    stream = torch.cuda.current_stream().cuda_stream
    if _COTS_DIAG_ENABLED:
        torch.cuda.nvtx.range_push("cots:py_sync_then_uva")
    try:
        infer.sync_on_stream(stream)
        # Build the CPU view over the slab pointer locally — never escapes
        # back to Python in a way Inductor would see.
        y_pinned = infer.y_pinned_view(task_id, num_tokens)
        infer.note_uva_request(num_tokens, y_pinned.shape[1])
        # Lazy import to avoid a top-level circular import (cots.py imports
        # this module via cots_ops and we'd loop on `from .cots import ...`).
        from vllm.model_executor.offloader.cots import (
            _uva_copy_trusted_host_into_gpu,
        )

        if _COTS_DIAG_ENABLED:
            torch.cuda.nvtx.range_push("cots:py_uva_copy")
        try:
            _uva_copy_trusted_host_into_gpu(y_pinned, y_gpu)
        finally:
            if _COTS_DIAG_ENABLED:
                torch.cuda.nvtx.range_pop()
    finally:
        if _COTS_DIAG_ENABLED:
            torch.cuda.nvtx.range_pop()


def _cots_sync_then_uva_fake(
    y_gpu: torch.Tensor,
    gpu_anchor_a: torch.Tensor,
    gpu_anchor_b: torch.Tensor,
    submit_anchor: torch.Tensor,
    runner_id: int,
    task_id: int,
    num_tokens: int,
) -> None:
    return


# --- registration ----------------------------------------------------------


def register_cots_offloader_ops() -> None:
    """Register the two custom ops. Idempotent at import time."""
    direct_register_custom_op(
        op_name="cots_submit_gemm",
        op_func=_cots_submit_gemm_impl,
        # §1c.20: `mutates_args=["x_gpu"]` only. x_gpu is the CUDA
        # dispatch anchor AND the ordering pin — mutating it forces
        # every subsequent GPU GEMM that reads x_gpu (F.linear
        # permanent / prefetched) to be ordered after submit, AND
        # `cots_sync_then_uva` reads x_gpu as `submit_anchor` to
        # stay ordered after submit. NEITHER `x_pinned` NOR
        # `y_pinned` appears in the op signature; both pinned
        # buffers are reached via slab pointers in C++.
        mutates_args=["x_gpu"],
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


# §1c.21: dump per-runner counters at process exit. Set
# VLLM_COTS_DUMP_COUNTERS=1 to enable. The captured-graph hot-path
# question we want answered is whether num_tokens at submit time is
# stuck at the captured graph-bucket size (e.g., 256) under capture
# vs the live decode count (e.g., 1) under eager. Printing a
# histogram once at process teardown is enough — counters are
# `relaxed` atomics with negligible per-call cost.
def _dump_counters_at_exit() -> None:
    if os.environ.get("VLLM_COTS_DUMP_COUNTERS", "0") != "1":
        return
    if not _COTS_INFER:
        return
    sys.stderr.write("\n[cots §1c.21 counters]\n")
    for rid, infer in _COTS_INFER.items():
        try:
            counters = dict(infer.get_counters())
        except Exception as e:
            sys.stderr.write(f"  runner_id={rid}: get_counters failed: {e}\n")
            continue
        non_zero = {k: v for k, v in counters.items() if v != 0}
        sys.stderr.write(f"  runner_id={rid}:\n")
        for k, v in sorted(non_zero.items()):
            sys.stderr.write(f"    {k}: {v}\n")


atexit.register(_dump_counters_at_exit)
