# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom ops for the COTS native CPU runner — Phase 1c.

Mirrors `prefetch_ops.py`'s pattern: registers two ops via
`direct_register_custom_op` (which lands them under `torch.ops.vllm.*`),
each with a `mutates_args` list that declares barrier-installing
dependencies so torch.compile / CUDA graph capture preserve the overlap
ordering between submit, GPU GEMMs, sync, and UVA copy.

Final schemas after the §1c.20 simplification and §1c.35 dispatch-state
fix (see `phase1c_findings.md §1c.20` / §1c.35):

  * vllm.cots_submit_gemm(x_gpu, runner_id, layer_idx, op_kind_code)
      mutates_args=["x_gpu"]
      x_gpu mutated → pins submit BEFORE every GPU GEMM that reads
      x_gpu, AND provides the (submit → sync) edge consumed by
      cots_sync_then_uva's `submit_anchor` read. The C++ impl
      resolves the current dispatch bucket from OOG runner state,
      resolves the slab task_id from (layer_idx, bucket, op_kind),
      then bundles `cudaMemcpyAsync(D2H)` from x_gpu to
      slab.x_pinned_ptr with the host-callback enqueue.

  * vllm.cots_sync_then_uva(y_gpu, gpu_anchor_a, gpu_anchor_b,
                            submit_anchor, runner_id, layer_idx,
                            op_kind_code)
      mutates_args=["y_gpu", "gpu_anchor_a", "gpu_anchor_b"]
      `gpu_anchor_a/_b` pin sync AFTER each independent GPU GEMM.
      `submit_anchor` (read-only, == x_gpu) pins sync AFTER submit.
      The impl reaches the slab's pinned output via
      `CotsCpuInfer.y_pinned_view(task_id, bucket)`.

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

from vllm.utils.cots_diag import NVTX_ENABLED as _COTS_NVTX_ENABLED
from vllm.utils.torch_utils import direct_register_custom_op

# §1c.26 / §1c.27: UVA-side ablation flag. The C++ side has its
# own `set_ablations(ablate_d2h, ablate_hostfn,
# ablate_submit_hostfn=False, ablate_sync_hostfn=False)` for the
# captured cudaMemcpyAsync (D2H) and cudaLaunchHostFunc (dispatch +
# sync, with the §1c.27 split-side narrow flags); UVA is launched
# from the Python `_cots_sync_then_uva_impl` so its ablation gate
# lives here. Toggled via `set_uva_ablation` at offloader install
# time, only when dry_run + DIAG. Probe-only.
_COTS_ABLATE_UVA: bool = False


def set_uva_ablation(enabled: bool) -> None:
    """§1c.26: enable/disable the UVA captured-kernel ablation.
    Probe-only — only meaningful with dry_run + VLLM_COTS_DIAG=1.
    Called from `CotsOffloader.post_init` after reading the env."""
    global _COTS_ABLATE_UVA
    _COTS_ABLATE_UVA = bool(enabled)


if TYPE_CHECKING:
    # Type-only import; avoids forcing _cots_C at module load on
    # non-CUDA builds.
    from vllm import _cots_C  # noqa: F401

# Module-private infer registry. Strong refs (NOT weak) — the registry
# IS the storage for the `CotsCpuInfer` instance. NativeCotsRunner's
# `__del__` / `close()` is the only thing that removes entries; if a
# runner is GC'd without close() the __del__ unregisters there.
_COTS_INFER: dict[int, Any] = {}
# §1c.33: parallel registry for the per-runner
# `(layer_idx, bucket, op_kind) -> task_id` map. Populated by
# NativeCotsRunner.install via `_register_task_id_for`. Read by
# the §1c.33 atexit dump so per-task fire counts can be
# cross-referenced with their COTS-op descriptors without the
# runner needing to live in EngineCore as a strong reference.
_COTS_TASK_ID_FOR: dict[int, dict[tuple[int, int, str], int]] = {}
_COTS_ACTIVE_DISPATCH: dict[int, tuple[int, int]] = {}
_NEXT_RUNNER_ID = itertools.count(1)

_OP_KIND_TO_CODE: dict[str, int] = {
    "qkv": 1,
    "mlp_block": 2,
}
_OP_KIND_BY_CODE: dict[int, str] = {v: k for k, v in _OP_KIND_TO_CODE.items()}


def op_kind_code(op_kind: str) -> int:
    """Encode a stable op kind for the native custom-op boundary."""
    try:
        return _OP_KIND_TO_CODE[op_kind]
    except KeyError as e:
        raise ValueError(f"unknown COTS op_kind {op_kind!r}") from e


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
    _COTS_TASK_ID_FOR.pop(runner_id, None)
    _COTS_ACTIVE_DISPATCH.pop(runner_id, None)


def _register_task_id_for(
    runner_id: int,
    task_id_for: dict[tuple[int, int, str], int],
) -> None:
    """§1c.33: store the runner's task_id map so the atexit dump
    can cross-reference per-task fire counts with their
    (layer_idx, bucket, op_kind) descriptors."""
    _COTS_TASK_ID_FOR[runner_id] = dict(task_id_for)


def set_active_dispatch_state(
    runner_id: int,
    *,
    bucket: int,
    live_num_tokens: int,
) -> None:
    """Publish the authoritative per-forward COTS dispatch state.

    Called outside the compiled forward by `CotsOffloader.on_dispatch`.
    Native custom ops read this state at eager execution / CUDA Graph
    capture time and resolve slab task_ids from it. This keeps bucket
    and task selection out of compile-visible scalar arguments.
    """
    bucket = int(bucket)
    live_num_tokens = int(live_num_tokens)
    if bucket <= 0:
        raise ValueError(f"active COTS dispatch bucket must be > 0, got {bucket}")
    if live_num_tokens < 0:
        raise ValueError(
            f"active COTS dispatch live_num_tokens must be >= 0, got {live_num_tokens}"
        )
    _COTS_ACTIVE_DISPATCH[int(runner_id)] = (bucket, live_num_tokens)


def _resolve_task_for_dispatch(
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
    op_name: str,
) -> tuple[int, int, int]:
    """Resolve active dispatch state to a concrete C++ slab task."""
    runner_id = int(runner_id)
    state = _COTS_ACTIVE_DISPATCH.get(runner_id)
    if state is None:
        raise RuntimeError(
            f"{op_name}: no active COTS dispatch state for runner_id={runner_id}. "
            "CotsOffloader.on_dispatch must publish the BatchDescriptor bucket "
            "before native custom ops execute or capture."
        )
    bucket, live_num_tokens = state
    try:
        op_kind = _OP_KIND_BY_CODE[int(op_kind_code)]
    except KeyError as e:
        raise RuntimeError(
            f"{op_name}: unknown op_kind_code={op_kind_code} for runner_id={runner_id}"
        ) from e
    task_id_for = _COTS_TASK_ID_FOR.get(runner_id)
    if task_id_for is None:
        raise RuntimeError(
            f"{op_name}: runner_id={runner_id} has no task_id map; "
            "NativeCotsRunner.install() must complete before dispatch."
        )
    key = (int(layer_idx), int(bucket), op_kind)
    task_id = task_id_for.get(key)
    if task_id is None:
        available = sorted(
            b
            for layer, b, kind in task_id_for
            if layer == int(layer_idx) and kind == op_kind
        )
        raise RuntimeError(
            f"{op_name}: no native COTS slab for key={key}; available buckets "
            f"for layer={int(layer_idx)}, op_kind={op_kind!r}: {available}. "
            "The resolved task bucket must come from the OOG dispatch state."
        )
    return int(task_id), int(bucket), int(live_num_tokens)


def _bounded_transfer_rows(bucket: int, tensor_rows: int, op_name: str) -> int:
    """Rows to copy for the active slab without reading past the tensor.

    The active dispatch bucket selects the native slab/task. In FULL
    graph replay, the CUDA tensor may be a larger persistent buffer, so
    the bucket limits transfer work. In eager/profile dummy paths, the
    slab can be the max-capacity fallback while the tensor is smaller,
    so the concrete tensor rows limit transfer work.
    """
    rows = min(int(bucket), int(tensor_rows))
    if rows <= 0:
        raise AssertionError(
            f"{op_name}: transfer row count must be > 0, got "
            f"bucket={bucket}, tensor_rows={tensor_rows}"
        )
    return rows


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
    max_num_tokens: int,
) -> None:
    """Allocate the C++ slab pool. Called once at offloader post_init."""
    infer = _lookup_infer(runner_id, "install_infer")
    infer.install(
        n_slabs=int(n_slabs),
        max_num_tokens=int(max_num_tokens),
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


def install_wait_kernel_sync_for_all_tasks(
    runner_id: int,
    n_slabs: int,
    *,
    fuse_uva_copy: bool = False,
) -> None:
    """§1c.29 commit 2: install wait-kernel sync (host-mapped pinned req/done
    slots, lazy diag-counter alloc when VLLM_COTS_DIAG=1) for
    every slab in the pool. Called from `CotsOffloader.post_init`
    only when `cots_capture_sync_mode="wait_kernel"`.

    Idempotent only at the offloader level — calling
    `install_wait_kernel_sync_for_task` twice for the same task_id raises
    (idempotency violation; see §1c.29 design). The offloader
    holds a single `_wait_kernel_sync_installed` flag to make sure this helper
    runs once per offloader.
    """
    infer = _lookup_infer(runner_id, "install_wait_kernel_sync_for_all_tasks")
    for tid in range(int(n_slabs)):
        infer.install_wait_kernel_sync_for_task(
            int(tid), fuse_uva_copy=bool(fuse_uva_copy)
        )


def set_worker_affinity(runner_id: int, mask: int) -> None:
    """Pin the worker thread to a CPU set (uint64 bitmask). One-shot
    call from `CotsOffloader.post_init` after install."""
    infer = _lookup_infer(runner_id, "set_worker_affinity")
    infer.set_worker_affinity(int(mask))


def dump_task_resolved_fire_counts(
    runner_id: int,
    task_id_for: dict[tuple[int, int, str], int],
) -> list[dict]:
    """§1c.33: per-task fire counts cross-referenced with the
    runner's `(layer_idx, bucket, op_kind) -> task_id` map.

    Returns one record per (layer_idx, bucket, op_kind) in
    sorted order. Each record:
      {
        "task_id": int,
        "layer_idx": int,
        "bucket": int,
        "op_kind": str,
        "fire_count": int,
      }

    Caller is responsible for resetting counters via
    `infer.reset_counters()` to define the measurement window.
    The fire-count counter is single relaxed atomic so per-fire
    cost is ~1 ns; safe to leave always-on.
    """
    infer = _lookup_infer(runner_id, "dump_task_resolved_fire_counts")
    raw = list(infer.get_task_fire_counts())  # type: ignore[attr-defined]
    inverse: dict[int, tuple[int, int, str]] = {
        tid: desc for desc, tid in task_id_for.items()
    }
    records = []
    for tid in range(len(raw)):
        layer, bucket, op_kind = inverse.get(tid, (-1, -1, "unknown"))
        records.append(
            {
                "task_id": tid,
                "layer_idx": layer,
                "bucket": bucket,
                "op_kind": op_kind,
                "fire_count": int(raw[tid]),
            }
        )
    return records


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


def set_live_num_tokens(runner_id: int, n: int) -> None:
    """Publish the live-token row cap to the native COTS worker.

    `n` is the number of semantically live rows in the active bucket.
    Slab capacity, graph capture, and buffer sizing stay bucket-based;
    the worker reads this value on the next host-callback fire and
    avoids CPU GEMM work for padded rows.
    """
    infer = _COTS_INFER.get(runner_id)
    if infer is None:
        # Best-effort: a stale runner_id call here shouldn't crash —
        # just skip. The next custom-op call will surface the missing
        # registry entry with a clearer error.
        return
    infer.set_runtime_num_tokens(int(n))


def set_runtime_num_tokens(runner_id: int, n: int) -> None:
    """Legacy alias for `set_live_num_tokens`."""
    set_live_num_tokens(runner_id, n)


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
    layer_idx: int,
    op_kind_code: int,
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
    pointer to C++. The slab task is selected by the OOG dispatch
    bucket. The transfer row count is bounded by the concrete tensor
    rows as well, because eager/profile dummy runs can route through a
    max-capacity slab while executing a smaller tensor.
    """
    if not x_gpu.is_cuda:
        raise RuntimeError(
            f"cots_submit_gemm: x_gpu must be on CUDA, got {x_gpu.device}"
        )
    if x_gpu.dtype != torch.bfloat16:
        raise RuntimeError(
            f"cots_submit_gemm: x_gpu must be bfloat16 (matches slab dtype), "
            f"got {x_gpu.dtype}"
        )
    if x_gpu.dim() != 2:
        raise RuntimeError(
            f"cots_submit_gemm: x_gpu must be 2D (num_tokens, in_dim); "
            f"got shape {tuple(x_gpu.shape)}"
        )
    task_id, bucket, _live_num_tokens = _resolve_task_for_dispatch(
        runner_id, layer_idx, op_kind_code, "cots_submit_gemm"
    )
    num_transfer_rows = _bounded_transfer_rows(
        bucket, int(x_gpu.shape[0]), "cots_submit_gemm"
    )
    # `stride(1) == 1` is the production contract — feature-dim
    # contiguous, possibly row-strided. The C++ side handles the
    # row-strided case via cudaMemcpy2DAsync; transposed/exotic
    # layouts (stride(1) != 1) would need a separate copy plan and
    # are rejected. Real Qwen2-style hidden_states tensors satisfy
    # this even when they come from padded / sliced bases.
    if x_gpu.stride(1) != 1:
        raise RuntimeError(
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
    if _COTS_NVTX_ENABLED:
        torch.cuda.nvtx.range_push("cots:py_submit_gemm")
    try:
        # Pass shape/stride so the C++ D2H can dispatch the right
        # cudaMemcpy* variant — see CotsCpuInfer::submit_on_stream for
        # the 1D-vs-2D branch.
        infer.submit_on_stream(
            task_id,
            num_transfer_rows,
            x_gpu.data_ptr(),
            x_gpu.shape[1],
            x_gpu.stride(0),
            x_gpu.stride(1),
            stream,
        )
    finally:
        if _COTS_NVTX_ENABLED:
            torch.cuda.nvtx.range_pop()


def _cots_submit_gemm_fake(
    x_gpu: torch.Tensor,
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
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
    layer_idx: int,
    op_kind_code: int,
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
    directly via the C++-side `y_pinned_view(task_id,
    num_transfer_rows)` helper. The Python-visible custom-op signature
    contains only CUDA tensors and scalar ids, so Inductor has nothing
    CPU-side to materialize. The trust boundary is install-time: the
    slab pointer came from `_y_pinned`, allocated `pin_memory=True` and
    validated there.
    """
    task_id, bucket, _live_num_tokens = _resolve_task_for_dispatch(
        runner_id, layer_idx, op_kind_code, "cots_sync_then_uva"
    )
    num_transfer_rows = _bounded_transfer_rows(
        bucket, int(y_gpu.shape[0]), "cots_sync_then_uva"
    )
    infer = _lookup_infer(runner_id, "cots_sync_then_uva")
    stream = torch.cuda.current_stream().cuda_stream
    if _COTS_NVTX_ENABLED:
        torch.cuda.nvtx.range_push("cots:py_sync_then_uva")
    try:
        # §1c.29 commit 2: unified entry. C++ side branches per-slab
        # on `slab.wait_kernel_sync_installed` — when wait-kernel sync
        # was installed at offloader
        # post_init under `cots_capture_sync_mode="wait_kernel"`, the captured
        # node is the wait kernel reading the worker-published
        # `done_slot=seq`. Otherwise the captured node stays the
        # legacy SyncCallback host_fn that blocks the driver thread
        # on TaskQueue::sync(0). Python ALWAYS calls this entry —
        # the A/B is controlled exclusively by whether the
        # offloader installed wait-kernel sync for this task at startup.
        fused_uva = False
        if not _COTS_ABLATE_UVA and hasattr(
            infer, "sync_or_wait_and_maybe_uva_on_stream"
        ):
            fused_uva = bool(
                infer.sync_or_wait_and_maybe_uva_on_stream(
                    task_id,
                    y_gpu.data_ptr(),
                    num_transfer_rows,
                    y_gpu.shape[1],
                    stream,
                )
            )
        else:
            infer.sync_or_wait_on_stream(task_id, stream)
        if fused_uva:
            return
        # Build the CPU view over the slab pointer locally — never escapes
        # back to Python in a way Inductor would see.
        y_pinned = infer.y_pinned_view(task_id, num_transfer_rows)
        infer.note_uva_request(num_transfer_rows, y_pinned.shape[1])
        # Lazy import to avoid a top-level circular import (cots.py imports
        # this module via cots_ops and we'd loop on `from .cots import ...`).
        from vllm.model_executor.offloader.cots import (
            _uva_copy_trusted_host_into_gpu,
        )

        if _COTS_NVTX_ENABLED:
            torch.cuda.nvtx.range_push("cots:py_uva_copy")
        try:
            # §1c.26 ablation: skip the captured Triton UVA kernel
            # entirely. Probe-only; gated upstream to dryrun + DIAG.
            # y_gpu is left with stale data; harmless in dryrun
            # because downstream consumers don't validate output.
            if not _COTS_ABLATE_UVA:
                _uva_copy_trusted_host_into_gpu(y_pinned, y_gpu)
        finally:
            if _COTS_NVTX_ENABLED:
                torch.cuda.nvtx.range_pop()
    finally:
        if _COTS_NVTX_ENABLED:
            torch.cuda.nvtx.range_pop()


def _cots_sync_then_uva_fake(
    y_gpu: torch.Tensor,
    gpu_anchor_a: torch.Tensor,
    gpu_anchor_b: torch.Tensor,
    submit_anchor: torch.Tensor,
    runner_id: int,
    layer_idx: int,
    op_kind_code: int,
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


def _dump_task_fire_counts_at_exit() -> None:
    """§1c.33: dump per-task fire counts, cross-referenced with
    each (layer_idx, bucket, op_kind) descriptor, at process exit.
    Gated by `VLLM_COTS_DUMP_TASK_FIRES=1`. Optionally write to
    a file via `VLLM_COTS_DUMP_TASK_FIRES_FILE=/path/to.json`;
    otherwise dump to stderr alongside the standard counters.

    The file output is an auditable artifact for native COTS dispatch
    accounting. Per-task fires let us see which
    (layer, bucket, op_kind) tuples execute during capture/replay.
    """
    if os.environ.get("VLLM_COTS_DUMP_TASK_FIRES", "0") != "1":
        return
    if not _COTS_INFER:
        return
    file_path = os.environ.get("VLLM_COTS_DUMP_TASK_FIRES_FILE", "").strip()
    out_records: dict[str, list[dict]] = {}
    for rid in list(_COTS_INFER.keys()):
        task_id_for = _COTS_TASK_ID_FOR.get(rid)
        if task_id_for is None:
            # Runner registered an infer but never reached install;
            # nothing to attribute.
            continue
        try:
            out_records[f"runner_{rid}"] = dump_task_resolved_fire_counts(
                rid, task_id_for
            )
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(
                f"[cots §1c.33] runner_id={rid}: "
                f"dump_task_resolved_fire_counts failed: {e}\n"
            )
            continue
    if not out_records:
        return
    if file_path:
        try:
            import json as _json

            with open(file_path, "w") as fh:
                _json.dump(out_records, fh, indent=2, default=str)
            sys.stderr.write(f"[cots §1c.33] per-task fire counts → {file_path}\n")
        except OSError as e:
            sys.stderr.write(
                f"[cots §1c.33] write to {file_path} failed: {e}; "
                f"falling back to stderr\n"
            )
            file_path = ""
    if not file_path:
        sys.stderr.write("\n[cots §1c.33 per-task fire counts]\n")
        for runner_key, records in out_records.items():
            sys.stderr.write(f"  {runner_key}:\n")
            for rec in records:
                if rec["fire_count"] == 0:
                    continue
                sys.stderr.write(
                    f"    task_id={rec['task_id']:>3}  "
                    f"layer={rec['layer_idx']:>2}  "
                    f"bucket={rec['bucket']:>4}  "
                    f"op_kind={rec['op_kind']:<11}  "
                    f"fires={rec['fire_count']}\n"
                )


atexit.register(_dump_task_fire_counts_at_exit)
